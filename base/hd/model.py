import os, sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xslu.modules import Embedding, EncoderRNN, Attention
import xslu.Constants as Constants

class SLUSystem(nn.Module):
    """Embeddings for SLU"""

    def __init__(self, enc_word_vocab_size, dec_word_vocab_size, act_vocab_size, slot_vocab_size, 
            emb_dim, hid_dim, dropout):

        super(SLUSystem, self).__init__()

        # Initialization
        self.hid_dim = hid_dim

        # embedding
        self.enc_word_emb = Embedding(enc_word_vocab_size, emb_dim, Constants.PAD, dropout)
        self.dec_word_emb = Embedding(dec_word_vocab_size, emb_dim, Constants.PAD, dropout)
        self.act_emb = Embedding(act_vocab_size, emb_dim, Constants.PAD, dropout)
        self.slot_emb = Embedding(slot_vocab_size, emb_dim, Constants.PAD, dropout)

        # shared encoder
        self.encoder_bidirectional = True
        if self.encoder_bidirectional:
            self.enc_hid_all_dim = 2 * hid_dim
        else:
            self.enc_hid_all_dim = hid_dim
        self.encoder = EncoderRNN('LSTM', self.encoder_bidirectional, 
                1, hid_dim, self.enc_word_emb, 0.)

        # decoder for auto-encoder training
        #self.auto_decoder = SeqDecoder(self.word_emb, word_vocab_size, hid_dim, dropout)

        # act-slot-value predictors
        self.act_stc = STC(self.enc_hid_all_dim, act_vocab_size, dropout, emb_dim)
        self.slot_stc = STC(self.enc_hid_all_dim + emb_dim, slot_vocab_size, dropout, emb_dim)
        self.enc_to_dec = Encoder2Decoder(self.enc_hid_all_dim, hid_dim)

        # the hid dim of decoder is 2 * hid_dim
        self.value_decoder = LSTMDecoder(self.dec_word_emb, self.act_emb, self.slot_emb,
                dec_word_vocab_size, hid_dim, dropout)

    def slot_predict(self, h_T, acts):
        act_nums = acts.size(0)  # act inputs are in size (act_num, 1)
        act_embs = self.act_emb(acts)  # act_num * 1 * emb_dim
        act_embs_4_slot = act_embs.squeeze(1)
        h_T_4_slot = h_T.expand(act_nums, -1)
        input_4_slot = torch.cat([act_embs_4_slot, h_T_4_slot], dim=1)
        slot_scores = self.slot_stc(input_4_slot)
        return slot_scores

    def auto_encoder_forward(self, data, lengths, values):
        """Attention: values here is a cuda tensor"""

        outputs, hiddens = self.encoder(data, lengths)
        s_decoder = self.enc_to_dec(hiddens)
        probs = self.auto_decoder(values, s_decoder)

        return probs

    def forward(self, data, lengths, acts, act_slot_pairs, values, extra_zeros, enc_batch_extend_vocab_idx):
        """
        Args meaning:
            - data: utterance word sequence;
            - lengths: None for batch_size = 1;
            - acts: acts need slots, may be None if no acts of the utterance need slots;
            - act_slot_pairs: act_slot need values, may be None, if value is not needed.
        Args type:
            - data (tensor): 1 * data_len
            - lengths: None
            - acts (tensor): act_num * 1
            - act_slot_pairs (list): containing tensors of size 1 * 2 for act_slot ids
            - values (list): containing tensors of size 1 * value_len for value ids
                * [<s>, values]
        Attention:
            - The model is specially for 1 example per batch:
                * So lengths can be None
                * possibly to do n examples per batch (if training speed is too slow, we refine it)
        """

        # utterance representation
        outputs, hiddens = self.encoder(data, lengths)
        h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, self.enc_hid_all_dim)

        # act prediction
        act_scores = self.act_stc(h_T)

        if acts is None:
            return act_scores, None, None

        # slot prediction
        slot_scores = self.slot_predict(h_T, acts)

        if act_slot_pairs is None:
            return act_scores, slot_scores, None

        # value prediction
        s_decoder = self.enc_to_dec(hiddens)
        ctx = outputs
        ctx_lengths = lengths
        assert len(act_slot_pairs) == len(values)
        pair_num = len(act_slot_pairs)
        dists = [ []  for _ in range(pair_num)]
        for pr in range(pair_num):
            act_slot = act_slot_pairs[pr]
            value = values[pr]
            s_t_1 = s_decoder
            for vl in range(value.size(1)):
                y_t = value[:, vl].unsqueeze(1)
                final_dist, s_t = self.value_decoder(y_t, s_t_1, 
                        ctx, ctx_lengths, act_slot, 
                        extra_zeros, enc_batch_extend_vocab_idx
                        )
                s_t_1 = s_t
                dists[pr].append(final_dist)
        values_dist_lis = [torch.cat(ds, dim=0) for ds in dists]

        return act_scores, slot_scores, values_dist_lis

class STC(nn.Module):
    """Semantic Tuple Classifier
    """

    def __init__(self, input_dim, class_size, dropout, inter_dim=None):

        super(STC, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.non_lin = nn.ReLU()
        #self.non_lin = nn.Tanh()
        self.inter_dim = inter_dim
        if self.inter_dim is not None:
            self.inter_lin = nn.Linear(input_dim, inter_dim)
            self.lin = nn.Linear(inter_dim, class_size, bias=True)
        else:
            self.lin = nn.Linear(input_dim, class_size, bias=True)
        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, vec):
        
        vec = self.dropout(vec)
        if self.inter_dim is not None:
            vec = self.inter_lin(vec)
            vec = self.non_lin(vec)
            logits = self.lin(vec)
        else:
            logits = self.lin(vec)
        scores = torch.sigmoid(logits)
        
        return scores

class Encoder2Decoder(nn.Module):
    """Initialize states of decoder from outputs of encoder
        Input: hidden states and context vectors from LSTM encoder;
        Output: hidden states and context vectors for LSTM decoder initial states.
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Encoder2Decoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.lin_h = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.lin_c = nn.Linear(dec_hid_dim, dec_hid_dim)
        
        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, hiddens):
        #h = hiddens[0].transpose(0, 1).contiguous().view(-1, self.enc_hid_dim)
        #c = hiddens[1].transpose(0, 1).contiguous().view(-1, self.enc_hid_dim)
        h = hiddens[0][0].view(-1, self.dec_hid_dim)
        c = hiddens[1][0].view(-1, self.dec_hid_dim)
        h = torch.tanh(self.lin_h(h)).unsqueeze(0)
        c = torch.tanh(self.lin_c(c)).unsqueeze(0)
        return h, c

class LSTMDecoder(nn.Module):
    """
    #LSTM Decoder with attention and copy mechanism implemented by pointer network.

    Inputs:
        - uttenrance representation from a BLSTM encoder.
        - act-slot pairs: concated label embeddings for each uttenrance. 

    Mentions:
        - first RNN forward, then attention calculation, features can be used:
            * concated with embedding each time step influence RNN states;
            * concated with hidden    each time step to directly calculate attention.
        - vocab_size is for decoder input embedding while class_size is for decoder output layer,
            these two values only differ when we use glove word embedding for inputs.
    """
    def __init__(self, word_emb, act_emb, slot_emb, class_size, hid_dim, dropout):

        super(LSTMDecoder, self).__init__()

        emb_dim = word_emb.emb_dim
        fea_dim = act_emb.emb_dim + slot_emb.emb_dim
        self.fea_dim = fea_dim
        self.class_size = class_size

        # Feature embedding
        self.word_emb = word_emb
        self.act_emb = act_emb
        self.slot_emb = slot_emb

        # RNN forward
        """
        self.rnn = nn.LSTM(emb_dim + fea_dim, hid_dim, 
                num_layers = 1, batch_first = True, bidirectional=False
            )
        self.attn_func = Attention('dot', hid_dim, hid_dim)
        """
        self.rnn = nn.LSTM(emb_dim, hid_dim, 
                num_layers = 1, batch_first = True, bidirectional=False
            )
        self.attn_lin = nn.Linear(hid_dim + fea_dim, 2 * hid_dim)
        self.attn_func = Attention('dot', 2 * hid_dim, 2 * hid_dim)
        # pointer network
        self.pointer_lin = nn.Linear(hid_dim * 4 + emb_dim, 1)

        # softmax output
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(hid_dim * 4, emb_dim)
        self.outlin = nn.Linear(emb_dim, class_size, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, y_t, s_t_1, ctx, ctx_lengths, act_slot_pairs,
            extra_zeros, enc_batch_extend_vocab_idx):
        
        # act_slot_pairs embedding
        pair_num = act_slot_pairs.size(0) # pairs are in size (pair_num, 2)
        acts = act_slot_pairs[:, 0].unsqueeze(1)
        slots = act_slot_pairs[:, 1].unsqueeze(1)

        # Feature embedding
        acts_emb = self.act_emb(acts)
        slots_emb = self.slot_emb(slots)
        fea_vec = torch.cat([acts_emb, slots_emb], 2)  # (pair_num, 1, 2)

        # RNN forward
        y_t_emb = self.word_emb(y_t)  # batch * 1 * emb_dim
        """
        lstm_inp = torch.cat([y_t_emb, fea_vec], 2)
        """
        lstm_inp = y_t_emb
        lstm_out, s_t = self.rnn(lstm_inp, s_t_1)
        lstm_out = lstm_out.squeeze(1)

        # Attention calculation
        lstm_out = self.attn_lin(torch.cat([lstm_out, fea_vec.squeeze(1)], dim=1))
        attn_dist, c_t = self.attn_func(lstm_out, ctx, ctx, ctx_lengths)

        # pointer ratio
        pointer_input = torch.cat([y_t_emb.squeeze(1), lstm_out, c_t], 1)
        pointer_input = self.dropout(pointer_input)
        pointer_ratio = torch.sigmoid(self.pointer_lin(pointer_input))

        # softmax output
        output = torch.cat([lstm_out, c_t], 1)
        output = self.lin1(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.outlin(output)
        vocab_dist = self.softmax(output)

        # combine pointer and softmax
        vocab_dist_v = pointer_ratio * vocab_dist
        attn_dist_v = (1 - pointer_ratio) * attn_dist
        if extra_zeros is not None:
            vocab_dist_v = torch.cat([vocab_dist_v, extra_zeros], 1)
        final_dist = vocab_dist_v.scatter_add_(1, enc_batch_extend_vocab_idx, attn_dist_v)
        final_dist = torch.log(final_dist + 1e-12)

        return final_dist, s_t

