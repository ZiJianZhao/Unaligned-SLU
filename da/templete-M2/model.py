import os, sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from xslu.modules import Embedding, EncoderRNN, Attention
import xslu.Constants as Constants

class DAModel(nn.Module):
    """Embeddings for SLU"""

    def __init__(self, enc_word_vocab_size, dec_word_vocab_size, emb_dim, hid_dim, dropout):

        super(DAModel, self).__init__()

        # Initialization
        self.hid_dim = hid_dim

        # embedding
        self.enc_word_emb = Embedding(enc_word_vocab_size, emb_dim, Constants.PAD, dropout)
        self.dec_word_emb = Embedding(dec_word_vocab_size, emb_dim, Constants.PAD, dropout)

        # encoder-decoder
        self.encoder = EncoderRNN('LSTM', True, 1, hid_dim, self.enc_word_emb, 0.)
        self.enc_to_dec = Encoder2Decoder(2 * hid_dim, 2 * hid_dim)
        self.decoder = LSTMDecoder(self.dec_word_emb, dec_word_vocab_size, 2 * hid_dim, dropout)

    def encode(self, enc_data, enc_length):
        outputs = []
        hiddens = 0
        content = 0
        for data in enc_data:
            output, hidden = self.encoder(data, None)
            outputs.append(output)
            hiddens += hidden[0]
            content += hidden[1]
        outputs = torch.cat(outputs, dim=1)
        hiddens = (hiddens, content)
        return outputs, hiddens

    def forward(self, enc_data, enc_length, dec_data, extra_zeros, enc_batch_extend_vocab_idx):
        """
        Args meaning:
            - data: utterance word sequence;
            - lengths: None for batch_size = 1;
        Args type:
            - data (tensor): 1 * data_len
            - lengths: None
        Attention:
            - The model is specially for 1 example per batch:
                * So lengths can be None
                * possibly to do n examples per batch (if training speed is too slow, we refine it)
        """

        # utterance representation
        outputs, hiddens = self.encode(enc_data, enc_length)

        s_decoder = self.enc_to_dec(hiddens)
        ctx = outputs
        ctx_lengths = enc_length
        s_t_1 = s_decoder
        dists = []
        for i in range(dec_data.size(1)):
            y_t = dec_data[:, i].unsqueeze(1)
            final_dist, s_t = self.decoder(y_t, s_t_1,
                ctx, ctx_lengths,
                extra_zeros, enc_batch_extend_vocab_idx
            )
            s_t_1 = s_t
            dists.append(final_dist)
        dist = torch.cat(dists, dim=0)

        return dist

class Encoder2Decoder(nn.Module):
    """Initialize states of decoder from outputs of encoder
        Input: hidden states and context vectors from LSTM encoder;
        Output: hidden states and context vectors for LSTM decoder initial states.
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Encoder2Decoder, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.lin_h = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.lin_c = nn.Linear(enc_hid_dim, dec_hid_dim)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, hiddens):
        h = hiddens[0].transpose(0, 1).contiguous().view(-1, self.enc_hid_dim)
        c = hiddens[1].transpose(0, 1).contiguous().view(-1, self.enc_hid_dim)
        #h = hiddens[0][0].view(-1, self.dec_hid_dim)
        #c = hiddens[1][0].view(-1, self.dec_hid_dim)
        h = torch.tanh(self.lin_h(h)).unsqueeze(0)
        c = torch.tanh(self.lin_c(c)).unsqueeze(0)
        return h, c

class LSTMDecoder(nn.Module):
    """
    #LSTM Decoder with attention and copy mechanism implemented by pointer network.

    Inputs:
        - uttenrance representation from a BLSTM encoder.

    Mentions:
        - first RNN forward, then attention calculation, features can be used:
            * concated with embedding each time step influence RNN states;
            * concated with hidden    each time step to directly calculate attention.
        - vocab_size is for decoder input embedding while class_size is for decoder output layer,
            these two values only differ when we use glove word embedding for inputs.
    """
    def __init__(self, word_emb, class_size, hid_dim, dropout):

        super(LSTMDecoder, self).__init__()

        emb_dim = word_emb.emb_dim
        self.class_size = class_size
        self.word_emb = word_emb

        self.rnn = nn.LSTM(emb_dim, hid_dim,
                num_layers = 1, batch_first = True, bidirectional=False
            )
        self.attn_func = Attention('dot', hid_dim, hid_dim)
        # pointer network
        self.pointer_lin = nn.Linear(hid_dim * 2 + emb_dim, 1)

        # softmax output
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(hid_dim * 2, emb_dim)
        self.outlin = nn.Linear(emb_dim, class_size, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, y_t, s_t_1, ctx, ctx_lengths, extra_zeros, enc_batch_extend_vocab_idx):

        # RNN forward
        y_t_emb = self.word_emb(y_t)  # batch * 1 * emb_dim
        lstm_out, s_t = self.rnn(y_t_emb, s_t_1)
        lstm_out = lstm_out.squeeze(1)

        # Attention calculation
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

