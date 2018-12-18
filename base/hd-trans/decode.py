import torch
import torch.nn as nn

import xslu.Constants as Constants
from text.dstc2 import process_sent

from dataloader import  ActDataset, SlotDataset, ValueDataset, SLUDataset

class Beam(object):

    def __init__(self, tokens, log_probs, state):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state

    def extend(self, token, log_prob, state):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            state = state
        )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

def beam_search(model, act_slot, extra_zeros, enc_batch_extend_vocab_idx,
        init_state, enc_outputs, enc_lengths, vocab_size, cuda,
        beam_size = 10, max_steps=5):

    def sort_beams(beams):
        return sorted(beams, key = lambda b: b.avg_log_prob, reverse=True)

    h, c = init_state  # 1 * 1 * hid_dim

    beams = [Beam(tokens = [Constants.BOS], log_probs = [0.0], state = (h, c)) 
            for _ in range(beam_size)]

    results = []
    step = 1
    while (step < max_steps) and (len(results) < beam_size):
        
        latest_tokens = [b.latest_token for b in beams]
        latest_tokens = [t if t < vocab_size else Constants.UNK for t in latest_tokens]
        y_t = torch.LongTensor(latest_tokens).view(-1, 1)
        if cuda: y_t = y_t.cuda()
        
        hs = [b.state[0] for b in beams]
        cs = [b.state[1] for b in beams]
        s_t_1 = (torch.cat(hs, dim=1), torch.cat(cs, dim=1))

        act_slot = act_slot.expand(y_t.size(0), -1)
        enc_outputs = enc_outputs.expand(y_t.size(0), -1, -1)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.expand(y_t.size(0), -1)
        if enc_batch_extend_vocab_idx is not None:
            enc_batch_extend_vocab_idx = enc_batch_extend_vocab_idx.expand(y_t.size(0), -1)

        dist_t, s_t = model(
            y_t, s_t_1, enc_outputs, enc_lengths, act_slot,
            extra_zeros, enc_batch_extend_vocab_idx
        )

        topk_log_probs, topk_ids = torch.topk(dist_t, beam_size * 2)

        all_beams = []
        for i in range(len(beams)):
            b = beams[i]
            state = (s_t[0][:, i:(i+1), :], s_t[1][:, i:(i+1), :])

            for j in range(beam_size * 2):
                new_beam = b.extend(
                    token = topk_ids[i, j].item(),
                    log_prob = topk_log_probs[i, j].item(),
                    state = state
                )
                all_beams.append(new_beam)

        all_beams = sort_beams(all_beams)
        beams = []
        for b in all_beams:
            if b.latest_token == Constants.EOS and step > 1:
                results.append(b)
            else:
                beams.append(b)
            if len(beams) == beam_size or len(results) == beam_size:
                break

        step += 1
    
    if len(results) == 0:
        results = beams

    results = sort_beams(results)

    return results[0].tokens

def decode_act(model, utterance, memory, cuda):

    sent_lis = process_sent(utterance)
    if len(sent_lis) == 0:
        return []

    data, lengths = ActDataset.data_info(utterance, memory, cuda)

    # Model processing
    ## encoder
    outputs, hiddens = model.encoder(data, lengths)
    #h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, model.enc_hid_all_dim)
    h_T = hiddens[0]

    ## act prediction
    act_scores = model.act_stc(h_T)
    act_scores = act_scores.data.cpu().view(-1,).numpy()
    pred_acts = [i for i,p in enumerate(act_scores) if p > 0.5]
    acts = [memory['idx2act'][i] for i in pred_acts]

    return acts

def decode_slot(model, utterance, class_string, memory, cuda):

    sent_lis = process_sent(utterance)
    if len(sent_lis) == 0:
        return []

    data, lengths = SlotDataset.data_info(utterance, memory, cuda)
    act_inputs, slot_label = SlotDataset.label_info(class_string, memory, cuda)

    # Model processing
    ## encoder
    outputs, hiddens = model.encoder(data, lengths)
    #h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, model.enc_hid_all_dim)
    h_T = hiddens[0]

    ## slot prediction
    slot_scores = model.slot_predict(h_T, act_inputs)
    slot_scores = slot_scores.data.cpu().view(-1,).numpy()
    pred_slots = [i for i,p in enumerate(slot_scores) if p > 0.5]
    slots = [memory['idx2slot'][i] for i in pred_slots]

    return slots

def decode_value(model, utterance, class_string, memory, cuda):

    sent_lis = process_sent(utterance)
    if len(sent_lis) == 0:
        return []

    data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
            ValueDataset.data_info(utterance, memory, cuda)
    act_inputs, act_slot_pairs, values_inp, values_out = \
            ValueDataset.label_info(class_string, memory, oov_list, cuda)

    # Model processing
    ## encoder
    outputs, hiddens = model.encoder(data, lengths)
    #h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, model.enc_hid_all_dim)
    h_T = hiddens[0]

    ## value decoder
    s_decoder = model.enc_to_dec(hiddens)
    s_t_1 = s_decoder
    act_slot_ids = act_slot_pairs[0]
    y_t = torch.tensor([Constants.BOS]).view(1, 1)
    if cuda: 
        y_t = y_t.cuda()
    value_ids = beam_search(model.value_decoder, 
        act_slot_ids, extra_zeros,enc_batch_extend_vocab_idx, 
        s_decoder, outputs, lengths, 
        len(memory['dec2idx']), cuda
    )[1:-1]
    value_lis = []
    for vid in value_ids:
        if vid < len(memory['idx2dec']):
            value_lis.append(memory['idx2dec'][vid])
        else:
            value_lis.append(oov_list[vid - len(memory['idx2dec'])])
    values = [' '.join(value_lis)]

    return values

def decode_slu(model, utterance, memory, cuda):

    sent_lis = process_sent(utterance)

    if len(sent_lis) == 0:
        return []

    data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
            SLUDataset.data_info(utterance, memory, cuda)

    act_slot_values = []

    # Model processing

    ## encoder
    outputs, hiddens = model.encoder(data, lengths)
    #h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, model.enc_hid_all_dim)
    h_T = hiddens[0]

    ## act prediction
    act_scores = model.act_stc(h_T)
    act_scores = act_scores.data.cpu().view(-1,).numpy()
    pred_acts = [i for i,p in enumerate(act_scores) if p > 0.5]
    act_pairs = [(i, memory['idx2act'][i]) for i in pred_acts]
    remain_acts = []
    for act in act_pairs:
        if act[1] == 'pad':
            continue
        elif act[1] in memory['single_acts']:
            act_slot_values.append(act[1])
        else:
            remain_acts.append(act)

    if len(remain_acts) == 0:
        return act_slot_values

    ## slot prediction
    remain_act_slots = []
    for act in remain_acts:
        act_input = torch.tensor([act[0]]).view(1,1)
        if cuda:
            act_input = act_input.cuda()
        slot_scores = model.slot_predict(h_T, act_input)
        slot_scores = slot_scores.data.cpu().view(-1,).numpy()
        pred_slots = [i for i,p in enumerate(slot_scores) if p > 0.5]
        slot_pairs = [(i, memory['idx2slot'][i]) for i in pred_slots]
        if act[1] in memory['double_acts']:
            for slot in slot_pairs:
                act_slot_values.append('-'.join([act[1], slot[1]]))
        else:
            for slot in slot_pairs:
                if slot[1] != 'pad':
                    remain_act_slots.append(list(zip(act, slot)))

    if len(remain_act_slots) == 0:
        return act_slot_values

    ## value decoder
    s_decoder = model.enc_to_dec(hiddens)
    for act_slot in remain_act_slots:
        s_t_1 = s_decoder
        act_slot_ids = torch.tensor(act_slot[0]).view(1, 2)
        y_t = torch.tensor([Constants.BOS]).view(1, 1)
        if cuda: 
            y_t = y_t.cuda()
            act_slot_ids = act_slot_ids.cuda()
        value_ids = beam_search(model.value_decoder, 
            act_slot_ids, extra_zeros,enc_batch_extend_vocab_idx, 
            s_decoder, outputs, lengths, 
            len(memory['dec2idx']), cuda
        )[1:-1]
        value_lis = []
        for vid in value_ids:
            if vid < len(memory['idx2dec']):
                value_lis.append(memory['idx2dec'][vid])
            else:
                value_lis.append(oov_list[vid - len(memory['idx2dec'])])
        values = ' '.join(value_lis)
        act_slot_values.append('-'.join(list(act_slot[1]) + [values]))

    return act_slot_values

