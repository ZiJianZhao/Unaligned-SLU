import torch
import torch.nn as nn

import xslu.Constants as Constants
from xslu.utils import process_sent
from dstc3.text.text_da import process_class

from dataloader import  DADataset

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

def beam_search(model, extra_zeros, enc_batch_extend_vocab_idx,
        init_state, enc_outputs, enc_lengths, vocab_size, cuda,
        beam_size = 10, max_steps=20, nbest=3):

    def sort_beams(beams):
        return sorted(beams, key = lambda b: b.avg_log_prob, reverse=True)

    h, c = init_state  # 1 * 1 * hid_dim

    beams = [Beam(tokens = [Constants.BOS], log_probs = [0.0], state = (h, c))
            for _ in range(1)]

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

        enc_outputs = enc_outputs.expand(y_t.size(0), -1, -1)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.expand(y_t.size(0), -1)
        if enc_batch_extend_vocab_idx is not None:
            enc_batch_extend_vocab_idx = enc_batch_extend_vocab_idx.expand(y_t.size(0), -1)

        dist_t, s_t = model(
            y_t, s_t_1, enc_outputs, enc_lengths,
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

    res = []
    for i in range(min(len(results), nbest)):
        res.append(results[i].tokens)

    return res

def decode_utterance(model, class_string, memory, cuda, nbest):

    class_string = process_class(class_string)
    sent_lis = process_sent(class_string)
    if len(sent_lis) == 0:
        return ['']

    data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
            DADataset.data_info(class_string, memory, cuda)

    # Model processing
    ## encoder
    outputs, hiddens = model.encode(data, lengths)

    s_decoder = model.enc_to_dec(hiddens)
    s_t_1 = s_decoder
    y_t = torch.tensor([Constants.BOS]).view(1, 1)
    if cuda:
        y_t = y_t.cuda()
    out_lis = beam_search(model.decoder,
        extra_zeros, enc_batch_extend_vocab_idx,
        s_decoder, outputs, lengths,
        len(memory['dec2idx']), cuda,
        nbest=nbest
    )
    res = [[] for _ in range(len(out_lis))]
    for i in range(len(out_lis)):
        out_ids = out_lis[i][1:-1]
        for vid in out_ids:
            if vid < len(memory['idx2dec']):
                res[i].append(memory['idx2dec'][vid])
            else:
                res[i].append(oov_list[vid - len(memory['idx2dec'])])
    utts = []
    for out_lis in res:
        utterance = ' '.join(out_lis)
        utts.append(utterance)

    return utts

