import torch
import torch.nn as nn

import xslu.Constants as Constants

from dataloader import  DataLoader

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
        beam_size = 10, max_steps=30):

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
            if b.latest_token == Constants.EOS and step >= 1:
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

def translate(model, src, word2idx, idx2word, cuda):

    sent_lis = src.strip().split()

    if len(sent_lis) == 0:
        return ''

    enc, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
            DataLoader.enc_info(src, word2idx, cuda)

    ## encoder
    ctx, hiddens = model.encoder(enc, lengths)
    s_decoder = model.enc_to_dec(hiddens)

    ## decoder
    dec_ids = beam_search(model.decoder,
        extra_zeros, enc_batch_extend_vocab_idx,
        s_decoder, ctx, lengths,
        len(word2idx), cuda,
        max_steps = 30
    )[1:-1]

    dec_lis = []
    for idx in dec_ids:
        if idx < len(idx2word):
            dec_lis.append(idx2word[idx])
        else:
            dec_lis.append(oov_list[idx - len(idx2word)])
    result = ' '.join(dec_lis)

    return result
