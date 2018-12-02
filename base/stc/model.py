import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from xslu.modules import Embedding, EncoderRNN, Attention
import xslu.Constants as Constants

class RNN2One(nn.Module):
    """
    Use the last hidden vectors of the last layer of the rnn as input to the final classifier.
    """

    def __init__(self, vocab_size, class_size, emb_dim=100, rnn_type='LSTM', 
            bidirectional=True, hid_dim=128, num_layers=1, dropout=0.5):

        super(RNN2One, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.lin_dim = num_directions * hid_dim

        self.emb = Embedding(vocab_size, emb_dim, Constants.PAD, dropout)
        self.rnn = EncoderRNN(rnn_type, bidirectional, num_layers, hid_dim, self.emb, dropout)

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(self.lin_dim, class_size)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.lin.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, data, lengths):
        
        batch_size = data.size(0)
        outputs, hiddens = self.rnn(data, lengths)
        if self.rnn_type == 'LSTM':
            h_t = hiddens[0]
        elif self.rnn_type == 'GRU':
            h_t = hiddens
        
        if self.bidirectional:
            index_slices = [self.num_layers * 2 - 2, self.num_layers * 2 - 1]
        else:
            index_slices = [self.num_layers * 1 - 1]
        index_slices = torch.tensor(index_slices, dtype=torch.long, device=data.device)
        h_t = torch.index_select(h_t, 0, index_slices)
        h_t = h_t.transpose(0, 1).contiguous().view(batch_size, self.lin_dim)
        h_t = self.dropout(h_t)

        scores = torch.sigmoid(self.lin(h_t))
        
        return scores
