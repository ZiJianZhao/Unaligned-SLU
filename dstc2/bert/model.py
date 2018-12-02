# -*- codind: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class BertSTC(nn.Module):
    
    def __init__(self, mode, bert, hid_dim, class_size, dropout=0.5):

        super(BertSTC, self).__init__()

        self.mode = mode
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hid_dim, class_size)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.lin.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                output_all_encoded_layers=False)
        
        if self.mode == 'first':
            vector = pooled_output
        elif self.mode == 'max':
            vector, _ = torch.max(encoder_layers, dim=1)
        elif self.mode == 'avg':
            vector = torch.sum(encoder_layers, dim=1) / encoder_layers.size(1)
        else:
            raise Exception('Undefined bert mode.')

        vector = self.dropout(vector)
        scores = torch.sigmoid(self.lin(vector))

        return scores

