# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xslu.modules.attention import Attention

class Embedding(nn.Module):
    r"""
    Applies a embedding layer to an input sequence.

    Args:
        vocab_size (int): the size of the vocab
        emb_dim (int): the dimensinality of each embedding vector
        padding_idx (int): pad the output with zeros whenever it encounters the index.
    """

    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):

        super(Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx, sparse=False)
        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def forward(self, input):
        """
        Applies a embedding layer to an input sequence.

        Args:
            input (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **output** (batch, seq_len, emb_dim): tensor containing embedding of the input sequence
        """
        emb = self.embedding(input)
        emb = self.dropout(emb)
        return emb

    def init_params(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                #nn.init.normal_(param, 0, 0.01)
                nn.init.uniform_(param, -0.1, 0.1)
            elif name.endswith('bias'):
                nn.init.constant_(param, 0)
            else:
                raise Exception('Wrong parameters')

    def init_weight_from_pre_emb(self, emb, fix_emb):
        for name, param in self.named_parameters():
            if name.endswith('weight') and param.data.size() == emb.size():
                emb = emb.to(param.data.device)
                param.data = emb
                if fix_emb:
                    param.requires_grad = False
