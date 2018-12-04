# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xslu.modules.attention import Attention
from xslu.modules.embedding import Embedding

class SASREmbedding(Embedding):
    """Self Attention Weighted Sum Representation Embedding
    
    # Reference: Attention is all you need.

    """
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(SASREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)
        self.attn = Attention('dot', emb_dim, emb_dim)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)
            #ctx = emb * weight
            ctx = emb

            _, attn_output = self.attn(ctx, ctx, ctx, None)

            attn_output = attn_output * weight
            attn_output = attn_output.sum(dim=1)

            tmp_len = max_seq_len - attn_output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(attn_output)
                attn_output = torch.cat([attn_output, tmp], dim=0)

            outputs.append(attn_output)

        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output

class MWSREmbedding(Embedding):
    """Modified Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(MWSREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)
        self.lin = nn.Linear(emb_dim, emb_dim)
        self.qry_emb = nn.Parameter(torch.randn(emb_dim))
        self.attn = Attention('dot', emb_dim, emb_dim)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)
            ctx_values = emb * weight
            ctx_keys = torch.tanh(self.lin(ctx_values))

            qry = self.qry_emb.view(1, -1).repeat(seq_len, 1)
            _, attn_output = self.attn(qry, ctx_keys, ctx_values, None)
                
            tmp_len = max_seq_len - attn_output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(attn_output)
                attn_output = torch.cat([attn_output, tmp], dim=0)

            outputs.append(attn_output)

        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output

class AWSREmbedding(Embedding):
    """Additional Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(AWSREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)
        self.lin = nn.Linear(emb_dim, emb_dim)
        self.qry_emb = nn.Parameter(torch.randn(emb_dim))
        self.attn = Attention('dot', emb_dim, emb_dim)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)
            ctx_values_1 = emb * weight
            output_1 = ctx_values_1.sum(dim=1)

            ctx_keys = torch.tanh(self.lin(emb))

            qry = self.qry_emb.view(1, -1).repeat(seq_len, 1)
            _, output_2 = self.attn(qry, ctx_keys, emb, None)
            
            attn_output = output_1 + output_2

            tmp_len = max_seq_len - attn_output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(attn_output)
                attn_output = torch.cat([attn_output, tmp], dim=0)

            outputs.append(attn_output)

        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output
 
class CWSREmbedding(Embedding):
    """Concated Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(CWSREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)

        self.emb_dim = emb_dim * 2

        self.lin = nn.Linear(emb_dim, emb_dim)
        self.qry_emb = nn.Parameter(torch.randn(emb_dim))
        self.attn = Attention('dot', emb_dim, emb_dim)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)
            ctx_values_1 = emb * weight
            output_1 = ctx_values_1.sum(dim=1)

            ctx_keys = torch.tanh(self.lin(emb))

            qry = self.qry_emb.view(1, -1).repeat(seq_len, 1)
            _, output_2 = self.attn(qry, ctx_keys, emb, None)
            
            attn_output = torch.cat([output_1, output_2], dim=1)

            tmp_len = max_seq_len - attn_output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(attn_output)
                attn_output = torch.cat([attn_output, tmp], dim=0)

            outputs.append(attn_output)

        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output

class SWSREmbedding(Embedding):
    """Simple Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(SWSREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)

            ctx_values = emb * weight
            output = ctx_values.sum(dim=1)

            tmp_len = max_seq_len - output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(output)
                output = torch.cat([output, tmp], dim=0)

            outputs.append(output)
        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output
 

class CAVGEmbedding(Embedding):
    """Simple Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(CAVGEmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)

        self.emb_dim = emb_dim * 2

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)

            tmp_len = emb.size(1)
            tmp_out = emb.sum(dim=1) / tmp_len

            ctx_values = emb * weight
            output = ctx_values.sum(dim=1)

            output = torch.cat([output, tmp_out], dim=1)

            tmp_len = max_seq_len - output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(output)
                output = torch.cat([output, tmp], dim=0)

            outputs.append(output)
        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output
 
class SAVGEmbedding(Embedding):
    """Simple Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(SAVGEmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)

            tmp_len = emb.size(1)
            tmp_out = emb.sum(dim=1) / tmp_len

            ctx_values = emb * weight
            output = ctx_values.sum(dim=1)

            output = output + tmp_out

            tmp_len = max_seq_len - output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(output)
                output = torch.cat([output, tmp], dim=0)

            outputs.append(output)
        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output
 
class OWSREmbedding(Embedding):
    """Only Weighted Sum Representation Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx, dropout=0.0):
        super(OWSREmbedding, self).__init__(vocab_size, emb_dim, padding_idx, dropout)

        self.lin = nn.Linear(emb_dim, emb_dim)
        self.qry_emb = nn.Parameter(torch.randn(emb_dim))
        self.attn = Attention('dot', emb_dim, emb_dim)

    def forward(self, input):
        """
        input (list of tuple): (data, weight)
            - list: of length batch_size
            - data: seq_len * arc_num
            - weight: seq_len * arc_num
        """
        batch_size = len(input)
        max_seq_len = max([d.size(0) for (d,w) in input])
        
        outputs = []
        for (data, weight) in input:
            seq_len = data.size(0)

            emb = self.embedding(data)
            weight = weight.unsqueeze(-1)

            ctx_keys = torch.tanh(self.lin(emb))

            qry = self.qry_emb.view(1, -1).repeat(seq_len, 1)
            _, attn_output = self.attn(qry, ctx_keys, emb, None)
            
            tmp_len = max_seq_len - attn_output.size(0)
            if tmp_len > 0: 
                tmp = torch.zeros(tmp_len, self.emb_dim).type_as(attn_output)
                attn_output = torch.cat([attn_output, tmp], dim=0)

            outputs.append(attn_output)

        output = torch.stack(outputs, dim=0)

        output = self.dropout(output)

        return output


