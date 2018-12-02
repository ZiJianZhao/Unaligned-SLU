# -*- coding: utf-8 -*-
import os, datetime
import logging
import time, math

import torch
import torch.nn as nn


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
            "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Borrow from OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Utils.py

    Args:
        lengths (LongTensor): containing sequence lengths
        max_len (int): maximum padded length
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class RoundNoGradient(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


