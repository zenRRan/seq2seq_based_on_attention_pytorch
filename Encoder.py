#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Encoder.py
@time: 2018/3/7 14:56
"""

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, word_num, config):
        super(Encoder, self).__init__()
        self.input_size = word_num
        self.hidden_size = config.hidden_size
        self.hidden_layers = config.hidden_layers
        self.embed_dim = config.embed_dim

        self.embedding = nn.Embedding(self.input_size, self.embed_dim)
        self.GRU = nn.GRU(self.embed_dim, self.hidden_size, self.hidden_layers)

    def forward(self, inputs, hidden):
        seq_len = len(inputs)
        inputs = self.embedding(inputs)
        inputs = inputs.view(seq_len, 1, -1)
        inputs, hidden = self.GRU(inputs, hidden)
        return inputs, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.hidden_layers, 1, self.hidden_size)
        return hidden

