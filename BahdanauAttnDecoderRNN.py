#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: BahdanauAttnDecoderRNN.py
@time: 2018/3/12 15:58
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from Attn import Attn

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, hidden_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('general', hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, hidden_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, word_input, last_hidden, encoder_ouyputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        word_embedded = self.dropout(word_embedded)

        attn_weights = self.attn(last_hidden[-1], encoder_ouyputs)
        context = attn_weights.bmm(encoder_ouyputs.transpose(0, 1))

        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        return output, hidden, attn_weights
