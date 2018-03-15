#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: AttnDecoderRNN.py
@time: 2018/3/11 10:36
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Attn import Attn


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, config):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = config.attn_model
        self.hidden_size = config.hidden_size
        self.output_size = output_size
        self.hidden_layers = config.hidden_layers
        self.dropout = config.dropout
        self.embed_size = config.embed_dim

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size*2, self.hidden_size, self.hidden_layers, dropout=self.dropout)
        self.out = nn.Linear(self.embed_size*2, self.output_size)

        if self.attn_model != 'none':
            self.attn = Attn(self.attn_model, self.hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input)
        word_embedded = word_embedded.view(1, 1, -1)

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        return output, context, hidden, attn_weights