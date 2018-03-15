#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Config.py
@time: 2018/3/7 14:55
"""

from configparser import ConfigParser

class configer:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path, encoding='utf-8')

    @property
    def corpus_path(self):
        return self.config.get('path', 'corpus_path')

    @property
    def GRU(self):
        return self.config.getboolean('network', 'GRU')

    @property
    def LSTM(self):
        return self.config.getboolean('network', 'LSTM')

    @property
    def Steps(self):
        return self.config.getint('parameters', 'Steps')

    @property
    def lr(self):
        return self.config.getfloat('parameters', 'lr')

    @property
    def hidden_size(self):
        return self.config.getint('parameters', 'hidden_size')

    @property
    def hidden_layers(self):
        return self.config.getint('parameters', 'hidden_layers')

    @property
    def dropout(self):
        return self.config.getfloat('parameters', 'dropout')

    @property
    def embed_dim(self):
        return self.config.getint('parameters', 'embed_dim')

    @property
    def attn_model(self):
        return self.config.get('parameters', 'attn_model')

    @property
    def max_length(self):
        return self.config.get('parameters', 'max_length')
