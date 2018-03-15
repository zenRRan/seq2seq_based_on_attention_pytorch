#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: main.py
@time: 2018/3/7 14:56
"""

from Reader import reader
from Config import configer
from Alphabet import Alphabet
import math
import time
from AttnDecoderRNN import AttnDecoderRNN
from BahdanauAttnDecoderRNN import BahdanauAttnDecoderRNN
from train import train
from Encoder import Encoder
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
PAD = '<PAD>'
def as_minutes(sec):
    min = math.floor(sec / 60)
    sec -= min * 60
    return '%dm %ds'% (min, sec)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def seq2id(Alphabet, seq_list):
    ids = []
    for line in seq_list:
        new_ids = []
        for word in line:
            index = Alphabet.string2id[word]
            if index is not None:
                new_ids.append(index)
            else:
                index = Alphabet.string2id(PAD)
                new_ids.append(index)
        new_ids.append(EOS_token)
        ids.append(new_ids)
    return ids

def filter_sent_label(sent, label):
    return len(sent) < MAX_LENGTH and len(label) < MAX_LENGTH

def filter_sents_labels(sents, labels):
    return [[sents[i], labels[i]] for i in range(len(sents)) if filter_sent_label(sents[i], labels[i])]



if __name__ == '__main__':
    config_path = 'D:\PyCharm\pycharm_workshop\seq2seq_based_on_attention_pytorch\config.cfg'
    config = configer(config_path)
    Reader = reader(config.corpus_path, needFresh=True, language='eng')
    text_sent_list, label_sent_list = Reader.getData()
    label_sent_list, text_sent_list = text_sent_list, label_sent_list
    # sent_label_list = filter_sents_labels(text_sent_list, label_sent_list)



    '''
        create dictionary
    '''
    # print(text_sent_list)
    text_word_state = {'SOS': 1, 'EOS': 1, PAD: 1}
    label_word_state = {'SOS': 1, 'EOS': 1, PAD: 1}
    for line in text_sent_list:
        for word in line:
            if word not in text_word_state:
                text_word_state[word] = 1
            else:
                text_word_state[word] += 1
    for line in label_sent_list:
        for word in line:
            if word not in label_word_state:
                label_word_state[word] = 1
            else:
                label_word_state[word] += 1

    '''
        create Alphabet
    '''
    text_alpha = Alphabet()
    label_alpha = Alphabet()
    text_alpha.initial(text_word_state)
    label_alpha.initial(label_word_state)

    # print(text_alpha.id2string)
    print('test word size:', text_alpha.m_size)
    print('label word size:', label_alpha.m_size)
    # print(label_alpha.id2string)

    '''
        seqs to id
    '''
    text_id_list = seq2id(text_alpha, text_sent_list)
    label_id_list = seq2id(label_alpha, label_sent_list)

    encoder = Encoder(text_alpha.m_size, config)
    decoder = AttnDecoderRNN(label_alpha.m_size, config)

    # print(encoder)
    # print(decoder)
    lr = config.lr
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    n_epochs = 1000
    plot_every = 200
    print_every = 10

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    '''
        start...
    '''
    for epoch in range(n_epochs):
        index = random.choice(range(len(text_sent_list)))
        text = Variable(torch.LongTensor(text_id_list[index]))
        label = Variable(torch.LongTensor(label_id_list[index]))
        loss = train(text, label, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH)

        print_loss_total += loss

        if epoch == 0:
            continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, float(epoch) / n_epochs), epoch,
                                                  float(epoch) / n_epochs*100, print_loss_avg)

            print(print_summary)

    fmodel = 'seq2seq-%d.param' % n_epochs
    torch.save([encoder, decoder], fmodel)
    encoder, decoder = torch.load(fmodel)


    def evaluate(sentence, max_length=MAX_LENGTH):
        ids = seq2id(text_alpha, [sentence])
        input_variable = Variable(torch.LongTensor(ids[0]))

        #through encoder
        encoder_hidden = Variable(encoder.init_hidden())
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        #through decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
        decoder_hidden = encoder_hidden

        decoder_words = []
        for i in range(max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            index = topi[0][0]
            if index == EOS_token:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(label_alpha.id2string[index])

            decoder_input = Variable(torch.LongTensor([[index]]))

        return decoder_words


    def evaluate_randomly():
        index = random.choice(range(len(text_sent_list)))
        text = text_sent_list[index]
        label = label_sent_list[index]

        words_output = evaluate(text)
        text = ' '.join(text)
        label = ' '.join(label)
        output = ' '.join(words_output[:-1])

        print('>', text)
        print('=', label)
        print('<', output)

    evaluate_randomly()


