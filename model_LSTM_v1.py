#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Xiao Jin
In this file we construct the model
"""
from config_v1 import * 
import tensorflow as tf
# from tf.keras.layers import Masking, Bidirectional, LSTM


class masking(tf.keras.Model):
    def __init__(self):
        super(masking, self).__init__()
        self.masking = tf.keras.layers.Masking()

    def forward(self, input):
        return self.masking(input)


class local_embedding(tf.keras.Model):
    def __init__(self):
        super(local_embedding, self).__init__()
        self.b = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_units, activation='tanh',
                                                                    return_sequences=True, dropout=dropout))
        self.l = tf.keras.layers.LSTM(units=num_units, activation='tanh', return_sequences=False, dropout=dropout)
        # LSTM layers
        '''
        self.b = Bidirectional(LSTM(units=num_units, activation='tanh', return_sequences=True, recurrent_dropout=rec_dropout, 
        dropout=dropout))
        # last lstm layers
        self.l = LSTM(units=num_units, activation='tanh', return_sequences=False, recurrent_dropout=rec_dropout, 
        dropout=dropout)
        '''

    def forward(self, input):
        # input size should be batch_size x 48 x Feature_Space
        # print(input)
        x = self.b(input)
        # shape = x.numpy().shape
        # noise = tf.random.normal(shape = shape) * 0.01
        # x = x + noise
        x = self.l(x)
        # shape = x.numpy().shape
        # noise = tf.random.normal(shape = shape) * 0.01
        # x = x + noise
        # x should be batch_size x 16
        # return tf.reshape(x, [-1, num_units])
        return x


class server(tf.keras.Model):
    def __init__(self):
        super(server, self).__init__()
        self.fc1 = tf.keras.layers.Dense(1, activation = 'sigmoid', name='fc1')

    def forward(self, input):
        x = tf.nn.dropout(input, 0.3)
        x = self.fc1(x)
        return x





