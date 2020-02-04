# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:58:10 2020

@author: hamed
"""


from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import zipfile
import pickle
import time
import json
import wget
import re
import os


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.embedder = tf.keras.layers.Dense(units)
        self.combinator = tf.keras.layers.Dense(units)
        self.attn_weights = tf.keras.layers.Dense(1)
        return

    def call(self, features, hidden):
        hidden_ = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.embedder(features) + self.combinator(hidden_))
        attention_weights = tf.nn.softmax(self.attn_weights(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class EFC_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(EFC_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
        return

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    
    
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.GRU(self.units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.units)
        return

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.rnn(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class CNN_Encoder(tf.keras.Model):
    def __init__(self, cnn_type='default_resnet152v2'):
        super(CNN_Encoder, self).__init__()
        if cnn_type == 'inceptionv3':
            cnn = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        elif cnn_type == 'vggnet':
            cnn = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        else:
            cnn = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')
        new_input = cnn.input
        hidden_layer = cnn.layers[-1].output
        self.f = tf.keras.Model(new_input, hidden_layer)
        return

    def call(self, x):
        return self.f(x)


def save_model_weights(model, model_name="NN"):
    model.save_weights("Trained/" + model_name + "/m{epoch:01d}.ckpt".format(epoch=0))
    return

