#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:54:36 2018

@author: weikaiqi
Note: IF you use LSTM and your model overfitting to the dev set, it's better to switch to GRU.
"""

#keras
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from pylib.Attention import Attention


# Simple bidirectional LSTM + two fully connected layers.
def LSTM_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))

    x = Embedding(max_features+1, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x1 = GlobalMaxPool1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = concatenate([x1,x2])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model

# Simple bidirectional GRU + two fully connected layers
def GRU_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features+1, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(GRU(50, return_sequences=True))(x)
    x = Bidirectional(GRU(50, return_sequences=True))(x)
    x1 = GlobalMaxPool1D()(x)
    x2 = GlobalAveragePooling1D()(x)
	#x = GlobalMaxPool1D()(x)
    x = concatenate([x1,x2])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    return model
