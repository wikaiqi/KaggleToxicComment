#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:31:01 2018

@author: weikaiqi
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def Text2Token(train, test, max_features, maxlen):
    '''
    Standard keras preprocessing, to turn each comment into a list of word 
    indexes of equal length (with truncation or padding as needed).
    
    The corpus includes all words in train and test sets.
    '''
    
    list_sentences_train = train["comment_text"].tolist()
    list_sentences_test  = test["comment_text"].tolist()
    
    list_all = []
    list_all.extend(list_sentences_train)
    list_all.extend(list_sentences_test)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list_all)
    
    word_index = tokenizer.word_index
    print(len(word_index))
    nb_words   = min(max_features, len(word_index))

    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test  = tokenizer.texts_to_sequences(list_sentences_test)

    X_train  = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test   = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    return X_train, X_test, word_index, nb_words
