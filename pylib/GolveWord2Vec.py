#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:43:20 2018

@author: weikaiqi
Notes: change line.split() by change line.split(' ')
"""
import numpy as np


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
def WordEmbedding(EMBEDDING_FILE, nb_words, word_index, embed_size, max_features):
    #def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    #embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

    embeddings_index = {}
    f = open(EMBEDDING_FILE)#,encoding='utf8')
    count = 0
    for line in f:
        values = line.split(' ')
        #values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    # Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. 
    # We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    #assgin random number to embedding matrix 
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))

    n_oov = 0
    #Use Golve if word in it
    for word, i in word_index.items():
        if i >= nb_words: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            n_oov += 1

    print("number words out of vocabulary: ", n_oov)
    return embedding_matrix, embeddings_index
