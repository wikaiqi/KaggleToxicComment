#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:54:30 2018

@author: weikaiqi
Function: DownSampling

"""
import numpy as np


def Random_DownSampling(train, X, y, rate = 4):
    '''
    Downsamping the sampe
    '''
    
    train["toxic_yes"] = (train['toxic']  + train['severe_toxic'] + train['obscene'] + 
                          train['threat'] + train['insult']       + train['identity_hate'])

    train["toxic_yes"] = train["toxic_yes"].apply(lambda x: 1 if x > 0 else 0)
    yc = train["toxic_yes"].values
    
    
    index = np.argwhere(yc==0).flatten()
    
    num_tot = y.shape[0]
    num_is = len(index)
    num_ns = num_tot - num_is
    
    print("number of non_toxic sample: {}".format(num_is))
    print("number of toxic     sample: {}".format(num_ns))
    print("before non_toxic/toxic rate: {}".format(num_is/num_ns))
    
    #num_af = int(num_is/2)
    num_af = int(num_is - num_ns*rate)
    
    index = np.random.choice(index,num_af)
    notindex = np.array([i for i in range(num_tot) if i not in index])
    
    yselect = np.take(y, notindex, axis=0)
    Xselect = np.take(X, notindex, axis=0)
    print("after non_toxic/toxic rate: {}".format(num_af/num_ns))
    
    return Xselect, yselect
