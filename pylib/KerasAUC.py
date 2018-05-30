#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:11:18 2018

@author: weikaiqi
"""

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

# Note: you can't calculate ROC&AUC by mini-batches, you can only calculate it on the end of one epoch
class RocAucEvaluation(Callback):
    def __init__(self, training_data=(), validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval          = interval
        self.X    , self.y     = training_data
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred     = self.model.predict(self.X, verbose=0)
            score      = roc_auc_score(self.y, y_pred)
            
            y_pred_val = self.model.predict(self.X_val, verbose=0)
            score_val  = roc_auc_score(self.y_val, y_pred_val)
            
            print("\n ROC-AUC - epoch: {:d} - train score: {: 6f} - val score: {:.6f}".format(epoch, score, score_val))

