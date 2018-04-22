#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:46:54 2018

@author: MikeLam
"""
import numpy as np
import os
os.chdir('/Users/MikeLam/Google Drive/Mikee/Self Learn/Barking Notice/')

from utils import create_training_balanced, fit_predict_wrap
from model import vanilla, CNN, transferLearning

###############################################################################
# get Data
def run(train_X, train_y, neg_map, random_seed, ratio, modelss, val_X, val_y,
        test_X, test_y, test_vid, par, result, threshold=0.5):
    # after split, create train dataset according to imbalance spec (ratio)
    input_shape = train_X.shape[1:]
    X,y = create_training_balanced(train_X, train_y, neg_map, random_seed, ratio)
    
    ###############################################################################
    if 1 in modelss:
        #A. build a pre-train model
        dim = [10, 5]
        saveName_v = 'vanilla-r'+str(ratio)+'.best.hdf5'
        model_v = vanilla(input_shape,dim,dropout=0.5)
    
    if 2 in modelss or 3 in modelss:
        tl_X = np.repeat(X[:, :, :, np.newaxis], 3, axis = 3)
        tl_val_X = np.repeat(val_X[:, :, :, np.newaxis], 3, axis = 3)
        tl_test_X = np.repeat(test_X[:, :, :, np.newaxis], 3, axis = 3)
    
    if 2 in modelss:
        #B. CNN from scratch
        dense = []
        saveName_c = 'cnn-r'+str(ratio)+'.best.hdf5'
        model_c = CNN(dense, input_shape, l='binary_crossentropy', opt='adam',
                      met=['binary_accuracy'], dropout=0.5)
    
    if 3 in modelss:
        #C. Transfer Learning
        dense = [100, 50]
        keep, untrain = 10, 19
        saveName_t = 'transferLearn-r'+str(ratio)+'.best.hdf5'
        model_t = transferLearning(
                keep, untrain, dense, input_shape, l='binary_crossentropy', 
                opt='adam', met=['binary_accuracy'], dropout=0.5)
    
    # fit models
    if 1 in modelss:
        hist_v, f1_v, ll_v = fit_predict_wrap(
                model_v, X, y, val_X, val_y, test_X, test_y, 
                test_vid, ratio, saveName_v, par, threshold)
        result['vanilla'].append((ratio, f1_v, ll_v))
        
    if 2 in modelss:
        hist_c, f1_c, ll_c = fit_predict_wrap(
                model_c, tl_X, y, tl_val_X, val_y, tl_test_X, test_y, 
                test_vid, ratio, saveName_c, par, threshold)
        result['CNN'].append((ratio, f1_c, ll_c))
        
    if 3 in modelss:
        hist_t, f1_t, ll_t = fit_predict_wrap(
                model_t, tl_X, y, tl_val_X, val_y, tl_test_X, test_y, 
                test_vid, ratio, saveName_t, par, threshold)
        result['transferLearn'].append((ratio, f1_t, ll_t))
    ###############################################################################
    print("")
    if 1 in modelss:
        print(model_v.summary())
    if 2 in modelss:
        print(model_c.summary())
    if 3 in modelss:
        print(model_t.summary())
    print("")
    if 1 in modelss:
        print("Pos:Neg=1:{0}, vanilla_NN has f1 score of {1:.4}, log-loss {2:.4}"\
              .format(ratio, f1_v, ll_v))
    if 2 in modelss:
        print("Pos:Neg=1:{0}, CNN has f1 score of {1:.4}, log-loss {2:.4}"\
              .format(ratio, f1_c, ll_c))
    if 3 in modelss:
        print("Pos:Neg=1:{0}, VGG learn has f1 score of {1:.4}, log-loss {2:.4}"\
              .format(ratio, f1_t, ll_t))
    return(result)
