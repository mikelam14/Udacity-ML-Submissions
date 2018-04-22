#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:05:20 2018

@author: MikeLam
"""

import numpy as np
import os
os.chdir('/Users/MikeLam/Google Drive/Mikee/Self Learn/Barking Notice/')

from utils import create_training_balanced, predict_wrap
from model import transferLearning
from preprocess import getDataWrap
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

random_seed = 99
par = {'epoch': 3, 'batch': 32, 'shuffle': True, 'verbose': 1, 'print': False}
result = {'vanilla':[], 'CNN':[], 'transferLearn':[]}
threshold, ratio = 0.5, 2
dense = [100, 100]
keep, untrain = 9, 19
fit = True

###############################################################################
train_X, train_y, val_X, val_y, test_X, test_y, train_vid, val_vid, test_vid, \
        neg_map = getDataWrap(random_seed)
        
# after split, create train dataset according to imbalance spec (ratio)
input_shape = train_X.shape[1:]
X,y = create_training_balanced(train_X, train_y, neg_map, random_seed, ratio)

tl_X = np.repeat(X[:, :, :, np.newaxis], 3, axis = 3)
tl_val_X = np.repeat(val_X[:, :, :, np.newaxis], 3, axis = 3)

tl_X, y = shuffle(tl_X, y, random_state=random_seed)

#C. Transfer Learning
saveName_t = 'transferLearn-r'+str(ratio)+'.best.hdf5'
for ratio in [10]:
    for keep in [6,10,14,18]:
        os.system('rm log/'+saveName_t)
        model_t = transferLearning(
                keep, untrain, dense, input_shape, l='binary_crossentropy',
                opt='adam', met=['binary_accuracy'], dropout=0.5)
        print(model_t.summary())
        if fit:
            verbose = par['verbose']
            epoch = par['epoch']
            batch = par['batch']
            shuffle = par['shuffle']
            printResult = par['print']
            checkpointer = ModelCheckpoint(filepath='log/'+saveName_t, 
                                           verbose=verbose,
                                           save_best_only=True)
            hist = model_t.fit(tl_X, y, batch_size=batch, epochs=epoch,
                               validation_split=0.1, callbacks=[checkpointer],
                               verbose=verbose, shuffle=False)
            # use best validation to test
            model_t.load_weights('log/'+saveName_t)
            ## need to pass through each 10 seconds, average output to determine
            f1_t, ll_t = predict_wrap(model_t, tl_val_X, val_y, val_vid, threshold, 
                                       verbo=printResult)
            result['transferLearn'].append((keep, ratio, f1_t, ll_t))

