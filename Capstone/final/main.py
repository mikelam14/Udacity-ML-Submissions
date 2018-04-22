#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:36:41 2018

@author: MikeLam
"""
from train import run
from preprocess import getDataWrap

# config
random_seed = 99
par = {'epoch': 3, 'batch': 32, 'shuffle': True, 'verbose': 1, 'print': True}
modelss = [3]
result = {'vanilla':[], 'CNN':[], 'transferLearn':[]}

# get data
train_X, train_y, val_X, val_y, test_X, test_y, train_vid, val_vid, test_vid, \
        neg_map = getDataWrap(random_seed)

test = [(2, 0.333), (3, 0.25), (5, 0.16667), (10, 0.0909)]
for (ratio, threshold) in test:
    result = run(train_X, train_y, neg_map, random_seed, ratio, modelss, 
                 val_X, val_y, test_X, test_y, test_vid, par, result, 
                 threshold=threshold)
