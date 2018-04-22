# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:02:10 2018

@author: pc
"""

import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, f1_score, confusion_matrix as CM

os.chdir('/Users/MikeLam/Google Drive/Mikee/Self Learn/Barking Notice/')

def fit_predict_wrap(model, X, y, val_X, val_y, test_X, test_y, test_vid, ratio,
                 saveName, par, threshold=0.5):
    saveName = 'log/'+saveName
    verbose = par['verbose']
    epoch = par['epoch']
    batch = par['batch']
    shuffle = par['shuffle']
    printResult = par['print']
    try:
        model.load_weights(saveName)
        hist = 'loaded from previous best'
    except (FileNotFoundError, OSError) as e:
        checkpointer = ModelCheckpoint(filepath=saveName, verbose=verbose,
                                       save_best_only=True)
        hist = model.fit(X, y, batch_size=batch, epochs=epoch,
                         validation_data=(val_X,val_y), callbacks=[checkpointer],
                         verbose=verbose, shuffle=shuffle)
        # use best validation to test
        model.load_weights(saveName)
    ## need to pass through each 10 seconds, average output to determine
    f1, logloss = predict_wrap(model, test_X, test_y, test_vid, threshold, 
                               verbo=printResult)
    return(hist, f1, logloss)

# accuracy - f1 score and logloss
def predict_wrap(model, test_X, test_y, test_vid, threshold=0.5, verbo=False):
    predictions = model.predict(test_X)
    y_true, y_prob, y_pred = [], [], []
    for vid in np.unique(test_vid):
        ind = np.where(test_vid == vid)[0]
        # if not len(ind) == 10: print(vid)  # debug
        truth = np.mean(test_y[ind],dtype=int)
        prob = np.mean(predictions[ind])
        y_true.append(truth)
        y_prob.append(prob)
        y_pred.append(1 if prob >= threshold else 0)
        if verbo: print(truth, prob)  # debug
    # print out confusion matrix
    if verbo: CM(y_true, y_pred)
    # compute metrics
    f1 = f1_score(y_true, y_pred)
    logloss = log_loss(y_true, y_prob)
    return(f1, logloss)

def split_data(train_X, train_y, test_X, test_y, random_seed):
    '''
    X.shape[0] == y.shape[0] == vid.shape[0]
    '''
    # get video id, then split 50-50
    vid_id = np.unique(test_X[:, 0, -1])
    id_ = shuffle(vid_id, random_state=random_seed)
    id_ = [int(i) for i in id_]
    val_id = id_[:len(id_)//2]
    test_id = id_[len(id_)//2:]
    # then get the index in the np array
    val_ind = np.where(np.isin(test_X[:, 0, -1], val_id))[0]
    test_ind = np.where(np.isin(test_X[:, 0, -1], test_id))[0]
    # split according to the index 
    val_X, val_y = test_X[val_ind], test_y[val_ind]
    test_X, test_y = test_X[test_ind], test_y[test_ind]
    # remove the video id from the nd array
    val_X, val_vid = val_X[:, :, :-1], val_X[:, 0, -1]
    test_X, test_vid = test_X[:, :, :-1], test_X[:, 0, -1]
    train_X, train_vid = train_X[:, :, :-1], train_X[:, 0, -1]
    # store the video id in nd array
    val_vid = np.array([int(i) for i in val_vid]) 
    test_vid = np.array([int(i) for i in test_vid])
    train_vid = np.array([int(i) for i in train_vid])
    neg_map = create_negative_map(train_y, train_vid)
    return(train_X, train_y, val_X, val_y, test_X, test_y,
           train_vid, val_vid, test_vid, neg_map)

def create_negative_map(y, vid):
    ind = np.where(y == 0)[0]
    neg_vid = np.unique(vid[ind])
    neg_map = {}
    for v in neg_vid:
        neg_map[v] = np.where(vid == v)[0]
    return(neg_map)

def create_training_balanced(train_X, train_y, neg_map, random_seed, ratio=1):
    '''
    positive:negative = 1:ratio
    use all positive anyway
    '''
    n = int(60 * ratio)
    neg_keys = shuffle(list(neg_map.keys()), random_state=random_seed)
    neg_X = train_X[neg_map[neg_keys[0]],:,:]
    for k in neg_keys[1:n]:
        neg_X = np.concatenate((neg_X, train_X[neg_map[k],:,:]), axis=0)
    neg_y = np.zeros(neg_X.shape[0], dtype=int)
    pos_ind = np.where(train_y == 1)[0]
    pos_X = train_X[pos_ind,:,:]
    pos_y = np.ones(pos_X.shape[0], dtype=int)
    X = np.concatenate((pos_X, neg_X), axis=0)
    y = np.concatenate((pos_y, neg_y), axis=0)
    return(X,y)
