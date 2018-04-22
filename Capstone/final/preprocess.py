#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 08:52:03 2018

@author: MikeLam
"""

import json
import pickle
import librosa, librosa.display
import subprocess
import os
import re
import pandas as pd
import numpy as np
from utils import split_data

# train.py uses only getDataWrap
def getDataWrap(random_seed=99):
    # get bark id
    json_file, which = 'ontology.json', 'Bark'
    allTags = getTags(json_file)
    barkID = getTagsID(json_file, which)
    
    # Load data
    ## download data - if name is given, then save audio, otherwise just the 
    ## data from spreadsheet
    train,test = loadData(allTags,barkID)
    train_X,train_y,tr_pos_err,tr_neg_err = train
    test_X, test_y,tt_pos_err,tt_neg_err = test
    # print(train_X.shape, test_X.shape, len(np.unique(train_X[:,0,-1])),
    #      len(np.unique(test_X[:,0,-1])))
    # print(train_y.shape, test_y.shape)

    # training set, balanced(1:1), (1:2), (1:3), (1:5)
    # split test into test and validation. test set (1:1.5) 
    train_X, train_y, val_X, val_y, test_X, test_y, train_vid, val_vid, test_vid, \
        neg_map = split_data(train_X, train_y, test_X, test_y, random_seed)
    return(train_X, train_y, val_X, val_y, test_X, test_y, train_vid, val_vid, test_vid, \
        neg_map)

## reading spreadsheet and get data
# getTags
def getTags(json_file):
    with open(json_file,encoding='utf-8') as json_data:
        Ds = json.load(json_data)
    result = []
    for i in range(len(Ds)):
        if len(Ds[i]['positive_examples']) > 0:
            result.append(Ds[i])
    return(result)

# getTagsID
def getTagsID(json_file, which):
    with open(json_file,encoding='utf-8') as json_data:
        Ds = json.load(json_data)
    for d in Ds:
        if d['name'] == which:
            return(d['id'])

# getData
def getData(file,soundID,name=None):
    meta,sound_df = getInfo(file,soundID)
    if name is None:
        return(meta,sound_df,None)
    else:
        error = saveVideo(sound_df,name)
        return(meta,sound_df,error)

# get info
def getInfo(file,soundID,per=None):
    fullfile = os.getcwd()+'/'+file
    meta = pd.read_csv(fullfile,header=None,nrows=2,delimiter='#').loc[:,1] # read file description
    spreadsheet = pd.read_csv(fullfile,skiprows=3,delimiter=' ',header=None) # get the rest                    
    # regex search for existence of the wanted soundID
    temp = spreadsheet.apply(lambda x: not(re.search(soundID,x[3]) is None),axis=1) 
    # extract the dataframe
    sound_df = spreadsheet[temp] # include only the positive ones
    sound_df.reset_index(inplace=True) # index is from spreadsheet
    sound_df = sound_df.iloc[:per]
    return(meta,sound_df)

# download video & save wav file
def saveVideo(sound_df,name,verbo=False):
    link = 'https://www.youtube.com/watch?v='
    error = []
    folder = os.getcwd()+'/data/'
    if isinstance(sound_df,pd.DataFrame):
        n = sound_df.shape[0]
    else:
        n = len(sound_df)
        
    for i in range(n):
        st = str(sound_df[1][sound_df.index[i]].split(",")[0].split(".")[0])
        ed = str(sound_df[2][sound_df.index[i]].split(",")[0].split(".")[0])
        t = str(int(ed)-int(st))
        youLink = link+ sound_df[0][sound_df.index[i]].split(",")[0] # combine video id
        # use youtube-dl to get full Link
        try:
            fullLink = subprocess.check_output(['youtube-dl','-f','bestaudio','-g',youLink]).decode('UTF-8').split("\n")[0]
            fullLink = '"' + fullLink+'"'
            # use ffmpeg to get audio and save
            loc = '"'+folder+name+'-'+str(i+1)+'.wav"'
            ffcom = 'ffmpeg -ss '+st+' -i '+fullLink+' -t '+t+' '+loc
            '''
            -t: duration
            -ss: position (before -i)
            -f 22: audio
            '''           
            c = subprocess.Popen(ffcom,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
            out,err = c.communicate()
            if out == b'':
                if verbo: print('file number - '+str(i+1)+': Done')
            else:  
                error.append(sound_df.index[i])
                if verbo: print('file number - '+str(i+1)+': Failed') 
        except:
            error.append(sound_df.index[i])
            if verbo: print('file number - '+str(i+1)+': Failed') 
    return(error)

## Load data
def loadData(allTags,soundID):
    train_f = 'balanced_train_segments.csv'
    test_f = 'eval_segments.csv'
    if not os.path.isfile('data/training-1-1.wav'):
        print('Getting positive video...')
        meta_tr, spreadsheet_tr, failed_tr = getData(train_f,soundID,'training-1')
        if not os.path.isfile('data/testing-1-1.wav'):
            meta_tt, spreadsheet_tt, failed_tt = getData(test_f,soundID,'testing-1')
    else:
        print('Positive audios already exist')
    
    # train feat    
    if not os.path.isfile('train.pickle'):
        (train_X,train_y,tr_pos_err,tr_neg_err) = extractWrap(train_f,allTags,soundID,save=True)
    else: 
        print('Training file already exists')
        pickle_in = open("train.pickle","rb")
        train_dict = pickle.load(pickle_in)
        (train_X,train_y,tr_pos_err,tr_neg_err) = train_dict['data']
    # test feat
    if not os.path.isfile('test.pickle'):
        (test_X,test_y,tt_pos_err,tt_neg_err) = extractWrap(test_f,allTags,soundID,save=True)
    else:
        print('Testing file already exists')
        pickle_in = open("test.pickle","rb")
        test_dict = pickle.load(pickle_in)
        (test_X,test_y,tt_pos_err,tt_neg_err) = test_dict['data']
    return((train_X,train_y,tr_pos_err,tr_neg_err),(test_X,test_y,tt_pos_err,tt_neg_err))
    
# extract features for one image
def extractFeature(audio,feats,which,window_size=960,plot=False):
    '''
    Input: a file (audio), list (feats), which (video number).
    Output: feats (mel-spectrogram 64*98 + one column of video number 64*1)
    '''
    toAdd = []
    # get signal and sampling rate
    (X, Fs) = librosa.core.load(audio)
    # break into windows and get spectrogram
    n_window = int(X.shape[0]/Fs/(window_size/1000))
    for i in range(n_window):
        x = X[i*window_size:(i+1)*window_size]
        D = np.abs(librosa.stft(x, n_fft=512, win_length=25, hop_length=10))**2
        S = librosa.feature.melspectrogram(S=D, y=x, n_mels=64) # S is 64 bin x 97 
        # https://github.com/librosa/librosa/issues/595
        # https://arxiv.org/pdf/1609.09430.pdf
        
        # visualize
        if plot:
            plt.figure()
            librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                                  x_axis='time', y_axis='mel', fmax=8000)
        # save feats
        # add "which" to indicate which 10 frames belong to same video
        s = np.concatenate((S,np.ones((S.shape[0],1))*which),axis=1)
        toAdd.append(s)
    if n_window == 10:
        feats = feats+toAdd
    else:
        print("Problem {}".format(audio)) 
    return(feats)

def extractPositiveFeatures(train=True,verbo=True,save=False):
    '''
    positive features, using the already downloaded videos
    returns (X,y,error) - X (size,64,98),y (size,1), error (problematic)
    X -> 64*97 mel spectrogram + 64*1 of video number
    y -> an array of 1, 0 
    '''
    X = []
    error = []
    if train:
        schema = os.getcwd()+'/data/training-1'
    else:
        schema = os.getcwd()+'/data/testing-1'
    for i in range(60):
        name = schema+'-'+str(i+1)+'.wav'
        try:
            X = extractFeature(name,X,which=i)
        except:
            error.append(i)
        if (i%5) == 0 and verbo: print('Extract Pos feature - Progress: '+str(i+1)+'/60')
    # after looping through all samples
    X = np.array(X)
    y = np.ones(X.shape[0]) # all positive here
            
    if save:
        nm = 'pos_train.pickle' if train else 'pos_test.pickle'
        pickle_out = open(nm,"wb")
        pickle.dump({'pos':(X,y,error)}, pickle_out)
        pickle_out.close()
    return(X,y,error)
                        
def extractNegativeFeatures(link_df,n_pos,train=True,verbo=True,save=False):
    '''
    negative features
    input: get link_df which store one row of dataframe containing negative samples
    returns (X,y,error) - X (size,64,98),y (size,1), error (problematic)
    X -> 64*97 mel spectrogram + 64*1 of video number
    y -> an array of 1, 0 
    '''
    X = []
    error = []
    name = os.getcwd()+'/data/negative-1.wav'
    for i in range(len(link_df)):
        # get video, save
        err = saveVideo(link_df[i],'negative')
        # extract feature
        if len(err) == 0:
            X = extractFeature(name,X,which=i+n_pos)
        else:
            error.append(link_df[i])
        if (i%5) == 0 and verbo: print('Extract Negative feature - Progress: '+str(i+1)+'/{}'.format(len(link_df)))
    X = np.array(X)
    y = np.zeros(X.shape[0])
    if save:
        nm = 'neg_train.pickle' if train else 'neg_test.pickle'
        pickle_out = open(nm,"wb")
        pickle.dump({'neg':(X,y,error)}, pickle_out)
        pickle_out.close()
    return(X,y,error)

# wrapper function to deal with all feature extraction
def extractWrap(file,allTags,soundID,verbo=True,save=False):
    '''
    # 1. positive
    # 2. determine number of videos need for negative
    # 2a. for each number of videos:
    # 2b. get video for neg
    # 2c. extract feature
    '''
    # 0 train or test
    train = False if re.search(r'train',file) is None else True
    if train: 
        print('This is to extract feature for training data')                          
    else:
        print('This is to extract feature for testing data')
        
    # 1 get positive
    pos_X,pos_y,pos_err = extractPositiveFeatures(train,verbo,save=False)
    n_pos = 60 # sum(pos_y)
    print('positive done, {} errors'.format(len(pos_err)))
    # 1.5 remove bark tag from allTags
    temp = [t for t in allTags if t['id']!= soundID]
    n_tag = len(temp)
    
    # 2. get negative videos
    '''
    neg - ratio of pos:neg
    need - actual video numbers needed for negative sample
    n_tag - how many other tags are there other than bark
    use - index of tags to be used later
    per - videos per tag to be download and extracted
    link_df - contains all links from the negative videos
    jump - for verbose purpose
    '''
    neg = 10 if train else 1.2
    need = int(neg*60)
    if need > n_tag: # use all tags if samples needed > allTag len
        use = np.arange(n_tag) 
        per = int(need/n_tag)
    else: # random
        np.random.seed(1) # reproducible
        use = np.random.choice(n_tag, need, replace=False)
        per = 1
    # get links
    if verbo: print('Start to get links for {} negative samples'.format(need))
    link_df = [] 
    jump = int(need/10)
    for i in range(len(use)):
        u = use[i]
        tag = temp[u]['id']
        _, tag_df = getInfo(file,tag,per)
        link_df.append(tag_df)
        if (i%jump) == 0 and verbo: 
            print('Link extract - Progress: '+str(i+1)+'/{}'.format(need))
    
    # start save video and extract, then next video
    neg_X,neg_y,neg_err = extractNegativeFeatures(link_df,n_pos,train,save=False)
    print('negative done, {} errors'.format(len(neg_err)))
    
    # 3. combine pos and neg
    X = np.concatenate((pos_X,neg_X))
    y = np.concatenate((pos_y,neg_y))
    
    # 4. save all pos and neg samples
    toSave = {'data':(X,y,pos_err,neg_err),'neg_link':link_df}
    if save:      
        nm = 'train.pickle' if train else 'test.pickle'
        pickle_out = open(nm,"wb")
        pickle.dump(toSave, pickle_out)
        pickle_out.close()
        print('Train and test feature saved')
    return(X,y,pos_err,neg_err)

###############################################################################
def loadPkl(file):
    pickle_in = open(file,"rb")
    pkl = pickle.load(pickle_in)
    data = pkl['data']
    # X,y,pos_err,neg_err = data
    return(data)