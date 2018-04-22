#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 08:49:03 2018

@author: MikeLam
"""

# 1.fully connect
def vanilla(input_shape, dim, l='binary_crossentropy', opt='adam',
            met=['binary_accuracy'], dropout=0.5):
    '''
    dim is a list of hidden nodes, excluding input and output
    '''
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Dropout
    
    model = Sequential()
    # input layer
    model.add(Flatten(input_shape=input_shape))
    # rest layer
    for d in dim:
        model.add(Dense(d,activation='relu'))
        model.add(Dropout(dropout))
    # output layer
    model.add(Dense(1,activation='sigmoid'))    
    
    # compile
    model.compile(loss=l,optimizer=opt,metrics=met)
    return(model)

# 2.CNN
def CNN(dense, input_shape, l='binary_crossentropy', opt='adam',
        met=['binary_accuracy'], dropout=0.5):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Dropout
    from keras.layers import Conv2D, MaxPooling2D
    
    # expand dim, since it is expecting an image with 3 channels
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 3)
    
    model = Sequential()
    # input layer
    model.add(Conv2D(filters=8, kernel_size=2, strides=1, padding='same',
                     activation='relu', input_shape=input_shape))
    # rest layer
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=4, strides=4, padding='same'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=8, strides=8, padding='same'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=8, strides=8, padding='same'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=8, strides=8, padding='same'))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=8, strides=8, padding='same'))
    # output layer
    model.add(Flatten())
    for den in dense:
        model.add(Dense(den, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile
    model.compile(loss=l, optimizer=opt, metrics=met)
    return(model)
    
# 3.Transfer Learning
def transferLearning(keep, untrain, dense, input_shape, l='binary_crossentropy', 
                     opt='adam', met=['binary_accuracy'], dropout=0.5):
    from keras.applications.vgg16 import VGG16
    from keras.layers import Flatten, Dense, Dropout, GlobalMaxPooling2D
    from keras.models import Model
    
    # expand dim, since it is expecting an image with 3 channels
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 3)
        
    # input layer
    vgg = VGG16(weights='imagenet', include_top=False, 
                     input_shape=input_shape)
    untrain = min(len(vgg.layers), untrain)
    # trainable layer(s) ***** model is a tensor
    for layer in vgg.layers[:untrain]:
        layer.trainable = False

    model = vgg.layers[keep].output
    model = GlobalMaxPooling2D()(model)
    # model = Flatten()(model)
    for den in dense:
        model = Dense(den, activation='relu')(model)
        model = Dropout(dropout)(model)
    # output layer
    output = Dense(1,activation='sigmoid')(model)
    
    # final model
    model_final = Model(input = vgg.input, output = output)
    
    # compile (choose optimizer)
    model_final.compile(loss=l, optimizer=opt, metrics=met)
    return(model_final)
