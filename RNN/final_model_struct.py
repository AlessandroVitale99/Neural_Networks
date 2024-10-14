# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:21:37 2022

@author: ale
"""

import random
import os
os.chdir('C:/Users/vital/Desktop/homework2')
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', size=16) 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
tfk = tf.keras
tfkl = tf.keras.layers

X_train=np.load("x_jitter1.npy")
y_train=np.load("y_jitter1.npy")
X_val=np.load("x_validation.npy")
Y_val=np.load("y_validation.npy")

#X_train=np.load("x_train.npy")
#y_train=np.load("y_train.npy")

#nb I had weird results for the min max scaling, that's why I wrote similar
#functions a lot of time, I wasn't sure what was going on

def min_max(X):
    dim=X.shape
    abs_max=[-100000 for i in range(dim[2])]
    abs_min=[100000 for i in range(dim[2])]
    for i in range(dim[0]):
        for j in range(dim[2]):
            if(abs_max[j]<max(X[i,:,j])):
                abs_max[j]=max(X[i,:,j])
            if(abs_min[j]>min(X[i,:,j])):
                abs_min[j]=min(X[i,:,j])
    return abs_min,abs_max
    
def min_max_scaling(X,m,M):
    dim=X.shape
    for i in range(dim[2]):
        X[:,:,i]=(X[:,:,i]-m[i])/(M[i]-m[i])
    return X

def mean(X):
    dim=X.shape
    means=[0 for i in range(dim[2])]
    
    for i in range(dim[2]):
        means[i]=np.mean(X[:,:,i])
    
    return means


def stdev(X):
    dim=X.shape
    std=[0 for i in range(dim[2])]
    
    for i in range(dim[2]):
        std[i]=np.std(X[:,:,i])
    
    return std

    

#max and min of the training set, so I don't have to find them every time
max_train=[32932.0,44394.0,37032.0,38086.0,38490.0,53020.0]
min_train=[-3420.0,-11585.0,-10289.0,-8009.9,-7326.6,-7584.1]

#mean and std dev of the training set so that I don't have to compute them each time,
#moreover I need those value for the submission
X_mean=[9.50595923040561,19.504463303268277,28.346173613703364,34.05115329570753,35.94713693748093,34.15311564603029]
X_std=[260.3083580496339,809.3848172649348,727.1158590403159,712.5119678652364,701.3204811906129,767.0129964214946]

dim=X_train.shape
for i in range(dim[2]):
    X_train[:,:,i]=(X_train[:,:,i]-X_mean[i])/X_std[i]

for i in range(dim[2]):
    X_val[:,:,i]=(X_val[:,:,i]-X_mean[i])/X_std[i]

# Convert the sparse labels to categorical values

y_train = tfk.utils.to_categorical(y_train,12,"int32")
y_val = tfk.utils.to_categorical(Y_val,12,"int32")

input_shape = X_train.shape[1:]
classes = 12
batch_size = 64
epochs = 400

#since I use spyder and not colab I use only one file for creating NNs so the model could not be 100% accurate
def Mix_NN(input_shape, classes):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor



    lstm = tfkl.LSTM(128, return_sequences=True)(input_layer)
    lstm = tfkl.LSTM(128)(lstm)
    dropout1 = tfkl.Dropout(.5)(lstm)
    
    bilstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(input_layer)
    bilstm = tfkl.Bidirectional(tfkl.LSTM(128))(bilstm)
    dropout2 = tfkl.Dropout(.5)(bilstm)

    cnn = tfkl.Conv1D(128,3,padding='same',activation='relu')(input_layer)
    cnn = tfkl.AveragePooling1D()(cnn)
    cnn = tfkl.Conv1D(256,3,padding='same',activation='relu')(cnn)
    cnn = tfkl.AveragePooling1D()(cnn)
    cnn = tfkl.Conv1D(512,3,padding='same',activation='relu')(cnn)
    cnn = tfkl.AveragePooling1D()(cnn)
    cnn = tfkl.Conv1D(1024,3,padding='same',activation='relu')(cnn)
    gap = tfkl.GlobalAveragePooling1D()(cnn)
    dropout3 = tfkl.Dropout(.5)(gap)

    cnn = tfkl.Conv1D(128,3,padding='same',activation='relu')(input_layer)
    cnn = tfkl.MaxPooling1D()(cnn)
    cnn = tfkl.Conv1D(256,3,padding='same',activation='relu')(cnn)
    cnn = tfkl.MaxPooling1D()(cnn)
    cnn = tfkl.Conv1D(512,3,padding='same',activation='relu')(cnn)
    gap = tfkl.GlobalAveragePooling1D()(cnn)
    dropout4 = tfkl.Dropout(.5)(gap)
    


    classifier1 = tfkl.Dense(128, activation='relu')(dropout1)
    classifier2 = tfkl.Dense(128, activation='relu')(dropout2)
    classifier3 = tfkl.Dense(128, activation='relu')(dropout3)
    classifier4 = tfkl.Dense(256, activation='relu')(dropout4)
    
    conc=tfkl.Concatenate(axis=1)([classifier1,classifier2,classifier3,classifier4])
    classifier=tfkl.Dense(256, activation='relu')(conc)
    drop= tfkl.Dropout(.5)(classifier)

    # Classifier

    output_layer = tfkl.Dense(classes, activation='softmax')(drop)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model


model = Mix_NN(input_shape, classes)
model.summary()

# Train the model
history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data=(X_val,y_val),
    callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=15, restore_best_weights=True),
        tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=7, factor=0.5, min_lr=1e-5)
    ]
).history















