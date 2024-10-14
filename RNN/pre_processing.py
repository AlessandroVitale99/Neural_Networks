# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:06:21 2022

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


X_train=np.load("x_new_train.npy")
Y_train=np.load("y_new_train.npy")
X_val=np.load("x_validation.npy")
Y_val=np.load("y_validation.npy")
Z_val=np.load("x_validation.npy")

#min max scaler first iteration+mean and std finder

def min_max(X,val=0,abs_max=0,abs_min=0):
    dim=X.shape
    if val==0:

        abs_max=[-100000 for i in range(dim[2])]
        abs_min=[100000 for i in range(dim[2])]
        for i in range(dim[0]):
            for j in range(dim[2]):
                if(abs_max[j]<max(X[i,:,j])):
                    abs_max[j]=max(X[i,:,j])
                if(abs_min[j]>min(X[i,:,j])):
                    abs_min[j]=min(X[i,:,j])
    
        for i in range(dim[0]):
            for j in range(dim[2]):
                if(abs_min[j]==abs_max[j]):
                    X[i,:,j]=0
                else:
                    X[i,:,j]=(X[i,:,j]-abs_min[j])/(abs_max[j]-abs_min[j])
        return [abs_min,abs_max]

    for i in range(dim[0]):
        for j in range(dim[2]):
            if(abs_min[j]==abs_max[j]):
                X[i,:,j]=0
            else:
                X[i,:,j]=(X[i,:,j]-abs_min[j])/(abs_max[j]-abs_min[j])
    

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


X_mean=[9.50595923040561,19.504463303268277,28.346173613703364,34.05115329570753,35.94713693748093,34.15311564603029]
X_std=[260.3083580496339,809.3848172649348,727.1158590403159,712.5119678652364,701.3204811906129,767.0129964214946]
X=X_train
X_new=[]
dim=X.shape
for i in range(dim[0]):
    m=[]
    for j in range(dim[1]):
        M=[]
        for k in range(dim[2]):
            M.append((X[i,j,k]-X_mean[k])/X_std[k])
        m.append[M]
    X_new.append(m)
           










    
            
        