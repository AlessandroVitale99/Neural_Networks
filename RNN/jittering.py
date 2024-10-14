# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:12:08 2022

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

X=np.load("x_new_train.npy")
Y=np.load("y_new_train.npy")

seed = 106

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

tfk = tf.keras
tfkl = tf.keras.layers

Z=np.load("x_new_train.npy")
dim=X.shape

#try to "create" new data to rebalance our training dataset by jittering each data
x_jit=[]
y_jit=[]

for i in range(dim[0]):
    m=[]
    if(Y[i]!=2 and Y[i]!=6 and Y[i]!=3 and Y[i]!=9):
        for j in range(dim[1]):
            M=[]
            for k in range(dim[2]):
                M.append(X[i,j,k]+np.random.normal(0,1))
            m.append(M)
        x_jit.append(m)
        y_jit.append(Y[i])
 
                
x_jit=np.array(x_jit)
y_jit=np.array(y_jit)

X=np.concatenate((X, x_jit), axis=0)
Y=np.concatenate((Y, y_jit), axis=0)

np.save("x_jitter1.npy",X)
np.save("y_jitter1.npy",Y)







