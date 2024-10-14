# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:32:06 2022

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

X_mean_class=[[0 for i in range(6)] for j in range(12)]
X_var_class=[[0 for i in range(6)] for j in range(12)]

dim=X.shape

tot=dim[2]*dim[1]

Y=[i for i in Y]

#find mean and variance for two different things: for each label and for each time series
#mean for each label
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_mean_class[Y[j]][i]=X[j,k,i]+X_mean_class[Y[j]][i]

X_mean_class=np.array(X_mean_class)/tot

#variance for each label
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_var_class[Y[j]][i]=(X[j,k,i]-X_mean_class[Y[j]][i])**2+X_var_class[Y[j]][i]
            

X_mean_class=np.array(X_mean_class)/tot
X_var_class=np.array(X_var_class)/(tot-1)

X_var=[[0 for i in range(6)] for j in range(dim[0])]
X_mean=[[0 for i in range(6)] for j in range(dim[0])]

#mean for each time series
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_mean[j][i]=X[j,k,i]+X_mean[j][i]

X_mean=np.array(X_mean)/36

#variance for each time series
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_var[j][i]=(X[j,k,i]-X_mean[j][i])**2+X_var[j][i]


X_var=np.array(X_var)/35


#np.save("x_mean_classes.npy",X_mean_class)
#np.save("x_var_classes.npy",X_var_class)
#np.save("x_mean.npy",X_mean)
#np.save("x_var.npy",X_var)





