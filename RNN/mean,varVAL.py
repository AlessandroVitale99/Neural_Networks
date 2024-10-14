# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:14:28 2022

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

X=np.load("x_validation.npy")
Y=np.load("y_new_train.npy")

seed = 106

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

tfk = tf.keras
tfkl = tf.keras.layers

#find mean and variance of our training data
dim=X.shape
X_var_val=[[0 for i in range(6)] for j in range(dim[0])]
X_mean_val=[[0 for i in range(6)] for j in range(dim[0])]

#mean
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_mean_val[j][i]=X[j,k,i]+X_mean_val[j][i]

X_mean_val=np.array(X_mean_val)/36

#variance
for i in range(dim[2]):
    for j in range(dim[0]):
        for k in range(dim[1]):
            X_var_val[j][i]=(X[j,k,i]-X_mean_val[j][i])**2+X_var_val[j][i]


X_var_val=np.array(X_var_val)/35


np.save("x_mean_val.npy",X_mean_val)
np.save("x_var_val.npy",X_var_val)


