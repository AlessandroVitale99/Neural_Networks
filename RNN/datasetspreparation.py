# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:03:58 2022

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

#find the proportions of labels
def proportion(classes,l):
    prop=[0 for i in range(classes)]
    for i in range(classes):
        total=0
        temp=0
        for j in range(len(l)):
            if(l[j]==i):
                temp=temp+1
            total=total+1
        prop[i]=temp/total
    return prop
                
    
# Random seed for reproducibility
seed = 106

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

tfk = tf.keras
tfkl = tf.keras.layers

X=np.load("x_train.npy")
Y=np.load("y_train.npy")

#create a list
l=[]

#join labels and data and shuffle them together
for i in range(Y.shape[0]):
        l.append([X[i],Y[i]])
        
random.shuffle(l)



Xnew=[]
Ynew=[]

#take the training and validation set
for i in range(len(Y)):
    Xnew.append(l[i][0])
    Ynew.append(l[i][1])
    
Xnew=np.array(Xnew)
Ynew=np.array(Ynew)

pp=proportion(12,Y)
pp1=proportion(12,Ynew[0:2186])

#np.save("x_new_train.npy",Xnew[0:2186])
#np.save("y_new_train.npy",Ynew[0:2186])
#np.save("x_validation.npy",Xnew[2186:])
#np.save("y_validation.npy",Ynew[2186:])

    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    