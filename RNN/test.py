# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:22:34 2022

@author: ale
"""

#try function
import numpy as np
temp=[[[1.0,0,3,-1,5],[0,0,0,0,0],[-5,-5,-5,-5,-5]],[[10,-4,-4,-4,-4],[-1,-1,-1,-1,-1],[-7,9,1,1,1]]]
M=[[[1,0,3,-1,5],[0,0,0,0,0],[-5,-5,-5,-5,-5]],[[10,-4,-4,-4,-4],[-1,-1,-1,-1,-1],[-7,9,1,1,1]]]
M=np.array(M)
temp=np.array(temp)

def min_max(X):
    dim=X.shape
    a_max=[-100000 for i in range(dim[2])]
    a_min=[100000 for i in range(dim[2])]
    for i in range(dim[0]):
        for j in range(dim[2]):
            if(a_max[j]<max(X[i,:,j])):
                a_max[j]=max(X[i,:,j])
            if(a_max[j]>min(X[i,:,j])):
                a_min[j]=min(X[i,:,j])
    
    for i in range(dim[0]):
        for j in range(dim[2]):
            X[i,:,j]=(X[i,:,j]-a_min[j])/(a_max[j]-a_min[j])

min_max(temp)