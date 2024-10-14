# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
tfk = tf.keras
tfkl = tf.keras.layers
from keras.utils import to_categorical
from keras.datasets import mnist
directory = r"C:\Users\vital\Desktop\Homework1\training_data_final"


from tensorflow.keras.preprocessing.image import ImageDataGenerator

#pre-processing
data_gen_train = ImageDataGenerator(rotation_range=90,
                              height_shift_range=0,
                              width_shift_range=0,
                              zoom_range=0.,
                              horizontal_flip=True,
                              vertical_flip=False,
                              fill_mode='reflect',
                              rescale=1/255,
                              brightness_range=(0.75,1.),
                              shear_range=0.)

dir_train = r"C:\Users\vital\Desktop\Homework1\Training"

gen_train = data_gen_train.flow_from_directory(directory=dir_train,
                                               target_size=(96,96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=64,
                                               shuffle=True) # set as training

data_gen_val = ImageDataGenerator(rescale=1/255)

dir_val = r"C:\Users\vital\Desktop\Homework1\Validation"

gen_test = data_gen_val.flow_from_directory(directory=dir_val,
                                               target_size=(96,96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=64,
                                               shuffle=True) # set as validation

#parameters and sets

batch_size = 64
epochs = 30
k_size=(3,3)
max_conv_layers=15
strd=(1,1)
n_filters=[32,64,128,256,512,1024]
max_perceptron_layers=7
perceptron_neurons_per_layer=[32,64,128,256,512,1024]
unit=8 #number of labels
n=30 #population
kill_rate=0.4
mutation_rate=4
iterations=60


#compute a CNN given the layers per convolutional layer and the neurons per layer in the non convolutional part
#for example filters_per_layer=[32,64,128]->3 convolutional layers with 32,64 and 128 layers in this order
#same logic holds for the neurons_per_layer
def compute_model(filters_per_layer=None,neurons_per_layer=None,input_shape=None):
        input_layer = tfkl.Input(shape=input_shape, name='Input')
        
        #nb it could have been implemented without this vector
        structure=[input_layer]
    
        for i in range(len(2*filters_per_layer)):
            layer=tfkl.Conv2D(
                filters=filters_per_layer[int(i/2)],
                kernel_size=k_size,
                strides = strd,
            padding = 'same',
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform()
            )(structure[-1]) if i%2==0 else tfkl.AveragePooling2D(pool_size = (2, 2))(structure[-1])
            structure.append(layer)
        
        global_avg = tfkl.GlobalAveragePooling2D()(structure[-1])
        structure.append(global_avg)
        
        for i in range(len(neurons_per_layer)):
            layer= tfkl.Dense(
                            units=neurons_per_layer[i], 
                            activation='relu',
                            kernel_initializer = tfk.initializers.HeUniform()
                            )(structure[-1])
            structure.append(layer)
            if i%2==1 and i<len(neurons_per_layer)-1:
                drp = tfkl.Dropout(0.2)(structure[-1])
                structure.append(drp)
            
        classifier_layer = tfkl.Dropout(0.3)(structure[-1])
        structure.append(classifier_layer)
        
        btc=tfkl.BatchNormalization()(structure[-1])
        structure.append(btc)
        
        output_layer = tfkl.Dense(
        units=unit, 
        activation='softmax', 
        kernel_initializer = tfk.initializers.GlorotUniform(),
        name='Output'
    )(structure[-1])
        structure.append(output_layer)
        
        model = tfk.Model(inputs=structure[0], outputs=structure[-1], name='model')

        model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

        # Return the model
        return model 


#class NN contains the model of the NN, the corresponding gene, the history of the NN and the score of the NN.
#the gene here is viewed as a vector of vectors containing the number of filters per layer and the neurons per layer
#for example [[32,64,128],[128,128]] is a possible gene
#the score of the NN is computed ad the accuracy on the validation set
#two operatos, == and < are implemented, the first one tells us that two NN are equivalent if they have the same gene->
#same structure
#the second is tells us that NN1<NN2 if NN1 has a lower score than NN2
class NN():
    def __init__(self,filters_per_layer=None,neurons_per_layer=None,gen_train=None,gen_test=None):
        self.model=compute_model(filters_per_layer,neurons_per_layer,(96,96,3))
        self.gene=[filters_per_layer,neurons_per_layer]
        self.hist = self.model.fit(
    x = gen_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = gen_test,
    callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
).history
        self.score=max(self.hist["val_accuracy"])
        print(self.score)
        
    def __eq__(self,other):
        return self.gene==other.gene
    
    def __lt__(self,other):
        return self.score<other.score


#class polulation contains a dictionary containing as key the gene of the NN converted to string, letters C and P are used
#to say when each part begins and end. For example C3264128P128128 means that the gene is [[32,64,128],[128,128]],
#and as value the NN object itself, this is used so that every structure is computed only once
class population():
    def __init__(self):
        self.history=dict()
        #the vector pop is a vector of NN, it is randomly initialized and kept sorted by bubble sort+ < operator of the NN
        self.pop=[self.player() for i in range(n)]
        self.pop=bubble_sort(self.pop)
        
    #randomly initialize the population of the NN
    def player(self):
        filt=random.choice([i+1 for i in range(max_conv_layers)])
        filters=[random.choice(n_filters) for i in range(filt)]
        lay=random.choice([i+1 for i in range(max_perceptron_layers)])
        neurons=[random.choice(perceptron_neurons_per_layer) for i in range(lay)]
        f="C"+"".join([str(i) for i in filters])
        n="P"+"".join([str(i) for i in neurons])
        if f+n in self.history:
            return self.history[f+n]
        N1=NN(n_filters,neurons,gen_train,gen_test)
        self.history[f+n]=N1
        return N1
    
    #kill kill_rate% of the population with the lowest score-> only the last kill_rate*n since it is sorted
    def kill(self):
        self.pop=self.pop[:-int(kill_rate*n)]
        
    #mutation is viewed as a random mutation starting from the best individual so far, to speed up the algo
    def mutate(self):
        y=random.choice([1,2])
        if y==1:
            filt=random.choice([i+1 for i in range(max_conv_layers)])
            filters=[random.choice(n_filters) for i in range(filt)]
            f="C"+"".join([str(i) for i in filters])
            n="P"+"".join([str(i) for i in self.pop[0].gene[1]])
            if f+n in self.history:
                return self.history[f+n]
            N1=NN(n_filters,self.pop[0].gene[1],gen_train,gen_test)
            self.history[f+n]=N1
            return N1

        if y==2:
            lay=random.choice([i+1 for i in range(max_perceptron_layers)])
            neurons=[random.choice(perceptron_neurons_per_layer) for i in range(lay)]
            f="C"+"".join([str(i) for i in self.pop[0].gene[0]])
            n="P"+"".join([str(i) for i in neurons])
            if f+n in self.history:
                return self.history[f+n]
            N1=NN(self.pop[0].gene[0],neurons,gen_train,gen_test)
            self.history[f+n]=N1
            return N1
        
    #reproduction/crossover is viewed as randomly getting subvectors of the vector gene of the parents
    #for example if parent1=[[32,64,128][128,128]] and parent2=[[64,1024],[512]] their child 
    #can inherit for the convolutional part [32,64,128] or [64,1024] and for the perceptron part [128,128] or [512]
    def reproduce(self,l):
        y=1 if random.choice([i+1 for i in range(mutation_rate)])==1 else 0
        if y:
            P=self.mutate()
        else:
            P1=random.choice(self.pop[0:l])
            P2=random.choice(self.pop[0:l])
            new_filt=random.choice([P1.gene[0],P2.gene[0]])
            new_lay=random.choice([P1.gene[1],P2.gene[1]])
            f="C"+"".join([str(i) for i in new_filt])
            n="P"+"".join([str(i) for i in new_lay])
            if f+n in self.history:
                return self.history[f+n]
            else:
                P=NN(new_filt,new_lay,gen_train,gen_test)
                self.history[f+n]=P
        return P
    
    #refill the population after having killed kill_rate individuals
    def refill(self):
        l=len(self.pop)
        while len(self.pop)<n:
            self.pop.append(self.reproduce(l))
        self.pop=bubble_sort(self.pop)
    
    #print the best individual found so far
    def prnt(self):
        f="C"+"".join([str(i) for i in self.pop[0].gene[0]])
        n="P"+"".join([str(i) for i in self.pop[0].gene[1]])
        print(f+n)
    
    
def bubble_sort(our_list):
    for i in range(len(our_list)):
        for j in range(len(our_list)-1):
            if our_list[j] < our_list[j+1]:
                our_list[j], our_list[j+1] = our_list[j+1], our_list[j]
    return our_list
        

p=population() #random initialization
for i in range(iterations):
            p.kill()
            p.refill()
            p.prnt()
            print(p.pop[0].score)

    
    
            
            
            
            
            
            
            
            
            
        
    