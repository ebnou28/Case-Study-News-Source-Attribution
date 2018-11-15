#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:35:11 2018

@author: ebnou
"""

"""
PARAMETERS:
X : Features Matrix 
Y : Target Matrix (Binary : Classification)
-----
cnn_layers = list / Size of each cnn layer in the architechture
cnn_kernels = list / Kernel of each cnn layer in the architechture
cnn_dropout = list / dropout rate of each cnn layer
-----
lstm_layers = list / Size of each Bidirectional LSTM.
lstm_dropout = list / Dropout rate of each Bidirectionnal LSTM.
-----
Vector_size : int / Size of the first Fully connected after the concatenation of the 
              Vectors. Will be the feature vector that the model extract.
-----
Training parameters:
    lr : learning rate.
    epochs : Number of epochs.
    batcj_size : batch_size.
    ntest_sers : Number of sequences that we use for validation.
-----
Verbose : Wether or not print the summary.
"""
#### model parameters ####    
cnn_layers = [32 , 64 , 128]
cnn_kernels = [3 , 3 , 3]
cnn_dropout = [.5 , .5 , .5]
lstm_layers = [128] 
lstm_dropout = [.5]
vector_size = 128
lr = 0.001
epochs = 20
batch_size = 64
ntest_sers = 1000
verbose = True
    
####### preprocessing and data ##########

data_csv_path = 'data/taonews.csv'
embedding_pretrained_model_path = 'data/glove.6B.100d.txt'



##################################

from preprocessing import get_data
from model import model, train

X, Y = get_data(data_csv_path,embedding_pretrained_model_path) 
model = model(X,Y,cnn_layers,cnn_kernels,cnn_dropout,lstm_layers,lstm_dropout,vector_size)
#training
train(model,X,Y,lr,epochs,batch_size,ntest_sers, verbose=True)

