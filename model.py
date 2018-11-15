#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential , load_model , Model
from keras.layers import Dense , LSTM, Dropout , Conv1D , MaxPooling1D , Input, Reshape , Masking , TimeDistributed
from keras.layers import Concatenate , BatchNormalization , Bidirectional , Activation , GlobalMaxPooling1D
from keras.preprocessing.sequence import TimeseriesGenerator , pad_sequences
from keras.callbacks import History , ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.constraints import maxnorm
from keras.regularizers import l1_l2

from keras.layers import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , precision_score , accuracy_score
from sklearn.metrics import recall_score , confusion_matrix, roc_curve, auc

from keras.metrics import top_k_categorical_accuracy

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
    

    
def model(X,Y,cnn_layers,cnn_kernels,cnn_dropout,lstm_layers,lstm_dropout,vector_size):
    ######### Model :    
    inp = Input(shape = (X.shape[1],X.shape[2]))
    
    
    # Convolutional Part
    
    x1 = inp
    # We apply the convolutional Filters.
    for i in range(len(cnn_layers)):
        x1 = Conv1D(cnn_layers[i] , cnn_kernels[i], padding = 'same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(cnn_dropout[0])(x1)
    # Global Pooling.
    x1 = GlobalMaxPooling1D()(x1)
    
    
    
    
    # BI lSTM part :
    x2 = Bidirectional( LSTM(lstm_layers[0] , return_sequences = False))(inp)
    x2 = Dropout(lstm_dropout[0])(x2)
    
    # Concatenation:
    x = Concatenate(axis = -1)([x1 , x2])
    
    # Dense layer : Feature Vector.
    x = Dense(vector_size , activation = 'relu')(x)
    
    # Output 
    x = Dense(Y.shape[1] , activation = 'softmax')(x)
    
    model = Model(inputs=inp, outputs=x)
    
    return model

def train(model,X,Y,lr,epochs,batch_size,ntest_sers, verbose=True):

    #####################
    
    # Model Summary
    if verbose : print(model.summary())
    
    # Optimize & Compilation.
    Nadam = keras.optimizers.Nadam(lr = lr , beta_1=0.9, beta_2=0.999, epsilon=1e-08)#, schedule_decay=0.0004)
    model.compile(loss='categorical_crossentropy', optimizer= Nadam , metrics = ['accuracy',top_k_categorical_accuracy])
    
    # We fit the model.
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    
    model.fit(X, Y, epochs= epochs, batch_size=batch_size, validation_split = ntest_sers/X.shape[0] 
              , callbacks = [checkpoint])