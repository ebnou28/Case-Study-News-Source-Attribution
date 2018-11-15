#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from collections import Counter

import unidecode
import re, string, timeit
from nltk.corpus import stopwords

import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy
import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

"""
data = pd.read_csv('taonews.csv', sep = ';')
data_count = data[["domain","text"]].groupby("domain").count()
domain_names = set(data_count[data_count['text']>500].index)
new_data = data.loc[data['domain'].isin(domain_names)]
new_data = new_data.sample(n=5000)



tokenizer = Tokenizer(num_words=100000,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~0123456789\n\xa0')
tokenizer.fit_on_texts(new_data.text)
sequences = tokenizer.texts_to_sequences(new_data.text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))



#Word Embedding

# -*- coding: utf-8 -*-
embeddings_index = {}
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index)) 

word_index_key = dict(zip(word_index.values() , word_index.keys()))


def convert_vector(e , d = 100):
    try:
        return embeddings_index[word_index_key[e]]
    except:
        return np.zeros(d)


emb_sequences = [ [ convert_vector(e) for e in l ] for l in sequences]

# We pad the sequences.
fill = np.zeros(100)
max_len = 800

for l in emb_sequences:
    while len(l) < max_len:
        l.append(fill)
        
        
X = pad_sequences(emb_sequences, maxlen=800)

Y = pd.get_dummies(new_data.domain).values
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)

"""

def get_data(data_csv_path,embedding_pretrained_model_path):
    #embedding_pretrained_model_path = 'glove.6B.100d.txt' in our case
    data = pd.read_csv(data_csv_path, sep = ';')
    data_count = data[["domain","text"]].groupby("domain").count()
    domain_names = set(data_count[data_count['text']>500].index)
    new_data = data.loc[data['domain'].isin(domain_names)]
    new_data = new_data.sample(n=5000)
    
    tokenizer = Tokenizer(num_words=100000,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~0123456789\n\xa0')
    tokenizer.fit_on_texts(new_data.text)
    sequences = tokenizer.texts_to_sequences(new_data.text)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    
    #Word Embedding
    # -*- coding: utf-8 -*-
    embeddings_index = {}
    f = open(embedding_pretrained_model_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index)) 
    
    word_index_key = dict(zip(word_index.values() , word_index.keys()))
    
    
    def convert_vector(e , d = 100):
        try:
            return embeddings_index[word_index_key[e]]
        except:
            return np.zeros(d)
    
    
    emb_sequences = [ [ convert_vector(e) for e in l ] for l in sequences]
    
    # We pad the sequences.
    fill = np.zeros(100)
    max_len = 800
    
    for l in emb_sequences:
        while len(l) < max_len:
            l.append(fill)
            
            
    X = pad_sequences(emb_sequences, maxlen=800)
    
    Y = pd.get_dummies(new_data.domain).values
    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', Y.shape)
    
    return X , Y
        
    
