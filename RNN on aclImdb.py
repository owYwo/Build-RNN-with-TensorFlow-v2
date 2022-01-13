# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:20:25 2022

@author: Yuwei Wang
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
import tensorflow_text
import nltk
from  tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense


#Set a memory limit, if program is run on a typical laptop.
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#-------------------------------Prepare data----------------------------------------------------
import os
basepath = 'E:/aclImdb_v1/aclImdb' #The path of dataset
import pyprind
labels = {'pos':1,'neg':0}
pbar = pyprind.ProgBar(50000) 
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path = os.path.join(basepath,s,l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path,file),'r',encoding ='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index = True)
            pbar.update()
df.columns = ['review','sentiment']
#------------------------------Create train,val,test dataset-------------------------------------
df1 = df.copy()
target = df1.pop('sentiment')
ds_raw_train_val, ds_raw_test, senti_raw_train_val, senti_raw_test = \
    train_test_split(df1.values, target.values,test_size = 0.5, random_state = 1)
ds_raw_train, ds_raw_valid, senti_raw_train, senti_raw_valid = \
    train_test_split(ds_raw_train_val, senti_raw_train_val, test_size = 0.2, random_state = 1, stratify = senti_raw_train_val)
#-----------------------------Tokenize the train dataset-----------------------------------------
token_counts = Counter()
tokenizer = TreebankWordTokenizer()
puncs = [',', '.', '--', '-', '!', '?',':', ';', '``', "''", '(', ')', '[', ']','<','/','>','...']
nltk.download('stopwords',quiet = True)
stopwords = nltk.corpus.stopwords.words('english')
Meaningless = puncs + stopwords
for example in ds_raw_train:
    #tokens = tensorflow_text.WhitespaceTokenizer().tokenize(example[0])
    tokens = tokenizer.tokenize(example[0].lower())
    tokens = [x for x in tokens if x not in Meaningless]
    token_counts.update(tokens)
#print(len(token_counts),token_counts)  
#----------------------------Encode the text and convert them to tensor--------------------------
encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)  

# A function for transformation
def encode(text_tensor,label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

#Wrap the first function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp = [text, label], Tout = (tf.int64, tf.int64))

tf_raw_train = tf.data.Dataset.from_tensor_slices((ds_raw_train,senti_raw_train))
tf_train = tf_raw_train.map(encode_map_fn)

tf_raw_valid = tf.data.Dataset.from_tensor_slices((ds_raw_valid,senti_raw_valid))
tf_valid = tf_raw_valid.map(encode_map_fn)

tf_raw_test = tf.data.Dataset.from_tensor_slices((ds_raw_test,senti_raw_test))
tf_test = tf_raw_test.map(encode_map_fn)

#Although in general RNNs can handle sequnces with different lengths, we still need to  make sure that all the sequences in a min-batch have the same length to store them efficiently in a tensor
train_data = tf_train.padded_batch(32,padded_shapes=([-1],[]))
valid_data = tf_valid.padded_batch(32,padded_shapes=([-1],[]))
test_data =  tf_test.padded_batch(32,padded_shapes =([-1],[]))

embedding_dim = 20
vocab_size = len(token_counts)+2 

bi_lstm_model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, name = 'embed-layer')
                                    ,tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name = 'lstm-layer'),name='bidir-lstm')
                                    ,tf.keras.layers.Dense(64, activation = 'relu')
                                    ,tf.keras.layers.Dense(1, activation = 'sigmoid')])

#compile and train:
bi_lstm_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3)
                     ,loss=tf.keras.losses.BinaryCrossentropy(from_logits = False)
                     ,metrics = ['accuracy'])
print(bi_lstm_model.summary())
history = bi_lstm_model.fit(train_data, validation_data = valid_data, epochs = 10)
test_results = bi_lstm_model.evaluate(test_data)

bi_lstm_model.save('bi_lstm.h5', overwrite=True, include_optimizer=True, save_format = 'h5')