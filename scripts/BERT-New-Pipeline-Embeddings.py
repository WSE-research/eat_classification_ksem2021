#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # specify which GPU(s) to be used

import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time
import gc
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback 
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

tf.get_logger().setLevel('ERROR')


# # Basic functions

# In[3]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


# In[4]:


def build_model(transformer, max_len=512, hidden_dim=32, n_classes=3):
    assert n_classes > 2 # only multiclass
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer.distilbert(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    
    out = Dense(n_classes, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# In[5]:


def train_model(n_steps, train_dataset, model):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        verbose=1,
        patience=1,
        mode='min',
        restore_best_weights=True
    )
    
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    return model, train_history


# # GPU Check

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print("---------- Available GPUs:", get_available_gpus(), "--------------")


# # Hyperparameter setup

# In[8]:


AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 8
BATCH_SIZE = 16*len(get_available_gpus())
MAX_LEN = 128
MODEL = 'distilbert-base-uncased'

TRAIN_LIST = ['LC-QuAD+QALD+CogComp+WebQuestions+SimpleQuestions']

types = ['Event', 'Place', 'Colour', 'SportsSeason', 'Name', 'DateTime', 'Device', 'Activity', 'Number', 'Biomolecule', 'Disease', 'Food', 'Work', 'AnatomicalStructure', 'Currency', 'TopicalConcept', 'Species', 'Boolean', 'Award', 'TimePeriod', 'Altitude', 'Agent', 'Language', 'Flag', 'Holiday', 'ChemicalSubstance', 'MeanOfTransportation', 'Medicine', 'EthnicGroup', 'PersonFunction', 'String', 'List']

# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

encoder = LabelEncoder()
encoder.fit(types)
np.save('../../../bin_eat/encoder.npy', encoder.classes_)

for train_name in TRAIN_LIST:
    train_df = pd.read_csv(f"../../data/UnifiedSubclassDBpedia/{train_name}-train.csv", sep=';')

    encoded_y_train = encoder.transform(train_df.type)
    dummy_y_train = np_utils.to_categorical(encoded_y_train)

    x_train = regular_encode(list(train_df.questionText.astype(str).values), tokenizer, maxlen=MAX_LEN)

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, dummy_y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN, n_classes=len(types))
    n_steps = x_train.shape[0] // BATCH_SIZE

    model, train_history = train_model(n_steps, train_dataset, model)
    model.save("../../../bin_eat/classifier")
    break

        