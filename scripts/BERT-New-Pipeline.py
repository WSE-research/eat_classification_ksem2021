#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # specify which GPU(s) to be used

import warnings
warnings.filterwarnings('ignore')

import mlflow

EXPERIMENT_NAME = 'EAT Classification_4'
mlflow.set_tracking_uri("http://0.0.0.0:41250")
mlflow.set_experiment(EXPERIMENT_NAME)


# In[2]:


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
    sequence_output = transformer(input_word_ids)[0]
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


# In[6]:


def evaluate_model(ids, test_dataset, y_true_encoded, model):
    y_pred_encoded = np.argmax(
        model.predict(test_dataset, verbose=1),
        axis=1)
    
    assert len(y_true_encoded) == len(y_pred_encoded)
    
    f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
    precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted')
    recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    
    # getting correctly and incorrectly classified questions
    results = list()
    for i in range(len(y_true_encoded)):
        if y_pred_encoded[i] != y_true_encoded[i]:
            results.append('+')
        else:
            results.append('-')
            
    df = pd.DataFrame.from_dict({'id': ids, 'result': results, 'true': y_true_encoded, 'pred': y_pred_encoded})
    
    return f1, precision, recall, accuracy, df


# # GPU Check

# In[7]:


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
BATCH_SIZE = 8 * len(get_available_gpus())
MAX_LEN = 128
MODEL = 'distilbert-base-uncased'

TRAIN_LIST = ['LC-QuAD',
                'QALD',
                'CogComp',
                'WebQuestions',
                'SimpleQuestions',
                'LC-QuAD+QALD',
                'LC-QuAD+CogComp',
                'LC-QuAD+WebQuestions',
                'LC-QuAD+SimpleQuestions',
                'QALD+CogComp',
                'QALD+WebQuestions',
                'QALD+SimpleQuestions',
                'CogComp+WebQuestions',
                'CogComp+SimpleQuestions',
                'WebQuestions+SimpleQuestions',
                'LC-QuAD+QALD+CogComp',
                'LC-QuAD+QALD+WebQuestions',
                'LC-QuAD+QALD+SimpleQuestions',
                'LC-QuAD+CogComp+WebQuestions',
                'LC-QuAD+CogComp+SimpleQuestions',
                'LC-QuAD+WebQuestions+SimpleQuestions',
                'QALD+CogComp+WebQuestions',
                'QALD+CogComp+SimpleQuestions',
                'QALD+WebQuestions+SimpleQuestions',
                'CogComp+WebQuestions+SimpleQuestions',
                'LC-QuAD+QALD+CogComp+WebQuestions',
                'LC-QuAD+QALD+CogComp+SimpleQuestions',
                'LC-QuAD+QALD+WebQuestions+SimpleQuestions',
                'LC-QuAD+CogComp+WebQuestions+SimpleQuestions',
                'QALD+CogComp+WebQuestions+SimpleQuestions',
                'LC-QuAD+QALD+CogComp+WebQuestions+SimpleQuestions']

TEST_LIST = ['LC-QuAD', 'QALD', 'CogComp', 'WebQuestions', 'SimpleQuestions'] 

# classes
types = ['Event', 'Place', 'Colour', 'SportsSeason', 'Name', 'DateTime', 'Device', 'Activity', 'Number', 'Biomolecule', 'Disease', 'Food', 'Work', 'AnatomicalStructure', 'Currency', 'TopicalConcept', 'Species', 'Boolean', 'Award', 'TimePeriod', 'Altitude', 'Agent', 'Language', 'Flag', 'Holiday', 'ChemicalSubstance', 'MeanOfTransportation', 'Medicine', 'EthnicGroup', 'PersonFunction', 'String', 'List']


# # Create Tokenizer

# In[9]:


# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# # The Pipeline

# In[10]:


def pipeline():
    encoder = LabelEncoder()
    encoder.fit(types)
        
    for train_name in TRAIN_LIST:
        train_df = pd.read_csv(f"../../data/UnifiedSubclassDBpedia/{train_name}-train.csv", sep=';')
        
        encoded_y_train = encoder.transform(train_df.type)
        dummy_y_train = np_utils.to_categorical(encoded_y_train)
        
        x_train = regular_encode(train_df.questionText.astype(str), tokenizer, maxlen=MAX_LEN)
        
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
        
        print(f"--------- START TRAINING {train_name} ------------")
        start = time.time()
        ########
        model, train_history = train_model(n_steps, train_dataset, model)
        ########
        train_time = time.time() - start
        print(f"--------- END TRAINING {train_name}------------")
        
        with open(f"../../data/experimental_metadata/{train_name}.hist", 'w') as file:
            json.dump(train_history.history, file, indent=4)
        
        for test_name in TEST_LIST:
            test_df = pd.read_csv(f"../../data/UnifiedSubclassDBpedia/{test_name}-test.csv", sep=';')
        
            x_test = regular_encode(test_df.questionText.astype(str), tokenizer, maxlen=MAX_LEN)
            y_true_encoded = encoder.transform(test_df.type)
            
            test_dataset = (
                tf.data.Dataset
                .from_tensor_slices(x_test)
                .batch(BATCH_SIZE)
            )
            
            print(f"--------- START EVALUATION {test_name} ------------")
            start = time.time()
            #########
            f1, precision, recall, accuracy, df_meta = evaluate_model(
                test_df.question,
                test_dataset,
                y_true_encoded,
                model
            )
            #########
            inference_time = time.time() - start
            print(f"--------- END EVALUATION {test_name} ------------")
            
            df_meta.to_csv(
                f"../../data/experimental_metadata/TRAIN:{train_name} || TEST:{test_name}.csv",
                sep=';',
                index=False
            )
            
            with mlflow.start_run():
                mlflow.log_param("Train Data", train_name)
                mlflow.log_param("Test Data", test_name)
                mlflow.log_param("EPOCHS", EPOCHS)
                mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
                mlflow.log_param("MAX_LEN", MAX_LEN)
                mlflow.log_param("MODEL", MODEL)
                mlflow.log_param("Metadata Path", f"/home/aperevalov/hs-anhalt-master-thesis/data/experimental_metadata/TRAIN:{train_name} || TEST:{test_name}.csv")
                mlflow.log_param("History Path", f"/home/aperevalov/hs-anhalt-master-thesis/data/experimental_metadata/{train_name}.hist")
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1 Score", f1)
                mlflow.log_metric("Training Time", train_time)
                mlflow.log_metric("Inference Time", inference_time)
                
                print("F1", f1)


# In[11]:


pipeline()


# In[ ]:




