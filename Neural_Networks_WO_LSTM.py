#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Loading the packages

import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import string
from collections import Counter
import math
import nltk
_wnl = nltk.WordNetLemmatizer()
import pandas as pd
from gensim.matutils import kullback_leibler
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import string
import pandas as pd
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten, Bidirectional, LSTM,BatchNormalization

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils 

### Loading the tfidf feature generated in feature engineering

import pickle
X_tfidf = pickle.load(open( "features//tfidf_25k.pkl","rb"))

X_train_tfidf=X_tfidf[:49972]
X_test_tfidf=X_tfidf[49972:]

### Loading the other features

X_features=pd.read_csv("features//features.csv")

## Splitting the dataset into training and test set
X_train=X_features.iloc[:49972,]
X_test=X_features.iloc[49972:,]

Stance=pd.read_csv("features//combinedf.csv")
Stance=Stance['Stance']

Stance1=Stance.replace({'unrelated':3})
Stance1=Stance1.replace({'agree':0})
Stance1=Stance1.replace({'disagree':1})
Stance1=Stance1.replace({'discuss':2})

Y_train=Stance1.iloc[:49972,]
Y_test=Stance1.iloc[49972:,]

#### Scipy matrix for models

import scipy
from scipy.sparse import coo_matrix, hstack
g_train=scipy.sparse.csr_matrix(X_train)
g_test=scipy.sparse.csr_matrix(X_test)

# Use these features for model with tfidf vectorizer

x_train=hstack([X_train_tfidf,g_train])
x_test=hstack([X_test_tfidf,g_test])

pred1=Y_test.values.ravel()

from keras.utils import np_utils 
YT=np_utils.to_categorical(Y_train)

### FUNCTION TO CALCULATE SCORE

def scorer(preds):
    sco=[]
    pred1=Y_test.values.ravel()
    #print(accuracy_score(preds,pred1))
    for i in range(len(pred1)):
        if ((pred1[i]==3) & (preds[i]==3)):
            sco.append(0.25)
        elif ((pred1[i]==1) & (preds[i]==1)):
            sco.append(1)
        elif ((pred1[i]==2) & (preds[i]==2)):
            sco.append(1)
        elif ((pred1[i]==0) & (preds[i]==0)):
            sco.append(1)
        elif ((pred1[i]==1) & (preds[i]==2)):
            sco.append(0.25)
        elif ((pred1[i]==1) & (preds[i]==0)):
            sco.append(0.25)
        elif ((pred1[i]==2) & (preds[i]==1)):
            sco.append(0.25)
        elif ((pred1[i]==2) & (preds[i]==0)):
            sco.append(0.25)
        elif ((pred1[i]==0) & (preds[i]==1)):
            sco.append(0.25)
        elif ((pred1[i]==0) & (preds[i]==2)):
            sco.append(0.25)
    return sum(sco),accuracy_score(preds,pred1)

# NEURAL NETWORK

import keras
from sklearn import preprocessing #scaling
import keras
from keras.layers import Dense    #for Dense layers
from keras.layers import BatchNormalization #for batch normalization
from keras.layers import Dropout            #for random dropout
from keras.models import Sequential #for sequential implementation
from keras.optimizers import Adam   #for adam optimizer
from keras import regularizers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils 
YT=np_utils.to_categorical(Y_train)

import numpy as np
X_tradin=x_train.todense()
X_testin=x_test.todense()

import pickle
with open('features//X_tradin.pkl','wb') as pi:
    pickle.dump(X_tradin,pi)

import pickle
with open('features//X_testin.pkl','wb') as pi:
    pickle.dump(X_testin,pi)

import keras
optimizer = keras.optimizers.Adam(lr=0.004)

import scipy.sparse
x_traindf=pd.DataFrame.sparse.from_spmatrix(x_train)
x_testdf=pd.DataFrame.sparse.from_spmatrix(x_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils 
YT=np_utils.to_categorical(Y_train)
model_N = Sequential()
model_N.add(Dense(100,input_shape=(X_tradin.shape[1],)))
model_N.add(Dropout(0.8))
model_N.add(Activation('sigmoid'))
model_N.add(Dense(4))
model_N.add(Activation('sigmoid'))
model_N.summary()
model_N.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history=model_N.fit(X_tradin,YT,batch_size=200,epochs=40)
pred_N=model_N.predict_classes(X_testin)
res_N,acc_N=scorer(pred_N)
print(res_N)

### Confusion Matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(pred_NN, pred1)

### Classification Report

from sklearn.metrics import classification_report
print(classification_report(pred_NN,pred1))

### PLOTS

plot_model(model_NN,show_layer_names=True, show_shapes=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracies')
plt.xlabel('No. of the Epoch')
plt.legend(['training', 'testing'], loc='upper left')
plt.show()

model_N.save("neural")
predicted=pd.DataFrame(pred_N)
predicted=predicted.replace({3:'unrelated',2:'discuss',1:'disagree',0:'agree'})
predicted.columns=['Stance']
df=pd.read_csv("data//competition_test_stances_unlabeled.csv")
result = pd.DataFrame(predicted)
ndf=pd.concat([df,result],axis=1)

