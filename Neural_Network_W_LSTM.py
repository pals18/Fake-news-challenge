#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Importing the packages

import pandas as pd
import os
import sys
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, Dropout, Activation, Flatten, Bidirectional, LSTM,BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models import Word2Vec
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from keras.utils import plot_model 

### Reading the tokenized dataframe

df=pd.read_csv("features//combinedf.csv")

### Splitting into training and testing

X_train=df['combined'].iloc[:49972,]
X_test=df['combined'].iloc[49972:,]

df['Stance']=df['Stance'].replace({'unrelated':3})
df['Stance']=df['Stance'].replace({'agree':0})
df['Stance']=df['Stance'].replace({'disagree':1})
df['Stance']=df['Stance'].replace({'discuss':2})

Y_train=df['Stance'].iloc[:49972,]
Y_test=df['Stance'].iloc[49972:,]

Y_train=to_categorical(np.asarray(Y_train))

### CREATING A WORD2VEC MODEL

w2vmodel=Word2Vec(df['combined'],min_count=1)

w2vmodel.wv.save_word2vec_format('word2vec//w2v.txt', binary=False)

f = open('word2vec//glove.twitter.27B.50d.txt', encoding = "utf-8")
embeddings_index = {};
import numpy as np
for line in f:
    vals = line.split()
    word = vals[0]
    coeffs = np.asarray(vals[1:])
    embeddings_index[word] = coeffs
    
f.close()

filter_list = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

tokenizer = Tokenizer(num_words=40000, filters=filter_list)
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
train_tok = tokenizer.texts_to_sequences(X_train)
test_tok = tokenizer.texts_to_sequences(X_test)

max_length_arr = [len(s) for s in (train_tok + test_tok)]
max_length = max(max_length_arr)

max_length = 150 
MAX_VOCAB_SIZE = 40000 
LSTM_DIM = 100
EMBEDDING_DIM = 50
BATCH_SIZE = 128
N_EPOCHS = 40 

x_train = pad_sequences(train_tok, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(test_tok, maxlen=max_length, padding='post', truncating='post')

embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, len(coeffs)))

for word, i in tokenizer.word_index.items(): 
    try:
        embeddings_vector = embeddings_index[word]
    except KeyError:
        embeddings_vector = None
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector

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

### NEURAL NETWORK

model0 = Sequential()
model0.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=max_length,
                          weights = [embeddings_matrix], trainable=False))
model0.add(Dense(64, activation='relu'))
model0.add(Flatten())
model0.add(Dropout(0.8))
model0.add(Dense(4, activation = 'sigmoid'))
model0.compile(loss = 'binary_crossentropy', optimizer = 'Adam',metrics=['accuracy'])

history=model0.fit(x_train, Y_train, validation_split=0.1,batch_size=200, epochs=40)

pred_NN=model0.predict_classes(x_test)
res_NN,acc_NN=scorer(pred_NN)
print(res_NN)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracies')
plt.xlabel('No. of the Epoch')
plt.legend(['training', 'testing'], loc='upper left')
plt.show()

### NEURAL NETWORK WITH LSTM BIDIRECTIONAL

model1 = Sequential()
model1.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=max_length,
                          weights = [embeddings_matrix], trainable=False))
model1.add(Bidirectional(LSTM(LSTM_DIM, return_sequences=False)))
model1.add(Dropout(rate=0.8))
model1.add(Dense(4, activation='softmax'))

model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history2=model1.fit(x_train, Y_train,validation_split=0.1,batch_size=1024,epochs=10)

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracies')
plt.xlabel('No. of the Epoch')
plt.legend(['training', 'testing'], loc='upper left')
plt.show()

pred_NN=model1.predict_classes(x_test)
res_NN,acc_NN=scorer(pred_NN)
print(res_NN)

### IMPORVED LSTM MODEL 

model2 = Sequential()
model2.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                          output_dim=len(coeffs), input_length=max_length,
                          weights = [embeddings_matrix], trainable=False))

model2.add(LSTM(LSTM_DIM, return_sequences=False))
model2.add(Dropout(rate=0.8)) 
model2.add(Activation(activation='relu'))
model2.add(Dropout(rate=0.2))
model2.add(Activation(activation='relu'))
model2.add(Dense(4, activation='softmax'))
model2.summary()

model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history2=model2.fit(x_train, Y_train,validation_split=0.1,batch_size=1024,epochs=10)

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracies')
plt.xlabel('No. of the Epoch')
plt.legend(['training', 'testing'], loc='upper left')
plt.show()

