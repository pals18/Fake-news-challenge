#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import string
import re
from nltk.corpus import stopwords
import scipy
from scipy.sparse import coo_matrix, hstack


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

Y=Y.replace({'unrelated':3})
Y=Y.replace({'agree':0})
Y=Y.replace({'disagree':1})
Y=Y.replace({'discuss':2})

### MODEL TO EVALUATE IF STANCES ARE RELATED OR UNRELATED

Y_train1=Y.iloc[:49972,]
Y_test1=Y.iloc[49972:,]

Y_train1=Y_train1.replace({1:0,2:0,3:1})
Y_test1=Y_test1.replace({1:0,2:0,3:1})

Y_train1=Y_train1.astype('category')
Y_test1=Y_test1.astype('category')

import xgboost as xgb
model_r=xgb.XGBClassifier(random_state=1)
model_r.fit(x_train, Y_train1)

model_r.score(x_test,Y_test1)

##### PREDICTING THE VALUES FOR RELATED OR UNRELATED

preds_r=model_r.predict_proba(x_test)

### MODEL TO EVALUATE IF STANCES ARE DISCUSS OR NOT DISCUSS

Y_train2=Y.iloc[:49972,]
Y_test2=Y.iloc[49972:,]
Y_train2=Y_train2.replace({1:0,2:1,3:0})
Y_test2=Y_test2.replace({1:0,2:1,3:0})
Y_train2=Y_train2.astype('category')
Y_test2=Y_test2.astype('category')

model_d=xgb.XGBClassifier(random_state=1)
model_d.fit(x_train, Y_train2)
model_d.score(x_test,Y_test2)

##### PREDICTING THE VALUES FOR DISCUSS OR NOT

preds_d=model_d.predict_proba(x_test)

### MODEL TO EVALUATE IF STANCES ARE AGREE OR DISAGREE

df=pd.read_csv("features//combinedf.csv")

df['Stance']=df['Stance'].replace({'unrelated':3})
df['Stance']=df['Stance'].replace({'agree':0})
df['Stance']=df['Stance'].replace({'disagree':1})
df['Stance']=df['Stance'].replace({'discuss':2})

df1=df[(df['Stance']==0)| (df['Stance']==1)]

train=df.iloc[:49972,]

X_train_3=df.iloc[:49972,]

X_test_3=df.iloc[49972:,]

X_train_3=X_train_3[(X_train_3['Stance']==0)| (X_train_3['Stance']==1)]

Y_train_3=X_train_3['Stance']
Y_test_3=X_test_3['Stance']

Y_train_3=Y_train_3.replace({0:1,1:0,2:0,3:0})
Y_test_3=Y_test_3.replace({0:1,1:0,2:0,3:0})
#Y_train_3.value_counts()

Y_train_3=Y_train_3.astype('category')
Y_test_3=Y_test_3.astype('category')

count_vector12      = CountVectorizer(ngram_range=(1,3),tokenizer=lambda doc: doc, lowercase=False)
count_x_train12     = count_vector12.fit_transform(X_train_3['combined'])
tfidf_transformer12 = TfidfTransformer()
tfidf_x_train12 = tfidf_transformer12.fit_transform(count_x_train12)


x_train_count_1 = count_vector12.transform(X_test_3['combined'])
tfidf_transformer_1=TfidfTransformer()
x_tfidf1=tfidf_transformer_1.fit_transform(x_train_count_1)

train=train[(train['Stance']==0)| (train['Stance']==1)].index.tolist()

new_x_train=pd.merge(X_train, X_train_3, left_index=True, right_index=True)
new_x_train=new_x_train.iloc[:,0:48]

g_train1=scipy.sparse.csr_matrix(new_x_train)
g_test1=scipy.sparse.csr_matrix(X_test)

x_train1=hstack([tfidf_x_train12,g_train1])
x_test1=hstack([x_tfidf1,g_test1])

x_train

model_d=xgb.XGBClassifier(random_state=1)
model_d.fit(x_train1, Y_train_3)
model_d.score(x_test1,Y_test_3)

preds_ab=model_d.predict_proba(x_test1)

preds_ab

prediction=pd.DataFrame(preds)
prediction=prediction.replace({3:'unrelated',0:'agree',2:'discuss',1:'disagree'})

prediction.columns=['Stance']

pre1=pd.DataFrame(preds_r)
pre2=pd.DataFrame(preds_d)
pre3=pd.DataFrame(preds_ab)
precom=pd.concat([pre1,pre2,pre3],axis=1)

precom.head()

precom.columns=['Related','Unrelated','No discuss','Discuss','Disagree','Agree']
precom=precom.drop(['Related','No discuss'],axis=1)

precom

data2.to_csv("testnew3.csv")

dt=pd.read_csv("competition_test_stances.csv")
ndf=pd.concat([dt,precom],axis=1)
ndf.to_csv("results_lat.csv",index=False)

pt=precom

pt

lp=[]
for b,d,e,f in pt.itertuples(index=False):
    if b>0.5:
        lp.append(3)#('unrelated')
    elif d>0.5:
        lp.append(2)#('discuss')
    elif e>f:
        lp.append(1)#('disagree')
    else:
        lp.append(0)#('agree')

lp=pd.DataFrame(lp)

lp.columns=['Stances']

dt=pd.read_csv("data//competition_test_stances_unlabeled.csv")
ndf=pd.concat([dt,lp],axis=1)
#ndf.to_csv("results_lat2.csv",index=False)

#print(ndf)

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

preds=lp
scorer(preds)

