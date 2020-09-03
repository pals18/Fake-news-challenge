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

# XGBoost Classifier

### HYPERPARAMETER TUNING FOR XGBOOST
learning_rate=0.4
min_child_weight=4
max_dept=3
gamma=0
n_estimators=100

X_trainv=X_train.iloc[:44975,:]
Y_trainv=Y_train.iloc[:44975]
X_val=X_train.iloc[44975:,:]
Y_val=Y_train.iloc[44975:]

### VALIDATION FOR PARAMETER TUNING


import xgboost as xgb
model=xgb.XGBClassifier(random_state=100,eta=learning_rate,min_child_weight=min_child_weight,max_depth=max_dept,gamma=gamma,n_estimators=n_estimators)
model.fit(X_trainv, Y_trainv)
print(model.score(X_val,Y_val))


### Model without tfidf transformer

import xgboost as xgb
model=xgb.XGBClassifier(random_state=100,eta=learning_rate,min_child_weight=min_child_weight,max_depth=max_dept,gamma=gamma,n_estimators=n_estimators)
model.fit(X_train, Y_train)
print(model.score(X_test,Y_test))
pred_xg=model.predict(X_test)
res_xg,acc_xg=scorer(pred_xg)
print(res_xg)



### Model with tfidf transformer

import xgboost as xgb
model2=xgb.XGBClassifier(random_state=100,eta=0.4,min_child_weight=4,n_estimators=1500)#,max_depth=6,gamma=3)#,n_estimators=1000)
model2.fit(x_train, Y_train)
print(model2.score(X_test,Y_test))
pred_xg2=model2.predict(X_test)
res_xg2,acc_xg2=scorer(pred_xg2)
print(res_xg2)

### Exporting the model as pickle file

import pickle
with open("models//xgboost.pkl",'wb') as pk:
    pickle.dump(model2,pk)

Y_val.shape

import xgboost as xgb
eval_s = [(X_trainv, Y_trainv), (X_val, Y_val)]
model2=xgb.XGBClassifier(random_state=100,eta=0.4,min_child_weight=4)#,n_estimators=1500)#,max_depth=6,gamma=3)#,n_estimators=1000)
model2.fit(X_trainv, Y_trainv,eval_set=eval_s)

def Snippet_189():
    print()
    print(format('Hoe to visualise XGBoost model with learning curves','*^82'))
    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from matplotlib import pyplot
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    

    # make predictions for x_test data
    y_pred = model2.predict(X_val)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(Y_val, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # retrieve performance metrics
    results = model2.evals_result()
    print(results)
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)
    # plot log loss

    # plot classification error
    fig, ax = pyplot.subplots(figsize=(6,6))
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    #ax.set_facecolor('white')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.grid(color='white', linestyle='solid')
    pyplot.show()

Snippet_189()

predicted=pd.DataFrame(pred_xg2)

predicted=predicted.replace({3:'unrelated',2:'discuss',1:'disagree',0:'agree'})

predicted.columns=['Stance']

df=pd.read_csv("data//competition_test_stances_unlabeled.csv")
result = pd.DataFrame(predicted)
ndf=pd.concat([df,result],axis=1)
ndf.to_csv('results//answer.csv', index=False, encoding='utf-8')


### Confusion Matrix

from sklearn.metrics import confusion_matrix
print(confusion_matrix(pred_xg2, pred1))

### Classification Report

from sklearn.metrics import classification_report
print(classification_report(preds,pred1))

# GRADIENT BOOSTING CLASSIFIER

### Model with tfidf transformer

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=0)#,learning_rate=0.2,min_samples_split=8)
clf.fit(x_train, Y_train)
print(model.score(x_test,Y_test))
pred_gd=model.predict(x_test)
res_gd,acc_xg=scorer(pred_gd)
print(res_xg2)

### Exporting the model as pickle file

import pickle
with open("models//gradient.pkl",'wb') as pk:
    pickle.dump(clf,pk)

### Model without tfidf transformer

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=0)#,learning_rate=0.2,min_samples_split=8)
clf.fit(X_train, Y_train)
print(model.score(X_test,Y_test))
pred_gd2=model.predict(X_test)
res_gd2,acc_gd2=scorer(pred_gd2)
print(res_gd2)

### Confusion Matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(pred_gd, pred1)

### Classification Report

from sklearn.metrics import classification_report
print(classification_report(pred_gd,pred1))

# DECISION TREE CLASSIFIER

### Model without tfidf transformer

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
print(model.score(X_test,Y_test))
pred_dt=model.predict(X_test)
res_dt,acc_dt=scorer(pred_dt)
print(res_dt)

### Model with tfidf transformer

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, Y_train)
print(model.score(x_test,Y_test))
pred_dt2=model.predict(x_test)
res_dt2,acc_dt2=scorer(pred_dt2)
print(res_dt2)

### Exporting the model as pickle file

import pickle
with open("models//decisiontree.pkl",'wb') as pk:
    pickle.dump(clf,pk)

### Confusion Matrix

from sklearn.metrics import confusion_matrix
print(confusion_matrix(pred_dt2, pred1))

### Classification Report

from sklearn.metrics import classification_report
print(classification_report(pred_dt2,pred1))

# RANDOM FOREST CLASSIFIER

### Model with tfidf transformer

from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(random_state=0,n_estimators=1000,max_depth=10)
clf1.fit(x_train, Y_train)
print(clf1.score(x_test,Y_test))
pred_rf=clf1.predict(x_test)
res_rf,acc_rf=scorer(pred_rf)
print(res_rf)

### Model without tfidf transformer

from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(random_state=0,n_estimators=1000,max_depth=10)
clf1.fit(X_train, Y_train)
print(clf1.score(X_test,Y_test))
pred_rf2=clf1.predict(X_test)
res_rf2,acc_dt2=scorer(pred_rf2)
print(res_rf2)

### Exporting the model as pickle file

import pickle
with open("models//randomforest.pkl",'wb') as pk:
    pickle.dump(clf1,pk)

### Confusion Matrix

from sklearn.metrics import confusion_matrix
print(confusion_matrix(pred_rf2, pred1))

### Classification Report

from sklearn.metrics import classification_report
print(classification_report(pred_rf2,pred1))

### ROC CURVE FOR XGBOOST
y_score = model.predict_proba(x_test)

l_prob=pd.DataFrame(y_score)

import numpy as np
from keras.utils import to_categorical
y_test=to_categorical(np.asarray(pred1))

from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_classes = 4
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], label='ROC curve (area = %0.2f)' % roc_auc[3])
plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

