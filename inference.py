import sys
import os
import pickle
import scipy.sparse
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def warn(*args, **kwargs):
    pass
warnings.warn = warn

f=sys.argv[1]


X_test=pd.read_csv("features//X_test.csv")
x_test=scipy.sparse.load_npz("features//x_test.npz")
X_testin=pickle.load(open("features//X_testin.pkl",'rb'))
df=pd.read_csv("data//competition_test_stances.csv")
Stance=df['Stance']
Stance1=Stance.replace({'unrelated':3})
Stance1=Stance1.replace({'agree':0})
Stance1=Stance1.replace({'disagree':1})
Stance1=Stance1.replace({'discuss':2})
pred1=Stance1.values.ravel()


def modelevaluate(preds):
    sco=[]
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
    print('\n------Model Evaluation------\n')
    print("Accuracy:(Actual vs predicted",accuracy_score(preds,pred1))
    print("Score:   ",sum(sco))
    print('\n------Confusion Matrix------\n')
    print(confusion_matrix(preds, pred1))
    print('\n----------------------Classification Report------------\n')
    print(classification_report(preds, pred1))



if f=='neural':
    model = keras.models.load_model("models//"+f)
    preds=model.predict_classes(X_testin)
    modelevaluate(preds)
elif f=='gradient':
    model=pickle.load(open("models//"+f+".pkl",'rb'))
    preds=model.predict(x_test)
    modelevaluate(preds)
elif f=='xgboost':
    model=pickle.load(open("models//"+f+".pkl",'rb'))
    preds=model.predict(x_test)
    modelevaluate(preds)
elif f=='randomforest':
    model=pickle.load(open("models//"+f+".pkl",'rb'))
    preds=model.predict(X_test)
    modelevaluate(preds)


