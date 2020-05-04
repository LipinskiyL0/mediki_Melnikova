# -*- coding: utf-8 -*-
"""

21/5000
model design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC 
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.tree import DecisionTreeClassifier as DTC 
from sklearn.neural_network import MLPClassifier 


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
from sklearn.externals import joblib
from sklearn.metrics import f1_score

def model(X_train, y_train, X_test=[], y_test=[], method="LR"):
     #X_train входы модели для обучения
     #X_test входы модели для тестирования
     #y_train -выходы модели для обучения
     #y_test - выводы модели для тестирования
     #method - модели машинного обучения
     
    if method=="LR":
        lr=LR()
    elif method=="KNC":
        lr=KNC()    
    elif method=="RFC":
        lr=RFC()
    elif method=="GBC":
        lr=GBC()    
    elif method=="DTC":
        lr=DTC() 
    elif method=="MLPClassifier":
        lr=MLPClassifier()
    elif method=="LinearSVC":
        lr=LinearSVC()
    elif method=="SVC":
        lr=SVC()
    else:
        print("unknown method")
        return False

    
    if ( (type(X_test)!=list) & (type(y_test)!=list)):
        lr=lr.fit(X_train, y_train.iloc[:,0])
        y_mod_train=lr.predict(X_train)
        y_mod_test=lr.predict(X_test)
        #average - параметр для рассчёта f-меры (micro, macro, weighted, samples)
        f1_train=f1_score(y_train, y_mod_train, average='macro')
        f1_test=f1_score(y_test, y_mod_test, average='macro')
        out={"model":lr, "f1_train":f1_train, "f1_test":f1_test,
                "y_mod_train":y_mod_train, "y_mod_test":y_mod_test }
        scores_train = cross_val_score(lr, X_train, y_train.iloc[:,0], cv=5,  scoring='f1_macro')
        for i in range(len(scores_train)):
            out["cros"+str(i)]=scores_train[i]
        return out
    
    else:
        #считаем кросс-валидацию
        scores_train = cross_val_score(lr, X_train, y_train.iloc[:,0], cv=5,  scoring='f1_macro')
        return np.mean(scores_train)
    return False



    

    
    
    
    