# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:42:19 2020

@author: Myself

opt - подключаемые параметры: n_ind - кол-во индивидов, n_iter - кол-во интераций, 
      selection - выбор селекции (турнирная="tur", пропорциональная="prop", 
      ранговая = "rang"), cross - выбор скрещивание 
      (Одноточечное="odin", Двухточечное="dva", Равновероятное="ravn"), 
      mutation_p - вероятность для мутации (-1 значение для стандартной вероятности мутации), 
      file_out- сохранение в файл,method -  выбор модели машинного обучения
      (LR, KNC, RFC, GBC, DTC, MLPClassifier, LinearSVC, SVC)
      X_test - значение тестовой выборки на входе, y_test- значение тестовой выборки на выходе)  
      
df - таблица входов. строки - пациенты, столбцы - призкаки (входы) 

out - балица выходов. строки - пациенты, столбец - значеные признака (выход) 

X - матрица входов модели для обучения
              
y - матрица выходов модели для обучения     
"""

import datetime
import pandas as pd
import numpy as np
from GaClassModel_V1 import GaClassGenotip
from ModelEnd2 import model
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

plt.close("all")
#Stat=[]
#df_fi=[]
#method="SVC"
df=pd.read_csv("enter1.csv")

out=pd.read_csv("Out_FR.csv")

ga=GaClassGenotip()

opt={ "n_ind":10, "n_iter":5, "n_tur":3, "selection":"tur",
     "file_out":False, "cross":"ravn", "mutation_p":-1, "method":method, 
     "X":df, "y":out}

t1=datetime.datetime.now()
ga.inicialization(opt)
ga.main()
df1=df.iloc[:, ga.best_fen]
cros=model(X_train=df1, y_train=out,  method=method)


print("кросс-валидация=", cros) 
print("Количество признаков ", np.sum(ga.best_fen))       
X_train, X_test, y_train, y_test = train_test_split(
                df1, out, test_size=0.33, random_state=42, stratify=out)

out_model=model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                method=method)
y_pred_test=out_model["y_mod_test"]
y_pred_train=out_model["y_mod_train"]

#оценка значимости признаков
results = permutation_importance(out_model["model"], X_train, y_train,
                                 scoring='f1_macro', n_repeats=10)


df_fi1=pd.DataFrame( results.importances_mean, columns=["important"])
df_fi1["features"]=list(X_train.columns)

df_fi1["method"]=method
df_fi1=df_fi1[["method", "features", "important"]]
df_fi.append(df_fi1)

#print("точность по тестовой выборке=", out_model["f1_test"]) 
#print("точность по обучающей выборке=", out_model["f1_train"]) 
#plt.figure()
#plt.plot(y_pred_test, "r*", label="Модель")
#plt.plot(y_test.values, "b*", label="Выборка")
#plt.title("Test")
#plt.legend()
#
#plt.figure()
#plt.plot(y_pred_train, "r*", label="Модель")
#plt.plot(y_train.values, "b*", label="Выборка")
#plt.title("Train")
#plt.legend()
#
#
t2=datetime.datetime.now()
print("Время работы алгоритма: {0}".format(t2-t1))

stat=[method, list(df1.columns), np.sum(ga.best_fen),cros, out_model["f1_test"], out_model["f1_train"]]
for i in range(5):
    stat.append(out_model["cros"+str(i)])
    
    
Stat.append(stat)