# -*- coding: utf-8 -*-
"""
Проверяем данные по тесту Мана-Уитни на однородность
"""
# Mann-Whitney U test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
name_files="FR_rez_model_test.csv"
df=pd.read_csv(name_files)

df=df.dropna()
models=list(set(df["method"]))

stlb=['cros0', 'cros1', 'cros2', 'cros3', 'cros4']

Rez=[]
for model in models:
    try:
        rez=[]
        df1=df[df["method"]==model]
        data1=df1["er_test"].values
        data2=df1[stlb].values
        data2=np.reshape(data2, [len(data1)*5, 1] )
    
        stat, p = mannwhitneyu(data1, data2[:, 0])
        rez.append(model)
        rez.append(stat)
        rez.append(p)
        
        # interpret
        alpha = 0.05
        if p > alpha:
        	rez.append('Same distribution (fail to reject H0)')
        else:
        	rez.append('Different distribution (reject H0)')
    except:   
        rez=[model, "error"]
    Rez.append(rez)
Rez=pd.DataFrame(Rez, columns=["method", "stat", "p","interpret" ])
Rez.to_csv(name_files[:-4]+"_ManaWhitneyu.csv")
