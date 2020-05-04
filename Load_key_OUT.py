# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:33:57 2020

@author: Myself
"""

import pandas as pd
import numpy as np


df=pd.read_csv("outs.csv", encoding="windows-1251") 

df=np.array(df, dtype=int)
df.sum(axis=0) 

#ФР - резистентность
fr=df[:,0]
names = ['ФР'] 
fr = pd.DataFrame(fr, columns=names)
fr.to_csv("Out_FR.csv", header=True,  index=False)
fr.sum(axis=0) 

#констипация
constip=df[:,1]
names = ['констипация'] 
constip = pd.DataFrame(constip,  columns=names)
constip.to_csv("Out_constip.csv", header=True,  index=False)

#все остальные побочные
c=df[:, 2:]
mas=np.zeros([len(c)])
for i in range(len(c)):
        if sum(c[i,:])>0 :
           mas[i]=1
           
names = ['Все остальные побочные']      
mas=pd.DataFrame(mas, columns=names)
mas.to_csv("Out_other.csv", header=True,  index=False)
 
