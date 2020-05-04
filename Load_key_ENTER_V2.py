# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:21:09 2020

@author: Леонид
"""

import pandas as pd
import numpy as np


df=pd.read_csv("data_in1.csv", encoding="windows-1251") 

key=pd.read_csv("key.csv", encoding="windows-1251",delimiter=";", index_col="index") 
key=key["Name"]

col=list(key.index)
col_df=[]
for c in col:
    if key[c]==1:
        #столбец качественный
        val_uniq=df[c].unique()
        val_uniq.sort()
        
        if len(val_uniq)==2:
            
            mas=np.zeros([len(df[c]), 1], dtype=int)
            names=[c+"_bin"]
            mas=pd.DataFrame(mas, columns=names)
            
            for i in range(len(df[c])):
                cc=df[c].iloc[i]
                if cc==val_uniq[1]:
                    mas[names[0]].iloc[i]=1
            
                
        else:
            mas=np.zeros([len(df[c]), len(val_uniq)], dtype=int)
            names=[]
            for cc in val_uniq:
                names.append(c+"_"+str(cc))
            mas=pd.DataFrame(mas, columns=names)
            
            for i in range(len(df[c])):
                cc=df[c].iloc[i]
                mas[c+"_"+str(cc)].iloc[i]=1
            
        del df[c]
        df=pd.concat([df, mas], axis=1)
        col_df=col_df+names
    else:
        col_df.append(c)

df=df[col_df]
df.to_csv("enter.csv",  index=False)
        
#    print("a[{0}]={1}".format(c, a[c]))
