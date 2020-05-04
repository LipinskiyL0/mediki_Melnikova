# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:21:09 2020

@author: Леонид
"""

import pandas as pd


df=pd.read_csv("data_in.csv", encoding="windows-1251") 

key=pd.read_csv("key.csv", encoding="windows-1251",delimiter=";", index_col="index") 
key=key["Name"]

col=list(key.index)

for c in col:
    if key[c]==1:
        #столбец качественный
        val_uniq=df[c].unique()
        for cc in val_uniq:
            print(c+"_"+cc)
        
#    print("a[{0}]={1}".format(c, a[c]))
