# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:17:35 2020

@author: Леонид
"""

import pandas as pd

def ComaToPoint(df):
    
    col=list(df.columns)
    for c in col:
        df[c] = df[c].astype(str)
        df[c] = [x.replace(',', '.') for x in df[c]]
        df[c] = df[c].astype(float)

    return df

df=pd.read_csv("enter.csv")

df=ComaToPoint(df)
df.to_csv("enter1.csv",  index=False)