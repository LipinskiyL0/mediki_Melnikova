# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:39:27 2020

@author: Myself

opt - подключаемые параметры из forGaClassModel_V1.

df - таблица входов. строки - пациенты, столбцы - призкаки (входы) 

out - талица выходов. строки - пациенты, столбец - значеные признака (выход)

*******************************************************************************
n_bits - длина бинарной строки - число признаков на входе.

Gen_p - матрица стартовой популяции. Строки - число индивидов, 
        солбцы - длина бинарной строки.

Fenot - матрица фенотипов по булевой логики True/false
       
X - матрица входов модели для обучения
              
y - матрица выходов модели для обучения
         
                 
"""

from GaClass_V18 import GaClass
from ModelEnd2 import model 

import numpy as np
#import pandas as pd

class GaClassGenotip(GaClass):
    
    def __init__(self):
        
        GaClass.__init__(self)
        #Переменные кода
        self.X=[]
        self.y=[]
 
        #настройки алгоритма
        self.method="LR" 

#***************************************************************************    
    def inicialization(self, opt):
        #стартовая инициализация. Инициализируем основные переменные и 
        #создаем стартовую популяцию
        n_ind=opt["n_ind"]
        n_iter=opt["n_iter"]
        n_tur=opt["n_tur"]

        #проверяем входные параметры
        if n_ind<=0:
            return False
        if n_iter<=0:
            return False
        #Сохраняем входы и выходы 
        self.y=opt["y"].copy()
        self.X=opt["X"].copy()   
        
        #определяем длину бинарной строк  (число признаков)  
        col=list(opt["X"].columns)
        self.n_bits=len(col)

        #генерируем стартовую популяцию (создаем нулевые вектора и мутируем их)
        #кол-во строк - число пациентов. кол-во столбцов - число признаков.
        self.Gen_p=np.random.randint(2, size=[n_ind, self.n_bits])

        #матрица фенотипов по булевой логики True/false
        self.Fenot=np.array(self.Gen_p, dtype=bool)
        
        self.Fit=np.zeros(n_ind)
        self.Select=np.zeros([n_ind, 2], dtype=int)
        self.n_tur=n_tur
        self.n_iter=n_iter
        self.selection=opt["selection"]
        self.file_out=opt["file_out"]
        self.cross=opt["cross"]
        self.p=opt["mutation_p"]
        
        self.method=opt["method"]
        return
#***************************************************************************   
    def fenotip(self):
        self.Fenot=np.array(self.Gen_p, dtype=bool)
        return True
#***************************************************************************            
        #Оценка пригодности     
    def fitness(self):
        #Массив всех значений пригодности родителей
        for i in range(len(self.Fenot[:, 0])):
            X=self.X.iloc[:, self.Fenot[i, :]]
            y=self.y
            X_1=np.array(X)
            if sum(X_1[0, :])==0:
                self.Fit[i]=0
            else:
            #пригодность минус штраф за количестово признаков
                self.Fit[i]=model(X_train=X, y_train=y,  method=self.method)-0.001*np.sum(self.Fenot[i, :])

        return True        
   
#ga=GaClassGenotip()     
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    