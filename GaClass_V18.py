# -*- coding: utf-8 -*-
"""
Формируется класс в котором реализется генетический алгоритм:

*******************ПЕРЕМЕННЫЕ***********************************************
Gen_p - массив генотипов родителей. Формат: Gen_p[i, :] - генотип i-го родителя
        Gen_p[i, j] - j-й ген i-го родителя
Fenot - массив фенотипов получаем на основе Gen_p. Fenot[i, :] фенотип i-го родителя
        Fenot[i, j] - j-я координата i-го родителя
MinMaxE - массив матрицы минимумов максимумов и точностей по каждой координате
          формат: MinMaxE[i, :] - данные по i-й координате, MinMaxE[i, 0] - минимум
          i-ой координаты, MinMaxE[i, 1] - максимум i-й координаты, 
          MinMaxE[i, 2] - точность

n_bits - массив количества бит. формат n_bits[i] - количество бит кодирующее 
        i-ую переменную.
        
Fit - массив приспособленностmи родителей. Формат: Fit[i] - приспособленность 
      i-го родителя

Select - массив индексов родительских пар. Формат: Select[i, :] - i-ая родительская пара.
         Select[i, 0] - первый родитель в i-ой паре, Select[i, 1] - второй
         родитель в i-ой паре

n_tur - размер турнира для турнирной селекции
n_iter - количество итераций алгоритма
selection - вид селекции (турнирная="tur", пропорциональная="prop", 
            ранговая = "rang")
file_out - если True то выводим в файл, иначе не выводим
         
Gen_ch - массив генотипов потомков после скрещивания. Форма: Individ[i] - генотип 
          i-го потомка
          
best_fit - значение пригодности лучшего
best_gen - генотип лучшего
best_fen - фенотип лучшего
*******************ФУНКЦИИ**************************************************
inicialization - функция инициализирует основные переменные и создает стартовую 
                популяцию
                
fitness - функция пригодности генотипов родителей. (Функция приспособленности)

celection_prop - пропорциональная селекция

selection_turnir - турнирная селекция

selection_rang - ранговая селекция 

cross_odin - функия одноточечного скрещивание 

cross_dva - функия двуточечного скрещивание 

cross_ravn - функия равновероятного скрещивание 

mutation - функция мутации

elitizm - функция запоминания лучшего
"""
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import pickle
#import shelve


class GaClass:
    def __init__(self):
        
        self.Gen_p=[]
        self.Fenot=[]
        self.n_bits=[]
        self.Fit=[]
        self.Select=[]
        self.Gen_ch=[]
        self.i=0
        self.j=0
        
        #настройки алгоритма
        self.selection="tur" 
        self.n_iter=0
        self.n_tur=0
        self.p=-1
        self.MinMaxE=[]
        self.file_out=False
        self.cross="odin"
        
        #лучший индивид
        self.best_fit=-1
        self.best_gen=[]
        self.best_fen=[]
        
        
        
        
#***************************************************************************    
    def inicialization(self, opt):
        #стартовая инициализация. Инициализируем основные переменные и 
        #создаем стартовую популяцию
        MinMaxE=opt["MinMaxE"]
        n_ind=opt["n_ind"]
        n_iter=opt["n_iter"]
        n_tur=opt["n_tur"]
        #проверяем входные параметры
        if n_ind<=0:
            return False
        if n_iter<=0:
            return False
        MinMaxE=np.array(MinMaxE)
        
        #Если нашлась такая координата, где минимум больше максимума возвращаем
        #ошибку
        if np.sum(MinMaxE[:, 0]>=MinMaxE[:, 1])>0:
            return False
        #если нашлась координата с нулевой или отрицательной точностью возвращаем
        #ошибку
        if np.sum(MinMaxE[:, 2]<=0):
            return False
        
        self.MinMaxE=MinMaxE.copy()
         #определяем длину бинарной строки
        N=(MinMaxE[:, 1]-MinMaxE[:, 0])/MinMaxE[:, 2]
        self.n_bits=np.ceil((np.log2(N)))
        self.n_bits=np.array(self.n_bits, dtype=int)
        N=2**self.n_bits
        MinMaxE[:, 2]=(MinMaxE[:, 1]-MinMaxE[:, 0])/N
        self.MinMaxE=MinMaxE.copy()
        self.Gen_p=np.random.randint(2, size=[n_ind, np.sum(self.n_bits)])
        self.Fenot=np.zeros([n_ind, len(self.n_bits)])
        self.Fit=np.zeros(n_ind)
        self.Select=np.zeros([n_ind, 2], dtype=int)
        self.n_tur=n_tur
        self.n_iter=n_iter
        self.selection=opt["selection"]
        self.file_out=opt["file_out"]
        self.cross=opt["cross"]
        self.p=opt["mutation_p"]
        return
#***************************************************************************   
    def fenotip(self):
        # Переводим генотип в фенотип.   
        Gen=self.Gen_p.copy()
        
        for i in range(len(self.n_bits)):
            X, Gen1=np.hsplit(Gen, [self.n_bits[i]])
            Gen=Gen1
            #создаем строку по степеням двойки
            dva=np.zeros([self.n_bits[i]], dtype=int)
            #находим точки разбиения
            for j in range(self.n_bits[i]):
                dva[j]=2**(self.n_bits[i]-j-1)
            dva=dva[np.newaxis, :]
            rez=np.sum(dva*X, axis=1)
            self.Fenot[:, i]=rez*self.MinMaxE[i, 2]+self.MinMaxE[i, 0]
        return True
#***************************************************************************            
        #Оценка пригодности     
    def fitness(self):
        #Массив всех значений пригодности родителей
        self.Fit=1/(1+np.sum(self.Fenot**2, axis=1))
#***************************************************************************       
        #Пропрорциональная селекция
    def selection_prop(self):
        #P - массив значений вероятностей. P[i] - вероятность выбора i-го родителя
        if np.sum(self.Fit<0)>0:
            return False
        P=self.Fit/np.sum(self.Fit)
         
        
        for i in range(len(self.Fit)):
            for j in range(2):
                #колесо рулетки
                s=0               
                k=0
                E=np.random.random([len(self.Fit), 2])
                while s<E[i, j]:
                    s+=P[k]
                    k+=1
                    if k==len(self.Fit): #защита от некорректных вычислений
                        break
                k-=1
                self.Select[i, j]=k    
        return True
#***************************************************************************                           
        #Турнарная селекция.         
    def selection_turnir(self):
        if self.n_tur<=0:
            return False
        n_ind=len(self.Fit)
        
        for i in range(len(self.Fit)):
            for j in range(2):  
                #присваиваем индексу случайное значение от 0 до n_ind     
                index=np.random.randint(0, n_ind)
                maxх=self.Fit[index]
                for s in range (self.n_tur-1): #размер турнира без последнего
                    k=np.random.randint(0, n_ind) 
                    if self.Fit[k]> maxх: 
                    #если пригодность предыдущего меньше пригодности нового
                        index=k
                        maxх=self.Fit[k] #запоминаем нового
                self.Select[i, j]=index 
        return True
#***************************************************************************        
        #Ранговая селекция   
    def selection_rang(self):
        if np.sum(self.Fit<0)>0:
            return False
        
        n_ind=len(self.Fit)
        Rang=np.arange(1,n_ind+1, 1)
        Rang=np.array(Rang, dtype=int)
        index=np.argsort(self.Fit)
        Rang1=np.zeros(n_ind)
        Rang1[index]=Rang
      
        P=Rang1/np.sum(Rang1)
         
        for i in range(len(self.Fit)):            
            for j in range(2):
                #колесо рулетки
                s=0               
                k=0
                E=np.random.random([len(self.Fit), 2])
                while s<E[i, j]:
                    s+=P[k]
                    k+=1
                    if k==len(self.Fit): #защита от некорректных вычислений
                        break
                k-=1
                self.Select[i, j]=k    
        return True
#***************************************************************************        
        #Cрещивание. (рекомбинация)
    def cross_odin(self):
        #Одноточечное скрещивание 
        self.Gen_ch=[]
        i=np.random.randint(1, len(self.Gen_p[0,:])-1)
        self.i=[]
        for k in range(len(self.Select)):
            i=np.random.randint(1, len(self.Gen_p[0,:])-1)
            self.i.append(i)
            p1=self.Select[k, 0]
            p2=self.Select[k, 1]
    #        p1 - индекс первого родителя нулевой пары
    #        p2 - индекс второго родителя нулевой пары
            gen1=self.Gen_p[p1, :].copy()
            gen2=self.Gen_p[p2, :].copy()
    #        gen1 - генотип первого родителя
    #        gen2 - генотип второго родителя
            child1=np.hstack([gen1[:i], gen2[i:]])
            child2=np.hstack([gen2[:i], gen1[i:]])
    #        child1 - первый потомок
    #        child2 - второй потомок
            if np.random.random()<0.5:
                self.Gen_ch.append(child1)
            else:
                self.Gen_ch.append(child2)
        self.Gen_ch=np.array(self.Gen_ch)
        return True 
#***************************************************************************    
    def cross_dva(self):
        #Двуточечное скрещивание 
        self.Gen_ch=[]
        self.i=[]
        self.j=[]
        for k in range(len(self.Select)):
            i=np.random.randint(1, len(self.Gen_p[0,:])-1)
            j=np.random.randint(1, len(self.Gen_p[0,:])-1)
            if  i>j:
                musor=i
                i=j
                j=musor
            
            self.i.append(i)
            self.j.append(j)
            
            p1=self.Select[k, 0]
            p2=self.Select[k, 1]
        #        p1 - индекс первого родителя нулевой пары
        #        p2 - индекс второго родителя нулевой пары
            gen1=self.Gen_p[p1, :].copy()
            gen2=self.Gen_p[p2, :].copy()
        #        gen1 - генотип первого родителя
        #        gen2 - генотип второго родителя
            child1=np.hstack([gen1[:i], gen2[i:j], gen1[j:]])
            child2=np.hstack([gen2[:i], gen1[i:j], gen2[j:]])
            if np.random.random()<0.5:
                self.Gen_ch.append(child1)
            else:
                self.Gen_ch.append(child2)
        self.Gen_ch=np.array(self.Gen_ch)
        return True 
    
        return False
#***************************************************************************
    def cross_ravn(self): 
        #Равновероятное скрещивание
        self.Gen_ch=[]
        for k in range(len(self.Select)):
            p1=self.Select[k, 0]
            p2=self.Select[k, 1]
            #        p1 - индекс первого родителя нулевой пары
            #        p2 - индекс второго родителя нулевой пары
            gen1=self.Gen_p[p1, :].copy()
            gen2=self.Gen_p[p2, :].copy()
            p=np.random.random(len(self.Gen_p[0,:]))
            index=p<=0.5
            gen1[index]=gen2[index]
            self.Gen_ch.append(gen1)
        self.Gen_ch=np.array(self.Gen_ch)
        return True 
    
#**************************************************************************        
     #Мутации 
    def mutation(self):
        if self.p==-1:
            p=1/len(self.Gen_p[0,:])
        else:
            p=self.p
        m=np.random.random(self.Gen_p.shape) 
        self.Gen_ch[m<p]=1-self.Gen_ch[m<p]
        return True                 
#**************************************************************************     
    def elitizm(self):
        
        
        if ((self.best_gen==[])|(np.max(self.Fit)>self.best_fit)):
            #либо первая итерация, либо в ходе эволюции получилась популяция
            #в которой лучший, лучше чем best
            best_ind=np.argmax(self.Fit)
            self.best_fit=self.Fit[best_ind]
            self.best_gen=self.Gen_p[best_ind, :].copy()
            self.best_fen=self.Fenot[best_ind, :].copy()
        elif (np.max(self.Fit)<self.best_fit):
            #в ходе эволюции получилась популяция в которой лучший хуже
            #чем best. Записываем best в случайное место
            best_ind=np.random.randint(len(self.Fit))
            self.Fit[best_ind]=self.best_fit
            self.Gen_p[best_ind, :]=self.best_gen.copy()
            self.Fenot[best_ind, :]=self.best_fen.copy()            
        
               
        return True       
#**************************************************************************               
    
    def main(self):
        #главная функция, управляющая эволюционными операторами
        #считаем что self.inicialization уде отработало и генотип
        #фенотип и пригодность уже построены
        
        self.fenotip()
        self.fitness()
        self.elitizm()
        
        if self.file_out==True:
            f=open(r"Evolution.txt", 'w')
            f.write('Эволюция генетического алгоритма\n')
            f.write('Исходные генотипы родителей\n')
            for i in range(len(self.Gen_p[:,0])):
                f.write(str(i)+"\t")
                for j in range(len(self.Gen_p[0,:])):
                    f.write(str(self.Gen_p[i,j]))
                f.write("\n")
            f.write('Фенотипы родителей исходной популяции\n')
            for i in range(len(self.Fenot[:,0])):
                f.write(str(i)+"\t")
                for j in range(len(self.Fenot[0,:])):
                    f.write(str(self.Fenot[i,j])+"\t")
                f.write("\n")
            f.write('Значение функции пригодности\n')
            for i in range(len(self.Fit)):
                f.write(str(i)+"\t")
                f.write(str(self.Fit[i]))
                f.write("\n")
            f.write('Генотип лучшего\n')
            f.write(str(self.best_gen)+"\n")
            f.write('Фенотип лучшего\n')
            f.write(str(self.best_fen)+"\n")
            f.write('Пригодность лучшего\n')
            f.write(str(self.best_fit)+"\n")
            
        best=[]
        worse=[]
        aver=[]
        best.append(np.max(self.Fit))
        worse.append(np.min(self.Fit))
        aver.append(np.mean(self.Fit))       

        for it in range(self.n_iter):
            #селекция
            if self.selection=="tur":
                self.selection_turnir()
                if self.file_out==True:    
                    f.write('Турнирная селекция\n')
            elif self.selection=="prop":
                self.selection_prop()
                if self.file_out==True:    
                    f.write('Пропорциональная селекция\n')
            elif self.selection=="rang":
                self.selection_rang()
                if self.file_out==True:    
                    f.write('Ранговая селекция\n')
            
            if self.file_out==True:    
                f.write('Результат селекции - индексы родительских пар\n')
                for i in range(len(self.Select[:,0])):
                    f.write(str(i)+"\t")
                    for j in range(len(self.Select[0,:])):
                        f.write(str(self.Select[i,j])+"\t")
                    f.write("\n")
            #рекомбинация 
            if self.cross=="odin":
                self.cross_odin()
                if self.file_out==True:    
                    f.write('Одноточечное скрещивание\n')
                    for i in range(len(self.Gen_p[:,0])):
                        f.write("Пара "+str(i)+"\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_p[self.Select[i, 0],j]))
                        f.write("\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_p[self.Select[i, 1],j]))
                        f.write("\n")
                        f.write("Точка скрещивания: "+str(self.i[i]))
                        f.write("\nПотомок\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_ch[i,j]))
                        f.write("\n\n")   
                            
                    
            elif self.cross=="dva":
                self.cross_dva()
                if self.file_out==True:    
                    f.write('Двухточечное скрещивание\n')
                    for i in range(len(self.Gen_p[:,0])):
                        f.write("Пара "+str(i)+"\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_p[self.Select[i, 0],j]))
                        f.write("\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_p[self.Select[i, 1],j]))
                        f.write("\n")
                        f.write("Точки скрещивания: "+str(self.i[i])+" "+str(self.j[i]))
                        f.write("\nПотомок\n")
                        for j in range(len(self.Gen_p[0,:])):
                            f.write(str(self.Gen_ch[i,j]))
                        f.write("\n\n")             
            elif self.cross=="ravn":
                self.cross_ravn()
                if self.file_out==True:    
                    f.write('Равновероятное скрещивание\n')
            
            if self.file_out==True:    
                f.write('Результат срещевания - Генотипы потомков\n')
                for i in range(len(self.Gen_ch[:,0])):
                    f.write(str(i)+"\t")
                    for j in range(len(self.Gen_ch[0,:])):
                        f.write(str(self.Gen_ch[i,j]))
                    f.write("\n")
            
            self.mutation()
            if self.file_out==True:    
                f.write('Результат мутации\n')
                for i in range(len(self.Gen_ch[:,0])):
                    f.write(str(i)+"\t")
                    for j in range(len(self.Gen_ch[0,:])):
                        f.write(str(self.Gen_ch[i,j]))
                    f.write("\n")
            
            self.Gen_p=self.Gen_ch.copy()
            
            self.fenotip()
            if self.file_out==True:
                f.write('Фенотипы потомков=родителей новой популяции\n')
                for i in range(len(self.Fenot[:,0])):
                    f.write(str(i)+"\t")
                    for j in range(len(self.Fenot[0,:])):
                        f.write(str(self.Fenot[i,j])+"\t")
                    f.write("\n")
                
            self.fitness()
            if self.file_out==True:
                f.write('Значение функции пригодности\n')
                for i in range(len(self.Fit)):
                    f.write(str(i)+"\t")
                    f.write(str(self.Fit[i]))
                    f.write("\n")
            
            self.elitizm()
            if self.file_out==True:
                f.write('Генотип лучшего\n')
                f.write(str(self.best_gen)+"\n")
                f.write('Фенотип лучшего\n')
                f.write(str(self.best_fen)+"\n")
                f.write('Пригодность лучшего\n')
                f.write(str(self.best_fit)+"\n")
                
            print("лучшее решение: {0},\t среднее решение: {1},\t худшее: {2}".format(
                    np.max(self.Fit), np.mean(self.Fit), np.min(self.Fit)))
            best.append(np.max(self.Fit))
            worse.append(np.min(self.Fit))
            aver.append(np.mean(self.Fit))

        plt.close("all")
        plt.figure()
        plt.plot(best, "r")
        plt.plot(aver, "b")
        plt.plot(worse, "g")
        plt.legend(["лучший", "средний", "худший"])
        plt.title("Работа ГА")
        
        if self.file_out==True:
            f.close()   
            
        return   
               
             

        
        
        
        
















        
        
        
        
        
        
        
        
        
        
        
        
        