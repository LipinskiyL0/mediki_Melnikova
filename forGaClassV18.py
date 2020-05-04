# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:40:38 2020

@author: Леонид
"""
#import datetime

from GaClass_V17 import GaClass

ga=GaClass()

MinMaxE=[[-1, 1, 0.1], [-10, 20, 0.1], [-100, 200, 0.1] ]
opt={"MinMaxE":MinMaxE, "n_ind":70, "n_iter":40, "n_tur":5, "selection":"tur",
     "file_out":True, "cross":"odin", "mutation_p":-1}
#t1=datetime.datetime.now()
ga.inicialization(opt)
ga.main()
#t2=datetime.datetime.now()
#print("Время работы алгоритма: {0}".format(t2-t1))
#ga.fenotip()
#ga.fitness()
#ga.selection_prop()
#ga.selection_turnir()
#ga.selection_rang()
#ga.cross_odin()
#ga.cross_dva()
#ga.cross_ravn()
#ga.mutation()
#ga.sort()
#ga.elitizm()

