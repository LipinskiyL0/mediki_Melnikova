# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:49:26 2020

@author: Леонид
"""

#import pandas as pd
#ind=0
import pandas as pd
Stat=[]
for m in ["LR", "SVC"]:
    method=m
    for i in range(3):
        exec(open("forGaClassModel_V3.py").read())
rez=pd.DataFrame(Stat, columns=["method",  "features","n_feature","cros", "er_test","er_train",
                                "cros0","cros1","cros2","cros3","cros4"])

rez.to_csv("rez_model_test.csv", index=False)
rez=pd.read_csv("rez_model_test.csv")
c=rez.groupby(["method"])["cros"].mean()
c1=rez.groupby(["method"])["cros"].std()
c2=rez.groupby(["method"])["n_feature"].mean()
c3=rez.groupby(["method"])["n_feature"].std()
c4=rez.groupby(["method"])["er_test"].mean()
c5=rez.groupby(["method"])["er_test"].std()
c6=rez.groupby(["method"])["er_train"].mean()
c7=rez.groupby(["method"])["er_train"].std()

c8=rez.groupby(["method"])["cros0"].mean()
c9=rez.groupby(["method"])["cros0"].std()
c10=rez.groupby(["method"])["cros1"].mean()
c11=rez.groupby(["method"])["cros1"].std()
c12=rez.groupby(["method"])["cros2"].mean()
c13=rez.groupby(["method"])["cros2"].std()
c14=rez.groupby(["method"])["cros3"].mean()
c15=rez.groupby(["method"])["cros3"].std()
c16=rez.groupby(["method"])["cros4"].mean()
c17=rez.groupby(["method"])["cros4"].std()

rez2=pd.DataFrame({"cros_mean":c, "cros_std":c1,
                   "n_feature_mean":c2,"n_feature_std":c3,
                   "er_test_mean":c4,"er_test_std":c5,
                   "er_train_mean":c6,"er_train_std":c7,
                   "cros0_mean":c8, "cros0_std":c9,
                   "cros1_mean":c10, "cros1_std":c11,
                   "cros2_mean":c12, "cros2_std":c13,
                   "cros3_mean":c14, "cros3_std":c15,
                   "cros4_mean":c16, "cros4_std":c17,
                   
                   
                   })
rez2=rez2.round(2)
rez2.to_csv("rez_model_test_stat.csv")
