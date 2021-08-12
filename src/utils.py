import numpy as np
import pdb
import math
import pickle
import time 
import json
import os
import random

def pakage_result(result,blank_json,result_json,y_scale=True):
    with open(blank_json,"r") as f_r:
        arch_dict = json.load(f_r)
    i=0
    for sub_dict in arch_dict.items():
        if y_scale == True:
            sub_dict[1]['acc'] = float(result[i])/100.0
        else:
            sub_dict[1]['acc'] = float(result[i])
        i+=1
    with open(result_json,"w") as f_w:
        json.dump(arch_dict,f_w)


def train_model(model,train_x,train_y,model_path):
    model.fit(train_x,train_y)
    with open(model_path,"wb") as f_w:
        pickle.dump(model,f_w)

def predict(model_path,test_x):
    with open(model_path,"rb") as f_r:
        model = pickle.load(f_r)
    result = model.predict(test_x)
    return result