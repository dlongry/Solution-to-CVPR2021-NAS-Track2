import os 
import json
import numpy as np
from utils import *

from sklearn import ensemble, linear_model, neighbors, \
    decomposition, manifold, neural_network, svm
from sklearn.decomposition import PCA

def read_data_whiteBox(filename,norm=False,stage1_path=None):
    Y_all = []
    X_all = []
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc'])
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])  
    
    if stage1_path!=None:
        with open(stage1_path,"r") as f:
            arch_dict = json.load(f)    
            for sub_dict in arch_dict.items():
                Y_all.append(sub_dict[1]['acc'])
                X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    if norm == True:
        kernel_ratio = [[3/7,3/6],[3/7,6/6],[5/7,3/6],[5/7,6/6],[7/7,3/6],[7/7,6/6]]
    else:
        kernel_ratio = [[3,3],[3,6],[5,3],[5,6],[7,3],[7,6]] 

    X_characs = []
    for i in range(len(X_all)):
        X_charac=[]
        for x in X_all[i]:
            X_charac+=kernel_ratio[x-1]
        X_characs.append(X_charac.copy())        
    X_characs,Y_all = np.array(X_characs), np.array(Y_all)
    return X_characs,Y_all

'''
in_param：
    test_path:         the path of testset
    map_path :         a dict:{second-level model's name : first-level models used} 
    level1_data_path:  the path of a json file witch saves outputs of first-level models
    cancat:            whether concat the reduction dimension features with PCA
    pca_reduce：       the dimension of PCA features
    sec_root:          the root folder path of second-level models
return：
    ave_result：       the final prediction result
'''
def multi_bagging_test(test_path, map_path, level1_data_path,
                            concat=False,pca_reduce=None,
                            pca_x_pro_path=None,sec_root=None):

    if pca_reduce != None:
        pca_characs,_ =  read_data_whiteBox(test_path,norm=True)
        pca = PCA(n_components=pca_reduce,random_state=0)  
        new_pca_characs = pca.fit_transform(pca_characs)
          
    #load outputs of first_level models:
    with open(level1_data_path,"r") as f_r:  
        level1_data = json.load(f_r)   
        level1_data = np.array(level1_data)
                
    
    with open(map_path,"r") as f_r:  
        level_map = json.load(f_r)
        
    results = []  
    i=0
    for level2_model_path,sam_index in level_map.items():
        print(f"compute {i}-th model's result...")
        level2_model_path = os.path.join(sec_root,level2_model_path.split("\\")[-1])
        X_data = level1_data[:,sam_index]
   
        
        if concat==True:
            if pca_reduce == None:
                X_characs,_ =  read_data(test_path,norm=True)
                X_data = np.concatenate((X_data,X_characs),axis=1)
            else:
                X_data = np.concatenate((X_data,new_pca_characs),axis=1)
        
        y_pre = predict(level2_model_path,X_data)
        results.append(y_pre)
        i+=1
    ave_result = sum(results)/len(results)
    return ave_result

if __name__ == "__main__":
    test_path = "../data/Track2_stage2_test.json" 
    sec_root = "../models/sec_level"       
    map_path   = os.path.join(sec_root,"level_map.json")   
    level1_data_path = os.path.join(sec_root,"level1_out.json") 
    result_path   = "../result/cascade_bagging_result.json" 
    pca_reduce=3   
    concat  = True  
    ave_result_fast=multi_bagging_test( test_path=test_path, map_path=map_path, 
                                        level1_data_path=level1_data_path,
                                        concat=concat,pca_reduce=pca_reduce,
                                        sec_root=sec_root)
                                        
    pakage_result(ave_result_fast,test_path,result_path,y_scale=False)
