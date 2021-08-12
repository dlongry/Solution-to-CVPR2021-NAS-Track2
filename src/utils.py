import numpy as np
import pdb
import math
import pickle
import time 
import json
import os
import random

def prepare_data(file_name):
    with open(file_name,"r") as f:
        arch_dict= json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2]
    X_all,Y_all = np.array(X_all), np.array(Y_all)
    X_train,Y_train,X_test,Y_test= X_all[0:150],Y_all[0:150],X_all[150:],Y_all[150:]
    return X_train,Y_train,X_test,Y_test

def read_data_blackBox(filename):
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    X_all,Y_all = np.array(X_all), np.array(Y_all)
    return X_all,Y_all

def read_data_one_hot(filename):
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    X_one_hots = []
    fix_len=7
    for i in range(len(X_all)):
        X_one_hot=[]
        for x in X_all[i]:
            x_one_hot = [0]*fix_len
            x_one_hot[x] = 1
            X_one_hot+=x_one_hot
        X_one_hots.append(X_one_hot.copy())
    X_one_hots,Y_all = np.array(X_one_hots), np.array(Y_all)
    return X_one_hots,Y_all


def input_gaussian_noise(x_data,y_data,seed=0,SNR=20):
    #fix random package
    random.seed(seed)
    #fix numpy package
    np.random.seed(seed)  
    noise = np.random.randn(x_data.shape[0],x_data.shape[1]) 
    noise = noise-np.mean(noise) 
    signal_power = np.linalg.norm(x_data - x_data.mean())**2 / x_data.size	
    noise_variance = signal_power/np.power(10,(SNR/10))         
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    
    noise_x = noise + x_data
    noise_y = y_data.copy()
    return noise_x,noise_y

def read_data_whiteBox(filename,norm=False,y_scale=True,y_pro=False):
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    if norm == True:
        kernel_ratio = [[3/7,3/6],[3/7,6/6],[5/7,3/6],[5/7,6/6],[7/7,3/6],[7/7,6/6]]
    elif norm == "max":
        kernel_ratio = [[3/7,3/7],[3/7,6/7],[5/7,3/7],[5/7,6/7],[7/7,3/7],[7/7,6/7]]
    else:
        kernel_ratio=[[3,3],[3,6],[5,3],[5,6],[7,3],[7,6]] 

    for sub_dict in arch_dict.items():
        if y_scale == True:
            y_data = sub_dict[1]['acc']*100
        else:
            y_data = sub_dict[1]['acc']
        
        if y_pro == True:
            assert y_data<=1
            y_data = pre_ydata_k(data)
            
        Y_all.append(y_data)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])  
    X_characs = []
    for i in range(len(X_all)):
        X_charac=[]
        for x in X_all[i]:
            X_charac+=kernel_ratio[x-1]
        X_characs.append(X_charac.copy())
    X_characs,Y_all = np.array(X_characs), np.array(Y_all)
    return X_characs,Y_all

def read_data_whiteBox_bias(filename,bias=[0],norm=False,y_scale=True):
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])  
  
    src_path = "../data/Track2_final_archs.json"
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            if y_scale==True:
                Y_all.append(sub_dict[1]['acc']*100+bias[0])
            else:
                Y_all.append(sub_dict[1]['acc']+bias[0])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    if norm == True:
        kernel_ratio = [[3/7,3/6],[3/7,6/6],[5/7,3/6],[5/7,6/6],[7/7,3/6],[7/7,6/6]]
    elif norm == "max":
        kernel_ratio = [[3/7,3/7],[3/7,6/7],[5/7,3/7],[5/7,6/7],[7/7,3/7],[7/7,6/7]]
    else:
        kernel_ratio=[[3,3],[3,6],[5,3],[5,6],[7,3],[7,6]] 

    X_characs = []
    for i in range(len(X_all)):
        X_charac=[]
        for x in X_all[i]:
            X_charac+=kernel_ratio[x-1]
        X_characs.append(X_charac.copy())        
    X_characs,Y_all = np.array(X_characs), np.array(Y_all)
    return X_characs,Y_all


def read_group_charac_bias(filename,bias=[0],norm=False,y_scale=True):
    with open(filename,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])  
  
    src_path = "../data/Track2_final_archs.json"
    group_num = len(bias)
    src_y=[]
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            if y_scale==True:
                src_y.append(sub_dict[1]['acc']*100)
            else:
                src_y.append(sub_dict[1]['acc'])
    src_num = len(src_y)
    src_y.sort()
    sort_y = src_y
    assert group_num<=src_num
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            if y_scale == True:
                new_y = sub_dict[1]['acc']*100
            else:
                new_y = sub_dict[1]['acc']
            sort_index = sort_y.index(new_y)
            bias_index = int(sort_index/(src_num/group_num))
            Y_all.append(new_y+bias[bias_index])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
            
    if norm == True:
        kernel_ratio = [[3/7,3/6],[3/7,6/6],[5/7,3/6],[5/7,6/6],[7/7,3/6],[7/7,6/6]]
    elif norm == "max":
        kernel_ratio = [[3/7,3/7],[3/7,6/7],[5/7,3/7],[5/7,6/7],[7/7,3/7],[7/7,6/7]]
    else:
        kernel_ratio=[[3,3],[3,6],[5,3],[5,6],[7,3],[7,6]] 
        
    X_characs = []
    for i in range(len(X_all)):
        X_charac=[]
        for x in X_all[i]:
            X_charac+=kernel_ratio[x-1]
        X_characs.append(X_charac.copy())        
    X_characs,Y_all = np.array(X_characs), np.array(Y_all)
    return X_characs,Y_all


def read_train_bias(few_path,bias=[0]):
    with open(few_path,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    src_path = "/home/aistudio/work/data/Track2_final_archs.json"
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            Y_all.append(sub_dict[1]['acc']*100+bias[0])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
            
    X_all,Y_all = np.array(X_all), np.array(Y_all)
    return X_all,Y_all   

def read_group_bias(few_path,bias=[0]):
    with open(few_path,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    src_path = "../data/Track2_final_archs.json"
    group_num = len(bias)
    src_y=[]
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            src_y.append(sub_dict[1]['acc']*100)
    src_num = len(src_y)
    src_y.sort()
    sort_y = src_y
    assert group_num<=src_num
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            new_y = sub_dict[1]['acc']*100
            sort_index = sort_y.index(new_y)
            bias_index = int(sort_index/(src_num/group_num))
            Y_all.append(new_y+bias[bias_index])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    X_all,Y_all = np.array(X_all), np.array(Y_all)
    return X_all,Y_all  

def read_one_hot_bias(few_path,bias=[0]):
    with open(few_path,"r") as f:
        arch_dict = json.load(f)
    Y_all = []
    X_all = []
    for sub_dict in arch_dict.items():
        Y_all.append(sub_dict[1]['acc']*100)
        X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    src_path = "/home/aistudio/work/data/Track2_final_archs.json"
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            Y_all.append(sub_dict[1]['acc']*100+bias[0])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    X_one_hots = []
    fix_len=7
    for i in range(len(X_all)):
        X_one_hot=[]
        for x in X_all[i]:
            x_one_hot = [0]*fix_len
            x_one_hot[x] = 1
            X_one_hot+=x_one_hot
        X_one_hots.append(X_one_hot.copy())
    X_one_hots,Y_all = np.array(X_one_hots), np.array(Y_all)
    return X_one_hots,Y_all   

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


def try_different_method(model,train_x,train_y,test_x,test_y,model_path=None,process_label=False):
    if process_label == True:
        train_y = preprocess_label(train_y)
    
    model.fit(train_x,train_y)
    result = model.predict(test_x)
    
    if process_label == True:
        result = depre_label(result)
        
    test_rmse = math.sqrt(sum((test_y-result)*(test_y-result))/len(test_y))
    
    if model_path!=None:
        with open(model_path,"wb") as f_w:
            pickle.dump(model,f_w)
            
    train_result = model.predict(train_x)
    train_rmse = math.sqrt(sum((train_y-train_result)*(train_y-train_result))/len(train_y)) 
    return test_rmse,train_rmse

def get_model(train_json,model,mode="index",bias=[0],unsuper_json=None,gene_json=None,norm=False,index=0,toops=0,y_scale=True):
    if mode=="index":
        x_data,y_data = read_data_blackBox(train_json)
        if unsuper_json!=None:
            unsu_X_train,unsu_Y_train = read_data_blackBox(unsuper_json)
            x_data = np.concatenate((x_data,unsu_X_train),axis=0)
            y_data = np.concatenate((y_data,unsu_Y_train),axis=0)   
        if gene_json!=None:
            gene_X_train,gene_Y_train = read_RN_aug(gene_json)
            x_data = np.concatenate((x_data,gene_X_train),axis=0)
            y_data = np.concatenate((y_data,gene_Y_train),axis=0)        
    elif mode=="one-hot":
        assert unsuper_json==None
        x_data,y_data = read_data_one_hot(train_json)
    elif mode == "bias":
        x_data,y_data = read_train_bias(train_json,bias)
        if unsuper_json!=None:
            unsu_X_train,unsu_Y_train = read_data_blackBox(unsuper_json)
            x_data = np.concatenate((x_data,unsu_X_train),axis=0)
            y_data = np.concatenate((y_data,unsu_Y_train),axis=0)
        if gene_json!=None:
            gene_X_train,gene_Y_train = read_RN_aug(gene_json)
            x_data = np.concatenate((x_data,gene_X_train),axis=0)
            y_data = np.concatenate((y_data,gene_Y_train),axis=0)
    elif mode == "one-hot-bias":
        assert unsuper_json==None
        x_data,y_data = read_one_hot_bias(train_json,bias)
        if unsuper_json!=None:
            unsu_X_train,unsu_Y_train = read_data_blackBox(unsuper_json)
            x_data = np.concatenate((x_data,unsu_X_train),axis=0)
            y_data = np.concatenate((y_data,unsu_Y_train),axis=0)  
    elif mode == "group_bias":
        x_data,y_data = read_group_bias(train_json,bias)
        if unsuper_json!=None:
            unsu_X_train,unsu_Y_train = read_data_blackBox(unsuper_json)
            x_data = np.concatenate((x_data,unsu_X_train),axis=0)
            y_data = np.concatenate((y_data,unsu_Y_train),axis=0)  
    elif mode == "charac":
        x_data,y_data = read_data_whiteBox(train_json,norm,y_scale)
    elif mode == "charac_bias":
        x_data,y_data = read_data_whiteBox_bias(train_json,bias,norm,y_scale)
        if gene_json!=None:
            gene_X_train,gene_Y_train = read_RN_charac_aug(gene_json)
            x_data = np.concatenate((x_data,gene_X_train),axis=0)
            y_data = np.concatenate((y_data,gene_Y_train),axis=0)
    elif mode == "charac_bias_similar":
        x_data,y_data = read_data_whiteBox_bias(train_json,bias,norm,y_scale)
        gene_X_train,gene_Y_train = aug_data_similar_charac_mod(train_json,kstep=0.001,cstep=0.002)
        x_data = np.concatenate((x_data,gene_X_train),axis=0)
        y_data = np.concatenate((y_data,gene_Y_train),axis=0)
    elif mode =="charac_group_bias":
        x_data,y_data = read_group_charac_bias(train_json,bias,norm,y_scale)
    elif mode == "charac_group_bias_noise":
        x_real,y_real = read_data_whiteBox(train_json,norm,y_scale)  
        x_data,y_data = read_group_charac_bias(train_json,bias,norm,y_scale)
        for i in range(1):
            noise_x,noise_y = input_gaussian_noise(x_real,y_real,seed=i,SNR=1)
            x_data=np.concatenate((x_data,noise_x),axis=0)
            y_data=np.concatenate((y_data,noise_y),axis=0)
    elif mode =="charac_similar_bias_noise":
        x_real,y_real = read_data_whiteBox(train_json,norm)   
        x_data,y_data = read_group_charac_bias(train_json,[2.24, 2.2, 2.22],norm)  
        for i in range(1):
            noise_x,noise_y = input_gaussian_noise(x_real,y_real,seed=i,SNR=1)
            x_data=np.concatenate((x_data,noise_x),axis=0)
            y_data=np.concatenate((y_data,noise_y),axis=0)
        ab5_file = "../data/Track2_stage2_few_show_trainning_5ab.json" 
     
        s_datax,s_datay = read_data_whiteBox(ab5_file,norm)
        for i in range(3):
            noise_x,noise_y = input_gaussian_noise(s_datax,s_datay,seed=i,SNR=0.4)
            x_data=np.concatenate((x_data,noise_x),axis=0)
            y_data=np.concatenate((y_data,noise_y),axis=0)
        
    else:
        print("can't support this mode:",mode)
        pdb.set_trace()
    
    model.fit(x_data,y_data)
    return model


def infer_test(model_path,test_json,mode="index",norm=False,r_result=False,y_scale=True,y_pro=False):
    if mode=="index" or mode=="bias" or mode=="group_bias":
        test_x,test_y = read_data_blackBox(test_json)
    elif mode == "one-hot" or mode =="one-hot-bias":
        test_x,test_y = read_data_one_hot(test_json)
    elif "charac" in mode:
        test_x,test_y = read_data_whiteBox(test_json,norm,y_scale,y_pro=y_pro)
    with open(model_path,"rb") as f_r:
        model = pickle.load(f_r)
    result = model.predict(test_x)    
    test_rmse = math.sqrt(sum((test_y-result)*(test_y-result))/len(test_y))
    if r_result==False:
        return test_rmse
    else:
        return result

def cross_validate(data_root,model_path,save_root,model=None,mode="index",bias=[0],floder_num=5,draw=False,r_train=False,
                   unsuper_json=None,gene_json=None,norm=False,index=0,toops=0,y_scale=True):
    test_rmses  = []
    train_rmses = []
    for i in range(floder_num):
        train_json = os.path.join(data_root,"train"+str(i)+".json")
        test_json = os.path.join(data_root,"test"+str(i)+".json")

        #get model:
        model = get_model(train_json,model,mode,bias,unsuper_json,gene_json,norm=norm,index=index,toops=toops,y_scale=y_scale)
        if model_path!=None:
            with open(model_path,"wb") as f_w:
                pickle.dump(model,f_w)   
               
        test_rmse = infer_test(model_path,test_json,mode,norm,y_scale=y_scale)
        test_rmses.append(test_rmse)
        train_rmse = infer_test(model_path,train_json,mode,norm,y_scale=y_scale)
        train_rmses.append(train_rmse)
    
    if draw==True:
        print("test_rmse:",test_rmses)
        print("train_rmse",train_rmses)
        print("ave test_rmse:",sum(test_rmses)/len(test_rmses))
        print("ave train_rmse",sum(train_rmses)/len(train_rmses))
   
    ave_test_rmses = sum(test_rmses)/len(test_rmses)
    ave_train_rmses = sum(train_rmses)/len(train_rmses)
    if r_train==True:
        return ave_test_rmses,ave_train_rmses
    else:
        return ave_test_rmses

def train_model(model,train_x,train_y,model_path):
    model.fit(train_x,train_y)
    with open(model_path,"wb") as f_w:
        pickle.dump(model,f_w)

def predict(model_path,test_x):
    with open(model_path,"rb") as f_r:
        model = pickle.load(f_r)
    result = model.predict(test_x)
    return result