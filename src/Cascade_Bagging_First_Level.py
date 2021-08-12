import os 
import numpy as np
from utils import *
import xgboost as xgb
from sklearn import ensemble, linear_model, neighbors, \
    decomposition, manifold, neural_network, svm
from sklearn.tree import DecisionTreeRegressor

model_seed = 0
model_zoo = {
    'rf': ensemble.RandomForestRegressor(n_estimators=55, min_samples_leaf=6, random_state=model_seed),
    'et': ensemble.ExtraTreesRegressor(n_estimators=30, min_samples_split=10, random_state=model_seed),
    'knn': neighbors.KNeighborsRegressor(n_neighbors=34, weights='distance', algorithm='auto'),
    'ridge': linear_model.Ridge(alpha=1.9, normalize=True, tol=1e-3, random_state=model_seed),
    'lasso': linear_model.Lasso(alpha=1e-5, normalize=True, max_iter=1000, tol=1e-4, random_state=model_seed),
    'bayes': linear_model.BayesianRidge(n_iter=300, alpha_1=1e-6, alpha_2=1.2e-3, lambda_1=1e-6, lambda_2=1e-6),
    'svm': svm.NuSVR(C=1, kernel='rbf', gamma='scale', tol=5e-3), #before 5e-3 
    'adaboost': ensemble.AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=4), n_estimators=100, 
                                           learning_rate=3, loss='exponential', random_state=model_seed),
}
best_xgb = xgb.XGBRegressor(max_depth=4,reg_lambda=2.4,learning_rate=0.24,
                            n_estimators=30,objective='reg:squarederror')


'''
Train first-level models 
'''
def read_group_charac_bias_sample(filename,bias,norm,real_sam=None,model200_sam=None,seed=0,
                                  y_scale=True,model200_path=None,y_pro=False):
    #fix random package
    random.seed(seed)
    #fix numpy package
    np.random.seed(seed)  
    X_all = []   
    Y_all = []
    Y_real = []
    X_real = []
    sam_X_real = []
    sam_Y_real = []
    with open(filename,"r") as f:
        arch_dict = json.load(f)    
    for sub_dict in arch_dict.items():
        if y_scale == True:
            y_data = sub_dict[1]['acc']*100
        else:
            y_data = sub_dict[1]['acc']

        if y_pro == True:
            assert y_data<=1  
            y_data = pre_ydata(y_data)
        Y_real.append(y_data)
        X_real.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])  
    

    if real_sam!=None:
        N = range(len(X_real))
        sam_index = np.random.choice(N,real_sam,replace=True)
        for i in sam_index:
            X_all.append(X_real[i])
            Y_all.append(Y_real[i])
            sam_X_real.append(X_real[i])
            sam_Y_real.append(Y_real[i])
    else:
        X_all = X_real
        Y_all = Y_real
    
    model200_X = []
    model200_Y = []
    src_path = model200_path
    group_num = len(bias)
    src_y=[]
    with open(src_path,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            if y_scale == True:
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
            if y_scale ==True:
                new_y = sub_dict[1]['acc']*100
            else:
                new_y = sub_dict[1]['acc']
            sort_index = sort_y.index(new_y)
            bias_index = int(sort_index/(src_num/group_num))
            
            new_y = new_y+bias[bias_index]
            if y_pro == True:
                assert new_y<=1
                new_y = pre_ydata(new_y)
                
            model200_Y.append(new_y)
            model200_X.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    
    if model200_sam!=None and model200_sam!="NO":
        N = range(len(model200_X))
        sam_index = np.random.choice(N,model200_sam,replace=True)
        for i in sam_index:
            X_all.append(model200_X[i])
            Y_all.append(model200_Y[i])
    elif model200_sam =="NO":
        pass 
    else:
        before_len = len(X_all)
        X_all = X_all+model200_X
        Y_all = Y_all+model200_Y
        after_len=len(X_all)
        assert after_len>before_len

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
    

    sam_characs_real = []
    for i in range(len(sam_X_real)):
        sam_charac_real=[]
        for x in sam_X_real[i]:
            sam_charac_real+=kernel_ratio[x-1]
        sam_characs_real.append(sam_charac_real.copy())        
    sam_characs_real,sam_Y_real = np.array(sam_characs_real), np.array(sam_Y_real)
    
    return X_characs,Y_all,sam_characs_real,sam_Y_real

def bagging_data_xgb(data_root,model, mode, 
                    bias, model_num, norm,
                    test_path,
                    result_path,
                    model_path,
                    sam_mode,
                    real_sam,
                    model200_sam,
                    y_scale,
                    model200_path,
                    y_pro=False):
    results=[]
    rmses=[]
    #无放回    
    for i in range(model_num):
        if sam_mode == "folder":
            train_json = os.path.join(data_root,"train"+str(i)+".json")
            test_json = os.path.join(data_root,"test"+str(i)+".json")
            model = get_model(train_json,model,mode,bias,norm=norm)
        elif sam_mode == "random":
            x_data,y_data, sam_x_real,sam_y_real = read_group_charac_bias_sample(data_root,bias,
                                                                                norm,real_sam=real_sam,
                                                                                model200_sam=model200_sam,
                                                                                seed=i,y_scale=y_scale,
                                                                                model200_path=model200_path,
                                                                                y_pro=y_pro)  
            for j in range(1):
                noise_x,noise_y = input_gaussian_noise(x_data,y_data,seed=j,SNR=1)    
                x_data=np.concatenate((x_data,noise_x),axis=0)
                y_data=np.concatenate((y_data,noise_y),axis=0)   
            print(i,"-th x_data:","number is ",len(x_data))
            model = model.fit(x_data,y_data)
        model_path_now = os.path.join(model_path,mode + "_" + str(i) + "_w.pt")
        if model_path_now!=None:
            with open(model_path_now,"wb") as f_w:
                pickle.dump(model,f_w)               
        pre_result = infer_test(model_path_now,test_path,mode,norm,r_result=True)
        #print(pre_result)
        if y_pro == True:
            for now_result in range(len(pre_result)):
                inv_result = inv_ydata(pre_result[now_result])
                pre_result[now_result]=inv_result
        results.append(pre_result)        
    
    #compute average values
    ave_result = sum(results)/len(results)
    
    #write result
    pakage_result(ave_result,test_path,result_path,y_scale)
    return ave_result

if __name__ == "__main__":  
    train_path = "../data/Track2_stage2_few_show_trainning.json"   #trainset path
    test_path  = "../data/Track2_stage2_test.json"                 #testset  path
    result_path = "../results/first_level_new_bagging_result.json" #the path  of first-levle models bagging result 
    model_path = "../models/first_level_new"                       #the save path of first-level models 
    model200_path = "../data/dynamic_xgb.json"                     #modified dataset 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    norm=True
    mode = "charac_group_bias"  
    real_sam = 200              
    model200_sam = 300          
    sam_mode = "random"         
    base = "svm"               
    model_num = 100           
    model = model_zoo[base]
    y_scale = False             
    y_pro = False                  
    ave_result=bagging_data_xgb(data_root=train_path,model=best_xgb, mode=mode, 
                                bias=[0,0,0],model_num=model_num, norm=norm,
                                test_path=test_path,result_path=result_path,model_path=model_path,
                                sam_mode=sam_mode,real_sam=real_sam,model200_sam=model200_sam,
                                y_scale=y_scale,model200_path=model200_path,y_pro=y_pro)