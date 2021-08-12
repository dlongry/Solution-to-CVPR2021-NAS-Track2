from utils import *
import xgboost as xgb
from sklearn import ensemble, linear_model, neighbors, \
    decomposition, manifold, neural_network, svm
svm_model  = svm.NuSVR(C=1, kernel='rbf', gamma='scale', tol=5e-3)

'''
find bias to calibrate weak labels
'''
def grid_search_bias():
    grid_max_depths = [4]                    
    grid_learn_rates = [0.24]
    grid_n_estimators= [30]
    grid_objectives = ['reg:squarederror']
    grid_lambdas = [2.4]  

    min_rmse=9999
    min_max_depth=999
    min_learn_rate = 999
    min_n_estimator = 999
    min_objective = ""
    min_lambda = 0
    min_bias = [0,0,0]
    i=0
    for bia1 in range(15,25,1):
        for bia2 in range(15,25,1):
            for bia3 in range(15,25,1)
                bias = [bia1/10,bia2/10,bia3/10]
                for max_depth in grid_max_depths:
                    for learn_rate in grid_learn_rates:
                        for n_estimator in grid_n_estimators:
                            for objective in grid_objectives:
                                for now_lambda  in grid_lambdas:
                                    i+=1
                                    print(i)
                                    model_xgb = xgb.XGBRegressor(max_depth=max_depth,reg_lambda=now_lambda,learning_rate=learn_rate,
                                                                 n_estimators=n_estimator,objective=objective)
                                    #Leave-One-Out Cross Validation
                                    test_rmse,train_rmse = cross_validate(data_root="../data/s2_choicecross",model=model_xgb,
                                                                          mode="charac_group_bias_noise",bias=bias,floder_num=31,
                                                                          r_train=True,norm=True,index=0,toops=0)
                                    #best bias has lowest rmse
                                    if test_rmse<min_rmse:
                                        min_rmse=test_rmse
                                        min_max_depth = max_depth
                                        min_learn_rate = learn_rate
                                        min_n_estimator = n_estimator
                                        min_objective = objective
                                        min_lambda = now_lambda
                                        min_bias = bias
                                        print("test",test_rmse,"train:",train_rmse,"mbias",min_bias,"depth:",min_max_depth,
                                              "lr:",min_learn_rate,"m_lam",min_lambda,'n_est:',min_n_estimator,"obj:",min_objective)
'''
merging prediction to calibrate weak label:
'''
model_root    = "../models/baggings1_svmt3_dy_xgb" # the path of predictor
model200_json = "../data/dynamic_xgb_new.json"  #the save path of modified data

group_bias= [0.0224, 0.022, 0.0222]  #obtained by  grid_search_bias() 
norm=True
mode = "charac_group_bias"
y_scale = False


def get_json200_data(model200_json):
    X_all=[]
    Y_all=[]
    with open(model200_json,"r") as f:
        arch_dict = json.load(f)    
        for sub_dict in arch_dict.items():
            Y_all.append(sub_dict[1]['acc'])
            X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4,16)[2])
    return X_all,Y_all
            
      
def get_pre200_data(model_root,model200_json, mode, norm,y_scale=False):    
    pre200_data=[]
    file_list = os.listdir(model_root)  
    for i in range(0,len(file_list)):
        path = os.path.join(model_root,file_list[i])
        if os.path.isfile(path) and ".pt" in path:
            pre_result = infer_test(path,model200_json,mode,norm,r_result=True)
            pre200_data.append(pre_result)
    ave_pre200_data = sum(pre200_data)/len(pre200_data)
    if y_scale==True:
        ave_pre200_data=ave_pre200_data/100
    return ave_pre200_data
'''
Dynamic Programming: one execution calibrate one data weak label 
'''
def dynamic_modify_200json(X_all,src_data,pre_data,src_200json,save_json,indexes,best_json):
    with open(src_200json,"r") as f_r:
        arch_dict = json.load(f_r)
    a_min = 999
    rmse_min = 9999
    for z in range(11):
        a=z/10
        i=0
        for sub_dict in arch_dict.items():
            if i<len(indexes):
                result = indexes[i]
            elif i==len(indexes):
                result=src_data[i]*a+(1-a)*pre_data[i]
            else:
                result=src_data[i]
            sub_dict[1]['acc'] = float(result)
            assert (X_all[i] == np.array(sub_dict[1]['arch']).T.reshape(4,16)[2]).all()  #确保结构是一样的

            i+=1
        with open(save_json,"w") as f_w:
            json.dump(arch_dict,f_w)

        svm_model  = svm.NuSVR(C=1, kernel='rbf', gamma='scale', tol=5e-3)
        train200_path = save_json
        test31_path = "../data/Track2_stage2_few_show_trainning.json"
        X_all_now,Y_all_now = read_data_charac(train200_path,norm=True,y_scale=False)
        X_test_now,Y_test_now = read_data_charac(test31_path,norm=True,y_scale=False)
        svm_model = svm_model.fit(X_all_now,Y_all_now)
        pre_result = svm_model.predict(X_test_now)    
        test_rmse = math.sqrt(sum((Y_test_now-pre_result)*(Y_test_now-pre_result))/len(Y_test_now))
        if test_rmse < rmse_min:
            a_min = a
            rmse_min = test_rmse
            with open(best_json,"w") as f_w:
                json.dump(arch_dict,f_w)
    print("index:",len(indexes),"a=",a_min," rmse=",rmse_min)        
    indexes.append(src_data[len(indexes)]*a_min+(1-a_min)*pre_data[len(indexes)])
    return indexes
        


if __name__=="__main__":
    save_json  = '../data/temp.json'               # the temp save file path
    best_json = "../data/dynamic_xgb_svm_t1.json"  #the save path of final calibration labels
    ave_pre200_data = get_pre200_data(model_root=model_root,model200_json=model200_json,mode=mode,norm=norm,y_scale=y_scale)
    X_all,Y_all = get_json200_data(model200_json=model200_json)
    indexes=[]
    for i in range(200):
        indexes=dynamic_modify_200json(X_all=X_all,src_data=Y_all,pre_data=ave_pre200_data,src_200json=model200_json,
                                       save_json=save_json,indexes=indexes,best_json=best_json)

 