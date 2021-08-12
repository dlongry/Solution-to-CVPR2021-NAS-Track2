# Solution to CVPR2021 NAS Track2

## Introduction
Tihs work ranks the 3rd place in the Performance Prediction Track of CVPR2021 1st Lightweight NAS Challenge.

For a detailed description of technical details and experimental results, please refer to our paper:  
[Cascade Bagging for Accuracy Prediction with Few Training Samples](https://arxiv.org/)


## Run pretrained model：
Step1: download pretrained models: 
* links ：https://pan.baidu.com/s/1QTvQIJ-Fdd8uu93XFlm-Bw 
* access code：ruza

Step2：set folder structure  as below:

    |─data    
    |─models  (unzip folder)  
    |  ├─first_level  
    |  └─sec_level  
    |─result  
    |─src  



Step3: run command:
        
        cd ./src
        python Pretrained_Prediction.py


## Train model:
 Step1: calibrate weak labels:  
    Please refer to ./src/Label_Calibration.py
    
 Step2: generate training sample with noise:  
    Please refer to ./src/utils.py (函数名)
    
  Step3: train first-level models    
    Please refer to ./src/Cascade_Bagging_First_Level.py
    
  Step4: train second-level models:  
    Please refer to ./src/Cascade_Bagging_Second_Level.py
 
 [You also can run the project on AI Studio ](https://aistudio.baidu.com/aistudio/projectdetail/1968445) 
