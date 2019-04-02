#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:58:53 2018

@author: chenxiaoxu
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import sklearn.metrics as metrics

from datetime import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import RepeatedKFold


saveDirPath = '~/code/preprocessing/result/processed_feature/'

forwardList = [6,12,18,24,30]


MAE_LR_all = []
MSE_LR_all = []

MAE_SVRl_all = []
MSE_SVRl_all = []

MAE_SVRr_all = []
MSE_SVRr_all = []

MAE_Rshift_all = []
MSE_Rshift_all = []

MAE_LR_delta_all = []
MSE_LR_delta_all = []

for forward in forwardList:
      
    print(forward)
    
    
    inputFileX = saveDirPath + 'DataX_forward' + str(forward) + '.npy'
    inputFileY = saveDirPath + 'DataY_forward' + str(forward) + '.npy'

    DataX = np.load(inputFileX)
    DataY = np.load(inputFileY)  
    
    skf = RepeatedKFold(n_splits = 5, n_repeats = 5, random_state = 1)
    
    MAE_LR_list = []
    MSE_LR_list = []  
    
    MAE_SVRl_list = []
    MSE_SVRl_list = [] 

    MAE_SVRr_list = []
    MSE_SVRr_list = [] 

    MAE_Rshift_list = []
    MSE_Rshift_list = [] 
    
    MAE_LR_delta_list = []
    MSE_LR_delta_list = []     
    
    
    for train_index, test_index in skf.split(DataX, DataY):
        trainX, testX = DataX[train_index], DataX[test_index]
        trainY, testY = DataY[train_index], DataY[test_index]
       
    
        LR = LinearRegression()
        LR.fit(trainX, trainY)
        testY_hat = LR.predict(testX)
    
        MAE_LR = metrics.mean_absolute_error(testY_hat, testY)
        MSE_LR = metrics.mean_squared_error(testY_hat, testY)
    
        MAE_LR_list.append(MAE_LR)
        MSE_LR_list.append(MSE_LR)


        SVR_linear = SVR(C=10, epsilon=0.001, kernel='linear')
        SVR_linear.fit(trainX, trainY)
        testY_hat = SVR_linear.predict(testX)

        MAE_SVRl = metrics.mean_absolute_error(testY_hat, testY)
        MSE_SVRl = metrics.mean_squared_error(testY_hat, testY)

        MAE_SVRl_list.append(MAE_SVRl)
        MSE_SVRl_list.append(MSE_SVRl)

        SVR_rbf = SVR(C=1, epsilon=0.001, gamma = 0.5, kernel='rbf')
        SVR_rbf.fit(trainX, trainY)
        testY_hat = SVR_rbf.predict(testX)
    
        MAE_SVRr = metrics.mean_absolute_error(testY_hat, testY)
        MSE_SVRr = metrics.mean_squared_error(testY_hat, testY)

        MAE_SVRr_list.append(MAE_SVRr)
        MSE_SVRr_list.append(MSE_SVRr)
#
        testY_hat  = testX[:,-2]

        MAE_Rshift = metrics.mean_absolute_error(testY_hat, testY)
        MSE_Rshift = metrics.mean_squared_error(testY_hat, testY)

        MAE_Rshift_list.append(MAE_Rshift)
        MSE_Rshift_list.append(MSE_Rshift)

        trainX_delta = np.zeros((trainX.shape[0], 2))
        trainX_delta[:,0] = trainX[:,-2]
        trainX_delta[:,1] = trainX[:,-4] - trainX[:,-5]

        LR_delta = LinearRegression()
        LR_delta.fit(trainX_delta, trainY)
        
        testX_delta = np.zeros((testX.shape[0], 2))
        testX_delta[:,0] = testX[:,-2]
        testX_delta[:,1] = testX[:,-4] - testX[:,-5]        
        
        testY_hat = LR_delta.predict(testX_delta)

        MAE_LR_delta = metrics.mean_absolute_error(testY_hat, testY)
        MSE_LR_delta = metrics.mean_squared_error(testY_hat, testY)

        MAE_LR_delta_list.append(MAE_LR_delta)
        MSE_LR_delta_list.append(MSE_LR_delta)
        
        
    MAE_LR_all.append(sum(MAE_LR_list)/len(MAE_LR_list))   
    MSE_LR_all.append(sum(MSE_LR_list)/len(MSE_LR_list))   
        
    MAE_SVRl_all.append(sum(MAE_SVRl_list)/len(MAE_SVRl_list))   
    MSE_SVRl_all.append(sum(MSE_SVRl_list)/len(MSE_SVRl_list))           
        
    MAE_SVRr_all.append(sum(MAE_SVRr_list)/len(MAE_SVRr_list))   
    MSE_SVRr_all.append(sum(MSE_SVRr_list)/len(MSE_SVRr_list)) 

    MAE_Rshift_all.append(sum(MAE_Rshift_list)/len(MAE_Rshift_list))   
    MSE_Rshift_all.append(sum(MSE_Rshift_list)/len(MSE_Rshift_list))     

    MAE_LR_delta_all.append(sum(MAE_LR_delta_list)/len(MAE_LR_delta_list))   
    MSE_LR_delta_all.append(sum(MSE_LR_delta_list)/len(MSE_LR_delta_list))   
         






forwardList = [3,6,9,12,15]


plt.plot(forwardList, MAE_LR_all, label = 'linear regresssion')
plt.legend()
plt.plot(forwardList, MAE_SVRl_all, label = 'linear SVR')
plt.legend()
plt.plot(forwardList, MAE_SVRr_all, label = 'rbf SVR')
plt.legend()
plt.plot(forwardList, MAE_Rshift_all, label = 'Rshift')
plt.legend()
plt.plot(forwardList, MAE_LR_delta_all, label = 'linear_delta')
plt.legend()
plt.title('Mean Absolute Error')
plt.xticks(forwardList)

#plt.plot(forwardList, MSE_LR_all, label = 'linear regresssion')
#plt.legend()
#plt.plot(forwardList, MSE_SVRl_all, label = 'linear SVR')
#plt.legend()
#plt.plot(forwardList, MSE_SVRr_all, label = 'rbf SVR')
#plt.legend()
#plt.plot(forwardList, MSE_Rshift_all, label = 'Rshift')
#plt.legend()
#plt.plot(forwardList, MSE_LR_delta_all, label = 'linear_delta')
#plt.legend()
#plt.title('Root Mean Square Error')
#plt.xticks(forwardList)







#
#testFileName = '~/code/preprocessing/result/raw_feature/foward30/featureSet_2018091618foward30.csv'
#OriginData = pd.read_csv(testFileName, header=0)
#
##            Data = np.array(OriginData)
##            DataXSingle = Data[:,1:-1]
##            DataYSingle = Data[:,-1]
#
#TimeStamp = OriginData['TimeStamp'].values
#TimeStamp = [time(hour = int(str(value)[-6:-4]), minute = int(str(value)[-4:-2]), second = int(str(value)[-2:])) for value in TimeStamp]
##TimeStamp = [str(value)[-6:-4] + ":" + str(value)[-4:-2] + ":" + str(value)[-2:] for value in TimeStamp]
#
#Data = np.array(OriginData)
#DataX = Data[:,1:-1]
#DataY = Data[:,-1]
#
#DataY_hat_LR = LR.predict(DataX)
#DataY_hat_SVRl = SVR_linear.predict(DataX)
#DataY_hat_SVRr = SVR_rbf.predict(DataX)
#
#DataY_hat_Rshift = DataX[:, -2]
#
#DataX_delta = np.zeros((DataX.shape[0], 2))
#DataX_delta[:,0] = DataX[:,-2]
#DataX_delta[:,1] = DataX[:,-4] - DataX[:,-5]
#DataY_hat_LRdelta = LR_delta.predict(DataX_delta)
#
#plt.plot(TimeStamp, DataY, label = 'ground-truth')
#plt.legend()
#plt.plot(TimeStamp, DataY_hat_LR, label = 'linear regression')
#plt.legend()
#plt.plot(TimeStamp, DataY_hat_LRdelta, label = 'linear delta')
#plt.legend()
#plt.plot(TimeStamp, DataY_hat_Rshift, label = 'radiance shift')
#plt.legend()
#plt.plot(TimeStamp, DataY_hat_SVRl, label  = 'SVR linear')
#plt.legend()
#plt.plot(TimeStamp, DataY_hat_SVRr, label = 'SVR rbf')
#plt.legend()
#
#
#plt.xticks([TimeStamp[0], TimeStamp[29], TimeStamp[59], TimeStamp[89],  TimeStamp[119]],
#          [TimeStamp[0], TimeStamp[29], TimeStamp[59], TimeStamp[89],  TimeStamp[119]])





