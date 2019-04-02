#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:43:10 2018

@author: chenxiaoxu
"""


import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import sklearn.metrics as metrics

from sklearn.model_selection import RepeatedKFold

saveDirPath = '~/code/preprocessing/result/processed_feature/'

forward = 6

inputFileX = saveDirPath + 'DataX_forward' + str(forward) + '.npy'
inputFileY = saveDirPath + 'DataY_forward' + str(forward) + '.npy'

DataX = np.load(inputFileX)
DataY = np.load(inputFileY)


C = [0.01, 0.1, 1, 10]
epsilon = [ 0.001, 0.01, 0.1]
gamma = [0.1,1,10,100]

#
#MAE_2 = []
#MSE_2 = []

MAE_max = 1000
MSE_max = 1000

for i in range(len(C)):
    
#    MAE_1 = []
#    MSE_1 = []
    
    for j in range(len(epsilon)):
        for k in range(len(gamma)):
        
            print('C:')
            print(C[i])
            print('epsilon =')
            print(epsilon[j])
            print('gamma = ')
            print(gamma[k])
            
            MAE_list = []
            MSE_list = []
        
            
            skf = RepeatedKFold(n_splits=5, n_repeats = 5, random_state=1)        
        
            for train_index, test_index in skf.split(DataX, DataY):
            
                trainX, testX = DataX[train_index], DataX[test_index]
                trainY, testY = DataY[train_index], DataY[test_index]
            
                SVR_linear = SVR(C=C[i], epsilon=epsilon[j], gamma = gamma[k], kernel='rbf')
                SVR_linear.fit(trainX, trainY)
                testY_hat = SVR_linear.predict(testX)


                MAE = metrics.mean_absolute_error(testY, testY_hat)
                MSE = metrics.mean_squared_error(testY, testY_hat)

                MAE_list.append(MAE)
                MSE_list.append(MSE)

            MAE_all = sum(MAE_list)/len(MAE_list) 
            MSE_all = sum(MSE_list)/len(MSE_list) 

            if (MAE_all < MAE_max):
                MAE_max = MAE_all
                C_max1 = C[i]
                epsilon_max1 = epsilon[j]
                gamma_max1 = gamma[k]
                
            if (MSE_all < MSE_max):
                MSE_max = MSE_all
                C_max2 = C[i]
                epsilon_max2 = epsilon[j]
                gamma_max2 = gamma[k]
             
                
            print("#######")       
                
                
            print(C_max1)
            print(epsilon_max1)
            print(gamma_max1)
            
            print('\n')
            
            print(C_max2)
            print(epsilon_max2)
            print(gamma_max2)    
            
            print("#######")      
#        MAE_1.append(sum(MAE_list)/len(MAE_list))
#        MSE_1.append(sum(MSE_list)/len(MSE_list))
        
        print("##################################")       
    
#    MAE_2.append(MAE_1)
#    MSE_2.append(MSE_1)
#    
