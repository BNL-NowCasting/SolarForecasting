#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:27:41 2020

@author: chenxiaoxu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import normalize,scale
import sklearn.metrics as metrics
from sklearn.model_selection import RepeatedKFold
from copy import deepcopy
import pickle
#matplotlib.use('agg')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize, scale
import os
import os.path
import argparse

parser = argparse.ArgumentParser(description= 'Solar')
parser.add_argument('--forward', type = int,  default = 10)
args = parser.parse_args()

# feature_file_path = '../data/feature/'
# GHI_file_path = '../data/GHI/'

data_file_dir = 'Training_data/'


DataX = np.load(data_file_dir + 'DataX_forward_' + str(args.forward) + '.npy')
DataY = np.load(data_file_dir + 'DataY_forward_' + str(args.forward) + '.npy')

C = [0.1, 1, 10,50,100,500]
C = [1]
epsilon = [0.1]
C_max1, C_max2, epsilon_max1, epsilon_max2 = [0]*4

MAE_max = 1e8
MSE_max = 1e8    

for cc in C:

    for ep in epsilon:

        print('C:', cc, '  epsilon =', ep)

        MAE_list = []
        MSE_list = []

        skf = RepeatedKFold(n_splits=2, n_repeats = 2, random_state=1)        

        for train_index, test_index in skf.split(DataX, DataY):

            trainX, testX = DataX[train_index], DataX[test_index]
            trainY, testY = DataY[train_index], DataY[test_index]

            SVR_linear = SVR(C=cc, epsilon=ep, kernel='linear', gamma='scale')
            SVR_linear.fit(trainX, trainY)
            testY_hat = SVR_linear.predict(testX)

            MAE = metrics.mean_absolute_error(testY, testY_hat)
            MSE = metrics.mean_squared_error(testY, testY_hat)

            MAE_list.append(MAE)
            MSE_list.append(MSE)

        MAE_all = sum(MAE_list)/len(MAE_list) 
        MSE_all = sum(MSE_list)/len(MSE_list) 
        print('MAE and MSE:', MAE_all, MSE_all,'\n')


        if (MAE_all < MAE_max):
            MAE_max = MAE_all
            C_max1 = cc
            epsilon_max1 = ep
            opt_model=deepcopy(SVR_linear)

        if (MSE_all < MSE_max):
            MSE_max = MSE_all
            C_max2 = cc
            epsilon_max2 = ep

if not os.path.isdir('Trained_SVM'):
    os.makedirs('Trained_SVM')
    os.chmod('Training_SVM', 0o755)
    
with open('Training_SVM/SVR_rbf_1010{:02d}.md99'.format(args.forward),'wb') as fmod:
    pickle.dump(opt_model,fmod)

print("#######")       
print('C_max1:', C_max1)
print('Eps_max1:', epsilon_max1)
print('C_max2:', C_max2)
print('Eps_max2:', epsilon_max2)
print("#######")      
print("##################################")  
print('Min MAE and MSE:', MAE_max, MSE_max)  
