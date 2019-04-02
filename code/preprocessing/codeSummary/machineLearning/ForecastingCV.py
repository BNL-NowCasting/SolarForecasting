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


saveDirPath = '~/code/preprocessing/result/processed_feature_CV/'

forward = 30

inputFileX = saveDirPath + 'DataX_forward' + str(forward) + '.npy'
inputFileY = saveDirPath + 'DataY_forward' + str(forward) + '.npy'

trainX = np.load(inputFileX)
trainY = np.load(inputFileY)  

    

    
LR = LinearRegression()
LR.fit(trainX, trainY)



SVR_linear = SVR(C=10, epsilon=0.001, kernel='linear')
SVR_linear.fit(trainX, trainY)


SVR_rbf = SVR(C=1, epsilon=0.001, gamma = 10, kernel='rbf')
SVR_rbf.fit(trainX, trainY)


trainX_delta = np.zeros((trainX.shape[0], 2))
trainX_delta[:,0] = trainX[:,-2]
trainX_delta[:,1] = trainX[:,-4] - trainX[:,-5]

LR_delta = LinearRegression()
LR_delta.fit(trainX_delta, trainY)
        
testFileName = '~/code/preprocessing/result/raw_feature/foward30/featureSet_2018091618foward30.csv'
OriginData = pd.read_csv(testFileName, header=0)


TimeStamp = OriginData['TimeStamp'].values
TimeStamp = [time(hour = int(str(value)[-6:-4]), minute = int(str(value)[-4:-2]), second = int(str(value)[-2:])) for value in TimeStamp]
#TimeStamp = [str(value)[-6:-4] + ":" + str(value)[-4:-2] + ":" + str(value)[-2:] for value in TimeStamp]

Data = np.array(OriginData)
DataX = Data[:,1:-1]
DataY = Data[:,-1]

DataY_hat_LR = LR.predict(DataX)
DataY_hat_SVRl = SVR_linear.predict(DataX)
DataY_hat_SVRr = SVR_rbf.predict(DataX)

DataY_hat_Rshift = DataX[:, -2]

DataX_delta = np.zeros((DataX.shape[0], 2))
DataX_delta[:,0] = DataX[:,-2]
DataX_delta[:,1] = DataX[:,-4] - DataX[:,-5]
DataY_hat_LRdelta = LR_delta.predict(DataX_delta)


plt.plot(TimeStamp, DataY_hat_LR, label = 'linear regression')
plt.legend()
plt.plot(TimeStamp, DataY_hat_SVRl, label  = 'SVR linear')
plt.legend()
plt.plot(TimeStamp, DataY_hat_SVRr, label = 'SVR rbf')
plt.legend()
plt.plot(TimeStamp, DataY_hat_Rshift, label = 'radiance shift')
plt.legend()
plt.plot(TimeStamp, DataY_hat_LRdelta, label = 'linear delta')
plt.legend()

plt.plot(TimeStamp, DataY, label = 'ground-truth')
plt.legend()
plt.title("forecasting GHIs")


plt.xticks([TimeStamp[0], TimeStamp[29], TimeStamp[59], TimeStamp[89],  TimeStamp[119]],
          [TimeStamp[0], TimeStamp[29], TimeStamp[59], TimeStamp[89],  TimeStamp[119]])





