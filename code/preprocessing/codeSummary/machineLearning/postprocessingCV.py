#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:34:19 2018

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
import glob, os

DirPath = '~/code/preprocessing/result/raw_feature_CV/'
forecastList = [6, 12, 18, 24, 30]
#forecastList = [6, 12, 18, 24, 30]
forecastList = [30]
saveDirPath = '~/code/preprocessing/result/processed_feature_CV/'



if not os.path.isdir(saveDirPath):
     os.makedirs(saveDirPath)
     os.chmod(saveDirPath, 0o755)

for forecast in forecastList:
    inputFileList = sorted(glob.glob(DirPath + 'foward' + str(forecast) + '/' + '*csv'))

    counter = 0
    for inputFile in inputFileList:
        
            print(inputFile)
        
            OriginData = pd.read_csv(inputFile, header=0)

            Data = np.array(OriginData)
            DataXSingle = Data[:,1:-1]
            DataYSingle = Data[:,-1]

            if (counter == 0):
                counter = 1
                DataX = DataXSingle
                DataY = DataYSingle
                continue

            DataX = np.concatenate((DataX, DataXSingle), axis = 0)
            DataY = np.concatenate((DataY, DataYSingle), axis = 0)
            
    saveFileNameX = saveDirPath + 'DataX_forward' + str(forecast) + '.npy'
    saveFileNameY = saveDirPath + 'DataY_forward' + str(forecast) + '.npy'
    
    np.save(saveFileNameX, DataX)
    np.save(saveFileNameY, DataY)
    
