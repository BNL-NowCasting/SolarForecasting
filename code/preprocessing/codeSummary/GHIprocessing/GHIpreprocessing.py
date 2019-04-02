#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:03:04 2018

@author: chenxiaoxu
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os


Sensor_number = 1

DirPath = '~/codeSummary/GHIprocessing/originalGHI/'
filename = DirPath + '2018-09_bps_' + str(Sensor_number) + '_second.dat'

outPath = '~/codeSummary/GHIprocessing/processedGHI/Sensor' + str(Sensor_number) + '/'

if not os.path.isdir(outPath):
    os.makedirs(outPath)
    os.chmod(outPath,0o755)


df = pd.read_csv(filename, sep = ',', header = 0,  skiprows = [0, 2, 3])
TimeStamp = df['TIMESTAMP']
Rad = df['SP2A_H']
#    del df
TimeStamp = TimeStamp.str.split()
DataAll = pd.DataFrame(TimeStamp.values.tolist(), columns = ['date', 'time'])
DataAll['Rad'] = Rad
#    del TimeStamp
Dates = DataAll['date']
Dates = DataAll['date'].unique()


for date in Dates:
    
    print(date)

    Time = DataAll[DataAll['date'] == date]['time']
    Ghi = DataAll[DataAll['date'] == date]['Rad']
    Time = pd.to_datetime(Time).dt.time
        
    StartTime = datetime.time(8, 0, 0)
    EndTime = datetime.time(19, 0, 0)
        
    mask = (Time > StartTime) & (Time <= EndTime)
            
    Time = Time.loc[mask]
    Ghi = Ghi.loc[mask]
        
    TimeSeries = pd.Series(Ghi.values, index = Time.values)
    year = date[0:4]
    month = date[5:7]
    day = date[8:]
            
    savepath_CSV = outPath +  'GHI' + year + month + day + '.csv'
        
    TimeSeries.to_csv(savepath_CSV)
        


    
#for date in Dates:
#    print(date)
#
#    Time = DataAll[DataAll['date'] == date]['time']
#    Ghi = DataAll[DataAll['date'] == date]['Rad']
#    Time = pd.to_datetime(Time).dt.time
#        
#    StartTime = datetime.time(12, 0, 0)
#    EndTime = datetime.time(22, 0, 0)
#        
#    mask = (Time > StartTime) & (Time <= EndTime)
#        
#    Time = Time.loc[mask]
#    Ghi = Ghi.loc[mask]
#        
#    TimeSeries = pd.Series(Ghi.values, index = Time.values)
#    outpath_CSV = outpath_CSV_abs + '/' + date
#    if not os.path.isdir(outpath_CSV):
#        os.makedirs(outpath_CSV)
#        os.chmod(outpath,0o755)        
#            
#    savepath_CSV = outpath_CSV +  '/' + date + '-' + str(sensor) + '.csv'
#        
#    TimeSeries.to_csv(savepath_CSV)
        
        
#        fig = plt.figure()
#        plt.plot(TimeSeries, 'b', linewidth = 0.5)
#        plt.title(date)
#        plt.xlabel('TimeStamp')
#        plt.ylabel('Radiance')
#        plt.xlim([Time.values[0], Time.values[-1]])
#        plt.ylim([0, 2000])
#        
#        outpath = outpath_abs + '/' + date
#        if not os.path.isdir(outpath):
#            os.makedirs(outpath)
#            os.chmod(outpath,0o755)
#              
#        savepath = outpath +  '/' + date + '-' + str(sensor) + '.jpg'
#        fig.savefig(savepath)      
#        plt.close()
