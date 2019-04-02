#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:09:27 2018

@author: chenxiaoxu
"""
import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
from collections import deque
import mncc,multiprocessing
import utility
import traceback
import pandas as pd
import datetime
import pysolar.solar as ps



MAX_INTERVAL = 180
SAVE_FIG = True
SAVE_RESULT = True
PRINT = True
WRITE = True
###########################################################################
#Since HD2C is in the center of BNL, nearest ot the solar winter farm,
#we only calculate the wind-field velocity using HD2C. 
#The timestamp when the velocity are calculated are treated as valid time
#points, with which cloud height are calculated 
###########################################################################

###########################################################################
#The HD1C only has data until Sep 12th
###########################################################################

###############################################################################
#########################################
#Global Variable Settings
#########################################

camCenterID = 'HD2C'
camIDs=['HD1A', 'HD1B', 'HD2B', 'HD2C', 'HD3A', 'HD3B', 'HD4A', 'HD4B', 'HD5A', 'HD5B']
#camIDs = ['HD2C']
#camIDs = ['HD1A', 'HD1B', 'HD2B', 'HD2C']
camIDs = ['HD2B', 'HD2C', 'HD3B']
groups={'HD1A':['HD1B', 'HD2B'], 
        'HD1B':['HD1A', 'HD2C'], 
        'HD2B':['HD2C', 'HD3A'], 
        'HD2C':['HD2B', 'HD3B'],
        'HD3A':['HD3B', 'HD4A'],
        'HD3B':['HD3A', 'HD4B'], 
        'HD4A':['HD4B', 'HD5A'],
        'HD4B':['HD4A', 'HD5A', 'HD3B'],
        'HD5A':['HD5B', 'HD4A', 'HD4B'],
        'HD5B':['HD5A', 'HD4B']}

GHI_Coor = {'GHI1':   [40.868972, -72.852225]}


camIDs4height = ['HD2C']
Date = '20180902'
TimeRange = [17,18,19]
inpath = '~/data/images/'
Dirpath = '~/pipeLineOutput/'
outpath = Dirpath + 'final/pipeline/'
GHIpath = '~/GHI/GHI' + Date + '.csv' 
GHIs = pd.read_csv(GHIpath, sep = ',', header = None, index_col = 0)

outpathFeature = outpath + 'feature/'
outpathDocument = outpath + 'document/' 

if not os.path.isdir(outpathFeature):
    os.makedirs(outpathFeature)
    os.chmod(outpathFeature, 0o755)

if not os.path.isdir(outpathDocument):
    os.makedirs(outpathDocument)
    os.chmod(outpathDocument, 0o755)

cameraObjects = {}

def cameraObjectInitialize(camID):
    cameraObject = cam.camera(camID, max_theta = 70, nx = 1000, ny = 1000)
    return cameraObject

p = multiprocessing.Pool(len(camIDs))
cameraObjectsList = p.map(cameraObjectInitialize, camIDs)
p.close()
if PRINT:
    print("finish initiliazing object")


for i in range(0, len(camIDs)):
    camID = camIDs[i]
    cameraObjects[camID] = cameraObjectsList[i]

cameraCenterObject = cameraObjects[camCenterID]

q = deque()
fft = deque()
flag =[-1]

shape = (cameraCenterObject.nx, cameraCenterObject.ny)
convolver = mncc.Convolver(shape, shape, threads = 4, dtype = np.float32) 

#tmpfsNameCenter = tmpfs + camCenterID + '/' + Date + '/'
#if not os.path.isdir(tmpfsNameCenter):
#    os.makedirs(tmpfsNameCenter)
#    os.chmod(tmpfsNameCenter, 0o755)

predictForwardList = [6,12,18,24,30, 48, 60]

availableList = [0]*len(predictForwardList)
totalList = [0]*len(predictForwardList)
    
for hour in TimeRange:
    
    FeatureSet = pd.DataFrame(columns = ['TimeStamp','Rave1', 'Rmin1', 'Rmax1', 'Gave1', 'Gmin1', 'Gmax1', \
                                     'Bave1', 'Bmin1', 'Bmax1', 'Rave2', 'Rmin2', 'Rmax2', \
                                     'Gave2', 'Gmin2', 'Gmax2', 'Bave2', 'Bmin2', 'Bmax2', 'RBR1', 'RBR2', 'CldFraction', 'GHI1', 'GHI2', 'GHIy']) 

    FeatureSetList = [FeatureSet]*len(predictForwardList)    
            
    DocumFilename = outpathDocument + Date + 'hour' + str(hour) + '.txt'
    Document = open(DocumFilename, 'w') 
    Document.write(Date + '\n')
    Document.close()

    flist = sorted(glob.glob(inpath + camCenterID + '/' + Date + '/' + camCenterID + '_' + Date + str(hour) + '*jpg'))
    
    if len(flist) == 0:
        if PRINT:
            print("Warning: File list is empty")      
        continue
    try:    
        for f in flist:
            TimeStamp = os.path.basename(f)[-10:-4]

       
        
            img = cam.image(cameraCenterObject, f)
            img.undistort(cameraCenterObject, rgb = True)        
            if img.rgb is None:
                continue
            img.cloud_mask(cameraCenterObject)
            
            cloudMask = img.cm
            cloudFraction = (cloudMask>0).sum()/(cloudMask.shape[0]*cloudMask.shape[1])            
            
            q.append(img)
    
            if len(q) <= 1:
                continue
        
            if (q[-1].time - q[-2].time).seconds >= MAX_INTERVAL:
                q.popleft() 
                continue
        
            for ii in range(len(fft)-2, 0):
                im = q[ii].red.astype(np.float32)
                mask = im>0
                fft.append(convolver.FFT(im, mask, reverse = flag[0]>0))
                flag[0] *= -1
        
            vy, vx, max_corr = cam.cloud_motion_fft(convolver, fft[-2], fft[-1], ratio = 0.8) 
 
            if vx is None or vy is None:
                q.popleft()
                fft.popleft()
                continue
        
            vy*=flag[0]
            vx*=flag[0]
        
            img.v+=[[vy, vx]]
        
            if PRINT:
                print("at timestep" + str(TimeStamp))
                print("vy = ")
                print(vy)
                print("vx = ")
                print(vx)
                print("max corr = ")
                print(max_corr)

            fft.popleft()
            q.popleft
            
            
          #  if (TimeStamp != '160509'):
          #      continue
        
                                
            args1 = [cameraObjects[camCenterID], [cameraObjects[cmr] for cmr in groups[camCenterID]], Date, TimeStamp, DocumFilename]
            CBH = utility.height_computing(args1)
            if CBH == -1:
                if WRITE:
                    Document = open(DocumFilename, 'a')
                    Document.write('missing JPG in cloud height computing\n')
                    Document.close()                    
                continue
            
#        args1 = [[cameraObjects[camID], [cameraObjects[cmr] for cmr in groups[camID]], Date, TimeStamp] for camID in camIDs4height] 
#        CBHList = p1.map(utility.height_computing, args1)
#        print(CBHList) 

            if WRITE:
                Document = open(DocumFilename, 'a')
                Document.write(str(TimeStamp) + ' vy= ' + str(vy) + ' vx= ' + str(vx) + 'CBH = ' + str(int(CBH)) + '\n')
                Document.close()
            
            
                                        
            args2 = [Date, TimeStamp, CBH, 2*vy, 2*vx, predictForwardList, shape[0], DocumFilename]
            ResultList = utility.stitching(args2)
            STLList = ResultList[0]
            featureVectorList = ResultList[1]
            
            print(STLList)
            for i in range(0, len(STLList)):
               
                print("getting the loop")
                year = Date[:4]
                month = Date[4:6]
                day = Date[6:]
            
                hour = TimeStamp[:2]
                minute = TimeStamp[2:4]
                second = TimeStamp[4:]                
                
                currenttime = datetime.datetime(year = int(year), month = int(month), day = int(day), hour = int(hour) - 5, minute = int(minute), second = int(second))
                timedelta = datetime.timedelta(seconds = 30)
                predicttime = currenttime + predictForwardList[i] * timedelta 
                beforetime = currenttime - 2*timedelta
                if (str(currenttime)[-8:] in GHIs.index and str(beforetime)[-8:] in GHIs.index and str(predicttime)[-8:] in GHIs.index): 
                    GHI1 = GHIs.loc[str(currenttime)[-8:]].values[0]
                    GHI2 = GHIs.loc[str(beforetime)[-8:]].values[0]
                    GHIy = GHIs.loc[str(predicttime)[-8:]].values[0]   
                
                    latitude = GHI_Coor['GHI1'][0]
                    longitude = GHI_Coor['GHI1'][1]
                
                    altitude_deg = ps.get_altitude(latitude, longitude, currenttime)
                    GHI1_CLS = ps.radiation.get_radiation_direct(currenttime, altitude_deg)
                
                    altitude_deg = ps.get_altitude(latitude, longitude, beforetime)
                    GHI2_CLS = ps.radiation.get_radiation_direct(beforetime, altitude_deg)   
                
                    altitude_deg = ps.get_altitude(latitude, longitude, predicttime)
                    GHIy_CLS = ps.radiation.get_radiation_direct(predicttime, altitude_deg)
                 
                else:
                    if PRINT:    
                        print("missing data in GHI for forward" + str(predictForwardList[i]))
                    if WRITE:
                        Document = open(DocumFilename, 'a')
                        Document.write('missing data in GHI' + str(predictForwardList[i]) + '\n')
                        Document.close()                    
                    continue                
                
                STL = STLList[i]

                if STL == -1:
                    if PRINT:
                        print("stithing not valid" + str(predictForwardList[i]))
                    if WRITE:
                        Document = open(DocumFilename, 'a')
                        Document.write('stiching not valid\n' +str(predictForwardList[i]) + '\n')
                        Document.close()
                    continue                   
            
                totalList[i] = totalList[i] + 1
                availableList[i] = availableList[i] + STL

                FeatureVector = featureVectorList[i]
                if STL == 1:
                    FeatureSetList[i]= FeatureSetList[i].append({'TimeStamp': TimeStamp}, ignore_index = True)
                    lastIndex = len(FeatureSetList[i]) - 1            

                    FeatureSetList[i].loc[lastIndex]['Rave1']= FeatureVector[0]
                    FeatureSetList[i].loc[lastIndex]['Rmin1']= FeatureVector[1]
                    FeatureSetList[i].loc[lastIndex]['Rmax1']= FeatureVector[2]
            
                    FeatureSetList[i].loc[lastIndex]['Gave1']= FeatureVector[3]
                    FeatureSetList[i].loc[lastIndex]['Gmin1']= FeatureVector[4]
                    FeatureSetList[i].loc[lastIndex]['Gmax1']= FeatureVector[5]    
    
                    FeatureSetList[i].loc[lastIndex]['Bave1']= FeatureVector[6]
                    FeatureSetList[i].loc[lastIndex]['Bmin1']= FeatureVector[7]
                    FeatureSetList[i].loc[lastIndex]['Bmax1']= FeatureVector[8]    
                    
                    FeatureSetList[i].loc[lastIndex]['Rave2']= FeatureVector[9]
                    FeatureSetList[i].loc[lastIndex]['Rmin2']= FeatureVector[10]
                    FeatureSetList[i].loc[lastIndex]['Rmax2']= FeatureVector[11]
                    
                    FeatureSetList[i].loc[lastIndex]['Gave2']= FeatureVector[12]
                    FeatureSetList[i].loc[lastIndex]['Gmin2']= FeatureVector[13]
                    FeatureSetList[i].loc[lastIndex]['Gmax2']= FeatureVector[14]
    
                    FeatureSetList[i].loc[lastIndex]['Bave2']= FeatureVector[15]
                    FeatureSetList[i].loc[lastIndex]['Bmin2']= FeatureVector[16]
                    FeatureSetList[i].loc[lastIndex]['Bmax2']= FeatureVector[17]      
                    
                    FeatureSetList[i].loc[lastIndex]['RBR1']= FeatureVector[18]      
                    FeatureSetList[i].loc[lastIndex]['RBR2']= FeatureVector[19]      

                    FeatureSetList[i].loc[lastIndex]['CldFraction']= cloudFraction     
            
                    FeatureSetList[i].loc[lastIndex]['GHI1'] = GHI1/GHI1_CLS
                    FeatureSetList[i].loc[lastIndex]['GHI2'] = GHI2/GHI2_CLS
                    FeatureSetList[i].loc[lastIndex]['GHIy'] = GHIy/GHIy_CLS               
                
                
                    outpathFeatureSingle = outpathFeature + 'foward' + str(predictForwardList[i]) + '/'
                    if not os.path.isdir(outpathFeatureSingle):
                        os.makedirs(outpathFeatureSingle)
                        os.chmod(outpathFeature, 0o755)
                
                    FeatureSetList[i].to_csv(outpathFeatureSingle + 'featureSet_' + Date + str(hour) + 'foward' + str(predictForwardList[i]) + '.csv', index = None)
            
                if WRITE:
                    Document = open(DocumFilename, 'a')
                    Document.write("foward" + str(predictForwardList[i]) + '\n')
                    Document.write("total = " + str(totalList[i]) + " available = " + str(availableList[i]) + '\n')
                    Document.close()    
          
    except Exception as e:
        if PRINT:    
            print(e)
        if WRITE:
            Document = open(DocumFilename, 'a')
            Document.write(str(e))
            Document.write(traceback.format_exc())
            Document.close()
                        
    finally:
#        if WRITE:
#            Document = open(DocumFilename, 'a')
#            Document.write("foward" + str(predictForwardList[i]) + '\n')
#            Document.write("total = " + str(totalList[i]) + " available = " + str(availableList[i]) + '\n')
#            Document.close()
        pass
