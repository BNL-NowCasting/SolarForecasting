#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:52:26 2018

@author: cxxu

"""

import argparse
import math
import datetime
import pandas as pd
import traceback

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import camera as cam

import sys, os, glob
import time, mncc
import stat_tools as st
from scipy.ndimage import morphology, filters  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque

parser = argparse.ArgumentParser(description = 'Preprocessing for HD images of certain camera and certain days')
parser.add_argument('--CamID', type = str, required = True, help = 'Camera ID')
parser.add_argument('--Date', type = str, required = True, help = 'Date')
parser.add_argument('--Dir', type = str, required = True, help = 'Directory Path')
parser.add_argument('--outpath', type = str, required = True, help = 'Output Path')

args = parser.parse_args()

Date = args.Date
CamID = args.CamID
DirPath = args.Dir
outpath = args.outpath


#CamID = 'HD17'
#Date = '20180310'
#DirPath = '~/Dropbox/BNL/Research/Solar/HistPipLine/'

year = Date[:4]
month = Date[4:6]
day = Date[6:]

inpath = DirPath + CamID + '/' + Date + '/' 
#outpath = DirPath + 'output/'

outpath_DataFrame = outpath + 'DataFrame/' + Date + '/'
outpath_ValidTimeStamp = outpath + 'ValidTimeStamp/' + Date + '/' 
outpath_CloudMask = outpath + 'CloudMask/' + Date + '/' + CamID + '/'
outpath_Undistorted = outpath + 'Undistorted/' + Date + '/' + CamID + '/'
outpath_txt = outpath + 'txt/' + Date + '/'
#outpath_error = outpath + 'error/' + Date + '/'

if not os.path.isdir(outpath_DataFrame):
    os.makedirs(outpath_DataFrame)
    os.chmod(outpath_DataFrame, 0o755)

if not os.path.isdir(outpath_ValidTimeStamp):
    os.makedirs(outpath_ValidTimeStamp)
    os.chmod(outpath_ValidTimeStamp, 0o755)

if not os.path.isdir(outpath_CloudMask):
    os.makedirs(outpath_CloudMask)
    os.chmod(outpath_CloudMask, 0o755)

if not os.path.isdir(outpath_Undistorted):
    os.makedirs(outpath_Undistorted)
    os.chmod(outpath_Undistorted, 0o755)

if not os.path.isdir(outpath_txt):
    os.makedirs(outpath_txt)
    os.chmod(outpath_txt, 0o755)
    
#if not os.path.isdir(outpath_error):
#    os.makedirs(outpath_error)
#    os.chmod(outpath_txt, 0o755)    
    
    
TxtFileName = outpath_txt + CamID + '_' + Date + '.txt'
TxtFile = open(TxtFileName, 'w')
TxtFile.write(Date + '  ')
TxtFile.write(CamID + '\n')
TxtFile.write("BEGIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
TxtFile.close()

ResultTable = pd.DataFrame(columns = ['TimeStamp', 'Qlen', 'LayerNum', 'V1', 'V2', 'MaxCorr1', 'MaxCorr2', 'CloudFraction'])
ValidTimeStamp = pd.Series()


TimeRange = range(13, 21)
camera = cam.camera(CamID, max_theta = 70)

################################################################
##begining
################################################################
convolver = None
flag = -1
    
q = deque() #q refers to image sequence 
err = deque() #refers to error image sequence 
fft = deque() # refers to fft object sequence  
Tq = deque()
    
for hour in TimeRange:     
    flist = sorted(glob.glob(inpath + CamID + '_' + Date + str(hour) + '*.jpg'))    
                
    if (len(flist) == 0):
        TxtFile = open(TxtFileName, 'a')
        TxtFile.write("################################\n")
        TxtFile.write(str(hour))
        TxtFile.write("flist is empty! Please check the input path.\n")
        TxtFile.close()
        
        
        ########################################
        #When the flist is empty, the program goes to the next hour's loop,
        #thus we need to re-initialized the following
        convolver = None
        flag = -1
    
        q.clear()  #q refers to image sequence 
        err.clear()  #refers to error image sequence 
        fft.clear()  # refers to fft object sequence  
        Tq.clear()         
        ########################################        
        
    for file in flist:            
        try:
            if len(q) >= 2:    
                try:
                    ImgTimeStamp = Tq[-2]
            
                    ImgRGB = Image.fromarray(q[-2].rgb)
#                   ImgCldMsk = Image.fromarray(q[-2].cm)
            
                    ImgRGBname = outpath_Undistorted + CamID + '_' + Date + ImgTimeStamp + ".jpg"
#                   ImgCldMskname = outpath_CloudMask + CamID + '_' + Date + ImgTimeStamp + ".jpg"
                        
                    ImgRGB.save(ImgRGBname, "JPEG")
#                   ImgCldMsk.save(ImgCldMskname, "JPEG")

#                    ValidTimeStamp = ValidTimeStamp.append(pd.Series(ImgTimeStamp), ignore_index = True)
                    ValidTimeStamp = ValidTimeStamp.append(pd.Series(ImgTimeStamp))
                    ValidTimeStamp.to_csv(outpath_DataFrame + CamID + '_' + Date + '_' + 'Table.csv')                
                        
                    q.popleft()
                    Tq.popleft()
                        
                except Exception as e:
                    TxtFile = open(TxtFileName, 'a')
                    TxtFile.write("#############################\n")
                    TxtFile.write("Error inside (if len(q) > = 2)")
                    TxtFile.write(str(e))
                    TxtFile.write(traceback.format_exc())
#                        TxtFile.write("Program interrupted\n")
                    TxtFile.close()
                        
                finally:
                    pass
                        
            TimeStamp = os.path.basename(file)[-18 : -4]
            TxtFile = open(TxtFileName, 'a')
            TxtFile.write("#############################\n")
            TxtFile.write("Working at time step %s\n" % TimeStamp)
            TxtFile.write("#############################\n")                
            TxtFile.close()
            print("Working at time step %s\n" % TimeStamp)

            ResultTable = ResultTable.append({'TimeStamp' : TimeStamp}, ignore_index = True)
            ResultTable = ResultTable.append({'Qlen' : len(q)}, ignore_index = True)

            img = cam.image(camera, file)  ###img object contains four data fields: rgb, red, rbr, and cm 

            img.undistort(rgb = True) 
            if img.rbr is None:
                TxtFile = open(TxtFileName, 'a')
                TxtFile.write("Image rbr is None. Getting into the next loop\n")
                TxtFile.close()
                q.clear() 
                err.clear()
                fft.clear()
                Tq.clear()
            
                continue

            img.cloud_mask()    
            q.append(img) 
            Tq.append(TimeStamp)
        
            if len(q) <= 1: 
                continue  

                #cloud motion for the dominant layer    
            if convolver is None:
                convolver = mncc.Convolver(q[-2].red.shape, img.red.shape, threads = 4, dtype = img.red.dtype)  # 
        
            for ii in range(len(fft) - 2, 0):
                im= q[ii].red.copy();
                mask = im > -254
                im[~mask] = 0        
                fft.append(convolver.FFT(im, mask, reverse = flag > 0));
                flag = -flag
            
            vy, vx, max_corr = cam.cloud_motion_fft(convolver, fft[-2], fft[-1], ratio = 0.7)
            vy *= flag
            vx *= flag 
            fft.popleft()

            index = len(ResultTable)
            ResultTable.loc[index-1]['LayerNum'] = 1 
            ResultTable.loc[index-1]['V1'] = (vy, vx)                
            ResultTable.loc[index-1]['MaxCorr1'] = max_corr  
                
            ResultTable.to_csv(outpath_DataFrame + CamID + '_' + Date + '_' + 'Table.csv')
                
            #####put the error image into the queue, for use in the multi-layer cloud algorithm
            red1 = st.shift_2d(q[-1].rgb[:,:,0].astype(np.float32), -vx, -vy)
            red1[red1 <= 0] = np.nan
            red2 = q[-2].rgb[:,:,0].astype(np.float32)
            red2[red2<=0] = np.nan #red2-=np.nanmean(red2-q[-1].rgb[:,:,0])
            er = red2 - red1;   ###difference image after cloud motion adjustment
            er[(red1==0)|(red2==0)] = np.nan; 
            a = er.copy()
            a[a>0] = 0
            er -= st.rolling_mean2(a, 500);
            err.append(-st.shift_2d(er, vx, vy))
                    
            if len(err) <= 1: ####secondar layer processing requires three frames
                continue
        
            if vy**2 + vx**2 >= 50**2:  ######The motion of the dominant layer is fast, likely low clouds. Do NOT trigger the second layer algorithm 
                err.popleft(); 
                continue

            #####process the secondary layer 
            ert = er + err[-2]    ####total error  
            scale = red2 / np.nanmean(red2) 
            nopen = max(5, int(np.sqrt(vx**2 + vy**2)/3))  
            cm2 = (ert > 15*scale) & (q[-2].cm); 
            cm2 = morphology.binary_opening(cm2, np.ones((nopen, nopen)))  ####remove line-like structures
            cm2 = remove_small_objects(cm2, min_size = 500, connectivity = 4);    ####remove small objects
            sec_layer = np.sum(cm2)/len(cm2.ravel())  ###the amount of clouds in secondary layer
            if sec_layer < 5e-3:   ###too few pixels, no need to process secondary cloud layer
                err.popleft();        
                continue
            elif sec_layer > 1e-1: ####there are significant amount of secondary layer clouds, we may need to re-run
                pass            ####the cloud motion algorithm for the dominant cloud layer by masking out the secondary layer
            
                #####obtain the mask for secondar cloud layer using a watershed-like algorithm    
            mred = q[-2].rgb[:,:,0].astype(np.float32) - st.fill_by_mean2(q[-2].rgb[:,:,0], 200, mask=~cm2)
            mrbr = q[-2].rbr - st.fill_by_mean2(q[-2].rbr, 200, mask=~cm2)
            merr = st.rolling_mean2(ert, 200, ignore = np.nan)
            var_err = (st.rolling_mean2(ert**2, 200, ignore = np.nan) - merr**2)
            #mk=(np.abs(q[-2].rgb[:,:,0].astype(np.float32)-mred)<3) & ((total_err)>-2) & (np.abs(q[-2].rbr-mrbr)<0.05)
            mk = (np.abs(mred) < 3) & (ert>-15) & (np.abs(mrbr) < 0.05) & (var_err > 20*20)
            cm2 = morphology.binary_opening(mk|cm2, np.ones((nopen, nopen)))  ####remove line objects produced by cloud deformation
            cm2 = remove_small_objects(cm2, min_size = 500, connectivity = 4)
            q[-2].cm[cm2] = 2;  #####update the cloud mask with secondary cloud layer    

                #####cloud motion for the secondary layer   
            mask2 = np.abs(err[-2]) > 5;
            mask2 = remove_small_objects(mask2, min_size = 500, connectivity = 4)
            mask2 = filters.maximum_filter(mask2, 20)   
            vy, vx, max_corr = cam.cloud_motion(err[-1], err[-2], mask1 = None, mask2 = mask2, ratio = None, threads = 4) 
                
            ResultTable.loc[index-1]['LayerNum'] = 2 
            ResultTable.loc[index-1]['V2'] = (vy, vx)        
            ResultTable.loc[index-1]['MaxCorr2'] = max_corr  
                
            ResultTable.to_csv(outpath_DataFrame + CamID + '_' + Date + '_' + 'Table.csv')
                
            err.popleft()
            
        except Exception as e:
            TxtFile = open(TxtFileName, 'a')
            TxtFile.write(TimeStamp)
            TxtFile.write(str(e))
            TxtFile.write(traceback.format_exc())
            TxtFile.write("Getting into the next file loop\n")
            print("####################\n")
            TxtFile.close()
    
    
            print(str(e))
            print("####################")
            print(traceback.format_exc())        
                
        finally:
            pass

TxtFile = open(TxtFileName, 'a')
TxtFile.write("########################################\n")
TxtFile.write("Finished!\n")
TxtFile.close()
