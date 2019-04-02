#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 13:09:50 2018

@author: chenxiaoxu
"""

from matplotlib import pyplot as plt
import camera as cam
import geo, os
import numpy as np
from matplotlib import pyplot as plt
#import stat_tools as st
from PIL import Image
#import geopy
from geopy.distance import VincentyDistance
import pysolar.solar as ps
import datetime
import pandas as pd
import geopy

inpath = '~/data/images/'
inpathUndistort = '~/pipeLineOutput/final/undistorted/undistorted/'
outpathStitched = '~/pipeLineOutput/final/Stitched/'

PRINT = True
PLOT = True
WRITE = True

##############################################################################
#global parameters for stitching
deg2rad = np.pi/180
max_theta = 70 * deg2rad     
max_tan = np.tan(max_theta)

Sensor_ind = 'GHI1'

GHI_Coor = {'GHI1':   [40.868972, -72.852225]}

coordinate = {'HD1A':[40.8580088,  -72.8575717],
              'HD1B':[40.8575056,  -72.8547344],
              'HD1C':[40.85785,    -72.8597],
              'HD2B':[40.872341,   -72.874354],
              'HD2C':[40.87203321, -72.87348295],
              'HD3A':[40.897122,   -72.879053], 
              'HD3B':[40.8975,     -72.877497],     
              'HD4A':[40.915708,   -72.892406], 
              'HD4B':[40.917275,   -72.891592],              
              'HD5A':[40.947353,   -72.899617], 
              'HD5B':[40.948044,   -72.898372],
             }
#########
#HD2C is the center camera and it must be put in the first place
camIDs4Stitching = ['HD2C', 'HD1A', 'HD1B', 'HD2B', 'HD3A', 'HD4A']
#camIDs4Stitching = ['HD2C', 'HD2B', 'HD1A', 'HD3A']

#########
camCenterID = 'HD2C'
##############################################################################

def height_computing(args):
    imager, neighbors, Date, TimeStamp, DocumFilename = args
    
    f = inpath + imager.camID + '/' + Date + '/' + imager.camID + '_' + Date + TimeStamp + '.jpg'
       
    if not os.path.exists(f):
        return -1
        
    img = cam.image(imager, f)
    img.undistort(imager, rgb = True)

    if img.red is not None:
        img.cloud_mask(imager)
    else:
#    if img.red is None or img.layers <= 0:
        if WRITE:
            Document = open(DocumFilename, 'a')
            Document.write("img.red is None or img.layers <=0")
            Document.close()            
        return -1
        
    h = [[]]*img.layers
    
    for neighbor in neighbors:    
        f_nb = f.replace(imager.camID, neighbor.camID)

        img_nb = cam.image(neighbor, f_nb)
        img_nb.undistort(neighbor, rgb = True)
        if img_nb.red is not None:
            img_nb.cloud_mask(neighbor)
        else:
            if PRINT:
                print("img_nb.red is None")
            continue

        distance = 6367e3*geo.distance_sphere(img.lat, img.lon, img_nb.lat, img_nb.lon)
        for ih in range(img.layers):
            if (len(h[ih])) >= 1:
                continue
            res = cam.cloud_height(img, img_nb, layer = ih, distance = distance)
       
            if len(res) >= 1 and res[0] < 30*distance and res[0]>0.5*distance:
                h[ih] = res[0]
                if PRINT:
                    print("with camera " + imager.camID + "neighbor " + neighbor.camID)
                    print("the cloud height is")
                    print(res[0])
                return res[0]

                
      #  if len(h) >= img.layers:
      #      if PRINT:
      #          print("with camera " + imager.camID + " neighbor " + neighbor.camID)
      #          print("len(h) = " + str(len(h)))
      #          print("img.layers =" + str(img.layers))
      #      break

    if (len(h[0]) == 0):
        if PRINT:
            print("len(h[0]) == 0")
    return -1
#    return h[0][0]
            
def stitching(args):
    
    Date, TimeStamp, CBH, vy, vx, forwardtimeList, CamImgSize, DocumFilename, = args
    
    if CBH < 900:    
        WindowSize = 100
        DestImg_shp = (8000, 6000 , 3)
    else:
        WindowSize = 150
        DestImg_shp = (12000, 9000 , 3)       
    
    DestImg_shp = (12000,9000,3) 
    #Destination Image
    DestImg = np.zeros(DestImg_shp, dtype = np.uint8)    
    
    for camID in camIDs4Stitching:        
        print(camID)  
        Img_name = inpathUndistort + '/' +  camID + '/' + Date +  '/' + camID + '_' + Date + TimeStamp + '.jpg'    
        
        if not os.path.exists(Img_name):
            if WRITE:
                Document = open(DocumFilename, 'a')
                Document.write(Img_name + '\n' + 'file not found\n')
                Document.close() 
            return [[-1]*len(forwardtimeList), [[]]*len(forwardtimeList)]
        
        CamImg = plt.imread(Img_name)
        CamImg = np.flip(CamImg, 0)
        CamImg = np.flip(CamImg, 1)
       
        CamImgSize = CamImg.shape[0]
        
        if (camID == camCenterID):
            Center = (int(DestImg_shp[0]/3), int(DestImg_shp[1]/3*2))
            CamCenterCoor = (coordinate[camID][0], coordinate[camID][1])
            
            LatiStt = Center[0] - int(CamImgSize/2)
            LatiEnd = Center[0] + int(CamImgSize/2)
            LongStt = Center[1] - int(CamImgSize/2)
            LongEnd = Center[1] + int(CamImgSize/2)
           
            #Camera Covrage in km
            TotalCoverage_km = 2*0.001*CBH*max_tan
            #km for every pixel
            Km1Pixel = TotalCoverage_km/CamImgSize
            GeoPoint = geopy.Point(CamCenterCoor[0], CamCenterCoor[1])
            #Coord for every Pixel
            Coor1Pixel = (VincentyDistance(kilometers = Km1Pixel).destination(GeoPoint, 0).latitude - CamCenterCoor[0],
                          VincentyDistance(kilometers = Km1Pixel).destination(GeoPoint, 90).longitude - CamCenterCoor[1])   
            
            for j in range(0,3):
                DestImg[LatiStt:LatiEnd, LongStt:LongEnd, j] = CamImg[:,:,j]
                
            continue
        
        #Coordinate of Camera
        CamCoor = (coordinate[camID][0], coordinate[camID][1])
        #Difference between this Camera and the Center Camera in Coordinate
        DeltaCoor = (CamCoor[0] - CamCenterCoor[0], CamCoor[1] - CamCenterCoor[1])    
        #Difference between this Camera and the Center Camera in Pixel
        DeltaPixel = (int(DeltaCoor[0]/Coor1Pixel[0]), int(DeltaCoor[1]/Coor1Pixel[1]))
                
        CamCenter = (Center[0] + DeltaPixel[0], Center[1] + DeltaPixel[1])
        
        LatiStt = CamCenter[0] - int(CamImgSize/2)
        LatiEnd = CamCenter[0] + int(CamImgSize/2)
        LongStt = CamCenter[1] - int(CamImgSize/2)
        LongEnd = CamCenter[1] + int(CamImgSize/2)       
        

        for j in range(0, 3):
            DestImg_Ori = DestImg[LatiStt : LatiEnd, LongStt : LongEnd, j].copy()
            DestImg_Ori = DestImg_Ori.reshape(-1)
            
            OriNonzeroInd = np.nonzero(DestImg_Ori)[0]
            
            if (len(OriNonzeroInd) >0):
                CamImgPad = CamImg[:,:,j].copy()
                CamImgPad =  CamImgPad.reshape(-1)
                CamImgPad[OriNonzeroInd] = DestImg_Ori[OriNonzeroInd]
                CamImgPad = np.reshape(CamImgPad, (CamImgSize, CamImgSize))
               
                DestImg[LatiStt:LatiEnd, LongStt:LongEnd, j] = CamImgPad
            else:
                DestImg[LatiStt:LatiEnd, LongStt:LongEnd, j] = CamImg[:,:,j]

            
####################################################################################    
#    for k in range(1, 26):
#        sensor = 'GHI' + str(k)
#        Sen_Coor = (GHI_Coor[sensor][0], GHI_Coor[sensor][1])
#        Sen_Coor = ((Sen_Coor[0] - Mu[0])/Sigma[0], (Sen_Coor[1] - Mu[1])/Sigma[1])
#        Mark_Lati = np.where(DestGrid[0] >= Sen_Coor[0])[0][0]
#        Mark_Long = np.where(DestGrid[1] >= Sen_Coor[1])[0][0]   
#        plt.scatter(Mark_Long, Mark_Lati, s=1, c='yellow', marker='o')
####################################################################################    
    print('a')
    Sensor_Coor = (GHI_Coor[Sensor_ind][0], GHI_Coor[Sensor_ind][1])
    Delta_Sensor_Coor = (Sensor_Coor[0] - CamCenterCoor[0], Sensor_Coor[1] - CamCenterCoor[1])
    Delta_Sensor_Pixel = (int(Delta_Sensor_Coor[0]/Coor1Pixel[0]), int(Delta_Sensor_Coor[1]/Coor1Pixel[1]))

    Sensor_loc = (Center[0] + Delta_Sensor_Pixel[0], Center[1] + Delta_Sensor_Pixel[1])
                
    year = Date[:4]
    month =  Date[4:6]
    day = Date[6:]
    
    hour = TimeStamp[:2]
    minute = TimeStamp[2:4]
    second = TimeStamp[4:]

    time = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), 000000, tzinfo = datetime.timezone.utc)
    sun_altitude = ps.get_altitude(Sensor_Coor[0], Sensor_Coor[1], time)
    sun_azimuth = ps.get_azimuth(Sensor_Coor[0], Sensor_Coor[1], time)
    
    hemi = -1
    
    #hemi  = 0 means west, hemi = 1 means east
    if ((sun_azimuth <= 0) | (sun_azimuth > -90)):
        hemi = 0
    elif (sun_azimuth <-270 | (sun_azimuth > -360)):
        hemi = 1
    else:
        if PRINT:
            print("problem in sun_azimuth\n")
            print(camID)
            print(sun_azimuth)

        if WRITE:
            Document = open(DocumFilename, 'a')
            Document.write('problem in sun_azimuth\n' + camID + str(sun_azimuth))
            Document.close()          
        STL = -1
        return [[STL]*len(forwardtimeList), [[]]*len(forwardtimeList)]            
            
        
    if (hemi == 0):
        sun_azimuth = (-sun_azimuth)*deg2rad
        distance_km = (CBH*0.001) * np.tan((90 - sun_altitude) * deg2rad)
        delta_south_km = distance_km*np.cos(sun_azimuth)
        delta_west_km = distance_km*np.sin(sun_azimuth)
        
        delta_south_pixel = int(delta_south_km / Km1Pixel)
        delta_west_pixel = int(delta_west_km / Km1Pixel)
        
        Sensor_loc_img = (Sensor_loc[0] - delta_south_pixel, Sensor_loc[1] - delta_west_pixel)
                    
    else:
        sun_azimuth = (360 + sun_azimuth) * deg2rad
        distance_km = (CBH*0.001) * np.tan((90 - sun_altitude) * deg2rad)
        delta_south_km = distance_km*np.cos(sun_azimuth)
        delta_east_km = distance_km*np.sin(sun_azimuth)
    
        delta_south_pixel = int(delta_south_km / Km1Pixel)
        delta_east_pixel =  int(delta_east_km / Km1Pixel)    
 
        Sensor_loc_img = (Sensor_loc[0] - delta_south_pixel, Sensor_loc[1] + delta_east_pixel)
        
    FeatureStt1 = (int(Sensor_loc_img[0] - WindowSize/2), int(Sensor_loc_img[1] - WindowSize/2))
    FeatureEnd1 = (int(Sensor_loc_img[0] + WindowSize/2), int(Sensor_loc_img[1] + WindowSize/2))


    if ((FeatureEnd1[0] < 0) | (FeatureStt1[0] > DestImg.shape[0]) |  (FeatureStt1[1] > DestImg.shape[1] | (FeatureEnd1[1] < 0))):
        STL = 0
        return [[STL]*len(forwardtimeList), [[]]*len(forwardtimeList)]
    
    else:
        FeatureImage1 = DestImg[max(0, FeatureStt1[0]) : min(DestImg_shp[0], FeatureEnd1[0]), max(0, FeatureStt1[1]) : min(DestImg_shp[1], FeatureEnd1[1]), :]
    
        ZeroCounter = 0  
        
        FeatureChannel = FeatureImage1[:, :, 0]
        FeatureVector = FeatureChannel.reshape(-1)
        NonZeroInd  = np.nonzero(FeatureVector)[0]
        
        if (len(NonZeroInd) == 0):
            ZeroCounter = ZeroCounter + 1
            Rave1 = 0
            Rmin1 = 0
            Rmax1 = 0
        else:
            Rave1 = np.average(FeatureVector[NonZeroInd])/255
            Rmin1 = np.min(FeatureVector[NonZeroInd])/255
            Rmax1 = np.max(FeatureVector[NonZeroInd])/255
       
        FeatureChannel = FeatureImage1[:, :, 1]
        FeatureVector = FeatureChannel.reshape(-1)
        NonZeroInd  = np.nonzero(FeatureVector)[0]
        
        if (len(NonZeroInd) == 0):
            ZeroCounter = ZeroCounter + 1
            Gave1 = 0
            Gmin1 = 0
            Gmax1 = 0
        else:
            Gave1 = np.average(FeatureVector[NonZeroInd])/255
            Gmin1 = np.min(FeatureVector[NonZeroInd])/255
            Gmax1 = np.max(FeatureVector[NonZeroInd])/255
                
        FeatureChannel = FeatureImage1[:, :, 2]
        FeatureVector = FeatureChannel.reshape(-1)
        NonZeroInd  = np.nonzero(FeatureVector)[0]
        
        if (len(NonZeroInd) == 0):
            ZeroCounter = ZeroCounter + 1
            Bave1 = 0
            Bmin1 = 0
            Bmax1 = 0
        else:
            Bave1 = np.average(FeatureVector[NonZeroInd])/255
            Bmin1 = np.min(FeatureVector[NonZeroInd])/255
            Bmax1 = np.max(FeatureVector[NonZeroInd])/255
            
        if (ZeroCounter == 3):
            STL = 0
            return [[STL]*len(forwardtimeList), [[]]*len(forwardtimeList)]
         
        RBR1 = (Rave1 - Bave1) / (Rave1 + Bave1)
                                    
    STLList = []
    featureVectorList = []
    #Since we flip CamImg before, the vy and vx become the opposite.
    for i in range(0, len(forwardtimeList)):
        
        dy = vy*forwardtimeList[i]
        dx = vx*forwardtimeList[i]
        
        BacktrackPoint = (Sensor_loc_img[0] + dy, Sensor_loc_img[1] + dx)


        FeatureStt2 = (int(BacktrackPoint[0] - WindowSize/2), int(BacktrackPoint[1] - WindowSize/2))
        FeatureEnd2 = (int(BacktrackPoint[0] + WindowSize/2), int(BacktrackPoint[1] + WindowSize/2))
    
        if PLOT and i == len(forwardtimeList) - 1:
            print("plotting")
            print(i)
            outpathfile = outpathStitched + Date + '/predict' + str(forwardtimeList[i]) + '/' 
            if not os.path.isdir(outpathfile):
                os.makedirs(outpathfile)
                os.chmod(outpathfile, 0o755)

            savename = outpathfile + TimeStamp + '.jpg'
            DestImgPlot = DestImg.copy()
            for kk in range(0,3):
                DestImgPlot[FeatureStt2[0]:FeatureEnd2[0], FeatureStt2[1]:FeatureEnd2[1],kk] = 0
                DestImgPlot[FeatureStt1[0]:FeatureEnd1[0], FeatureStt1[1]:FeatureEnd1[1],kk] = 0
        
            DestImgPlot[FeatureStt1[0]:FeatureEnd1[0], FeatureStt1[1]:FeatureEnd1[1],0] = 255
            DestImgPlot[FeatureStt2[0]:FeatureEnd2[0], FeatureStt2[1]:FeatureEnd2[1],1] = 255

            print(FeatureStt2)
            print(FeatureEnd2)


            DestImgPlot = np.flip(DestImgPlot, 0)
            FinalImage = Image.fromarray(DestImgPlot)
            FinalImage.save(savename)
    
        if ((FeatureEnd2[0] < 0) | (FeatureStt2[0] > DestImg.shape[0]) |  (FeatureStt2[1] > DestImg.shape[1] | (FeatureEnd2[1] < 0))):
            STL = 0
            STLList.append(STL)
            featureVectorList.append([])
            continue
    
        else:
            FeatureImage2 = DestImg[max(0, FeatureStt2[0]) : min(DestImg_shp[0], FeatureEnd2[0]), max(0, FeatureStt2[1]) : min(DestImg_shp[1], FeatureEnd2[1]), :]
        
            ZeroCounter = 0  
        
            FeatureChannel = FeatureImage2[:, :, 0]
            FeatureVector = FeatureChannel.reshape(-1)
            NonZeroInd  = np.nonzero(FeatureVector)[0]
        
            if (len(NonZeroInd) == 0):
                ZeroCounter = ZeroCounter + 1
                Rave2 = 0
                Rmin2 = 0
                Rmax2 = 0
            else:
                Rave2 = np.average(FeatureVector[NonZeroInd])/255
                Rmin2 = np.min(FeatureVector[NonZeroInd])/255
                Rmax2 = np.max(FeatureVector[NonZeroInd])/255
       
        
            FeatureChannel = FeatureImage2[:, :, 1]
            FeatureVector = FeatureChannel.reshape(-1)
            NonZeroInd  = np.nonzero(FeatureVector)[0]
        
            if (len(NonZeroInd) == 0):
                ZeroCounter = ZeroCounter + 1
                Gave2 = 0
                Gmin2 = 0
                Gmax2 = 0
            else:
                Gave2 = np.average(FeatureVector[NonZeroInd])/255
                Gmin2 = np.min(FeatureVector[NonZeroInd])/255
                Gmax2 = np.max(FeatureVector[NonZeroInd])/255
                
        
            FeatureChannel = FeatureImage2[:, :, 2]
            FeatureVector = FeatureChannel.reshape(-1)
            NonZeroInd  = np.nonzero(FeatureVector)[0]
        
            if (len(NonZeroInd) == 0):
                ZeroCounter = ZeroCounter + 1
                Bave2 = 0
                Bmin2 = 0
                Bmax2 = 0
            else:
                Bave2 = np.average(FeatureVector[NonZeroInd])/255
                Bmin2 = np.min(FeatureVector[NonZeroInd])/255
                Bmax2 = np.max(FeatureVector[NonZeroInd])/255
            
            if (ZeroCounter == 3):
                STL = 0
                STLList.append(STL)
                featureVectorList.append([])
                continue
        
            RBR2 = (Rave2 - Bave2) / (Rave2 + Bave2)
                        
            STL = 1
            featureVector = [Rave1, Rmin1, Rmax1,
                             Gave1, Gmin1, Gmax1,
                             Bave1, Bmin1, Bmax1,
                             Rave2, Rmin2, Rmax2,
                             Gave2, Gmin2, Gmax2,
                             Bave2, Bmin2, Bmax2,
                             RBR1, RBR2]
            STLList.append(STL)
            featureVectorList.append(featureVector)
        
    return [STLList, featureVectorList]
