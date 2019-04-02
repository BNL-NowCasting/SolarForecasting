import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
import time, sys
import stat_tools as st
from scipy.ndimage import morphology,filters, sobel  ####more efficient than skimage
from scipy import signal
from skimage.morphology import remove_small_objects
from collections import deque

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation

camID='HD1C' if len(sys.argv)<=1 else sys.argv[1];
if len(sys.argv)>=3:
    days=[sys.argv[2]]
else:
    days=['20180821221[1,3,5,7,9]']   ####overcast
    # days=['2018082417?1']  ###clear greyish
    # days=['201808211639']    #####scattered cloud
    # days=['20180830181']   ###clear blue
    # days=['201808211326']  ##### partial cloud
    # days=['201808211351'] 
    days=['2018082314?1']   ####partial cloud
    # days=['201808231241']   ####partial cloud
    days=['20180829165']    #####scattered cloud
    # days=['2018082912?1']    #####clear
    # days=['20180830181']    #####blue sky
    # days=['20180821132[1-2]']  ##### partial cloud
    # days=['20180823120']  ##### partial cloud
    # days=['20180821120']  ####overcast
    
    ##200824-12,18 clear
    ##20180823-14 partial cloudy
    ##200825-1251 high and low clouds 
    ##2018082122 overcast clouds  

inpath='~/data/images/' 
outpath='~/data/results/' 

camera=cam.camera(camID,max_theta=70,nx=1000,ny=1000)
  

for day in days:
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+camera.camID+'/'+ymd+'/'+camera.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        continue

    q=deque();      
    for f in flist:
        print("Start preprocessing ", f[-23:])
        img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
        img.undistort(camera,rgb=True);  ###undistortion

        if img.rgb is None:
            continue
        q.append(img) 

        if len(q)<=1: 
            continue
        ####len(q) is always 2 beyond this point
        if (q[-1].time-q[-2].time).seconds>=MAX_INTERVAL:
            q.popleft(); q.popleft();
            continue;
        
#         r1=q[-2].rgb[...,0].astype(np.float32); r1[r1<=0]=np.nan
#         r2=q[-1].rgb[...,0].astype(np.float32); r2[r2<=0]=np.nan
#         err0 = r2-r1; err0-=np.nanmean(err0)

        cam.cloud_mask(camera,q[-1],q[-2]); ###one-layer cloud masking        
        
        q.popleft(); 
      
# print(time.time()-t0) 

