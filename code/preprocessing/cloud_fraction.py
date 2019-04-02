import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
import time, sys
import stat_tools as st
from scipy.ndimage import morphology,filters, measurements  ####more efficient than skimage
from scipy import signal
from skimage.morphology import remove_small_objects
from collections import deque
import pickle,multiprocessing
import subprocess
from threading import Event, Thread

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation
SAVE_FIG=False

cameraIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
# cameraIDs=['HD2C', 'HD2B', 'HD3B'];
# cameraIDs=['HD2B'];

inpath='~/data/images/' 
outpath='~/ldata/results/cm/' 
# tmpfs='/dev/shm/'
tmpfs='~/ldata/tmp/'


def call_repeatedly(intv, func, *args):
    stopped = Event()
    def loop():
        while not stopped.wait(intv): # the first call is in `intv` secs
            func(*args)
    Thread(target=loop).start()    
    return stopped.set

def motion(args):
    camera,day=args  
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+camera.camID+'/'+ymd+'/'+camera.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return None
    
    q=deque();
    fh=open(outpath+camera.camID+'/'+camera.camID+'.'+ymd+'.txt','w');
    for f in flist:
#         print("Start preprocessing ", f[-23:])
        img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
        img.undistort(camera,rgb=True);  ###undistortion
#         print("Undistortion completed ", f[-23:])
        if img.rgb is None:
            continue
        q.append(img)  
        if len(q)<=1: 
            continue
        
        ####len(q) is always 2 beyond this point
        if (q[-1].time-q[-2].time).seconds>=MAX_INTERVAL:
            q.popleft(); q.popleft();
            continue;
        
        r1=q[-2].rgb[...,0].astype(np.float32); r1[r1<=0]=np.nan
        r2=q[-1].rgb[...,0].astype(np.float32); r2[r2<=0]=np.nan
        err0 = r2-r1; err0-=np.nanmean(err0)

        cam.cloud_mask(camera,q[-1],q[-2]); ###one-layer cloud masking   
        fh.write(f[-10:-4],', ',np.sum(q[-1].cm));
#         q[-1].dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.msk');
        q.popleft(); 
        
if __name__ == "__main__":  
    cameras=[];
    for camID in cameraIDs:
        cameras += [cam.camera(camID,max_theta=70,nx=1000,ny=1000)]
    
    p = multiprocessing.Pool(len(cameraIDs))      
    for day in days:
        if not os.path.isdir(tmpfs+day[:8]):
            try:
                subprocess.call(['mkdir', tmpfs+day[:8]]) 
            except:
                print('Cannot create directory,',tmpfs+day[:8])
                continue       
        args=[[camera,day] for camera in cameras]   
                        
        p.map(motion,args)
