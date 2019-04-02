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
import mncc,multiprocessing

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation
SAVE_FIG=True

# camIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
camIDs=['HD1B', 'HD1C'];
camIDs=['HD1C'];

# days=['20180823124025','20180823124055','20180823124125','20180823124155'] #,'20180824???1','20180829???1'];
days=['20180829165'];

inpath='~/data/images/' 
outpath='~/data/results_parallel/'
tmpfs='/dev/shm/'

def preprocess(args):
    camera,day=args  
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+camera.camID+'/'+ymd+'/'+camera.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return None

    q=deque(); fft=deque();  ###queues 
    flag=[-1];  
    shape=(camera.nx,camera.ny)
    convolver = mncc.Convolver(shape, shape, threads=4, dtype=np.float32)  #      
    for f in flist:
#         print("Start preprocessing ", f[-23:])
        img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
        img.undistort(camera,rgb=True);  ###undistortion
#         print("Undistortion completed ", f[-23:])
        if img.rgb is None:
            continue
        img.cloud_mask(camera);    ###one-layer cloud masking   
#         print("Cloud masking completed ", f[-23:])                   
            
        q.append(img)       
        if len(q)<=1: 
            img.dump_img(tmpfs+f[-23:-4]+'.pkl');
            continue
        ####len(q) is always 2 beyond this point
        if (q[-1].time-q[-2].time).seconds>=MAX_INTERVAL:
            img.dump_img(tmpfs+f[-23:-4]+'.pkl');
            q.popleft();
            continue;
    #####cloud motion for the dominant layer   
        for ii in range(len(fft)-2,0):
            im=q[ii].red.astype(np.float32);
            mask = im>0; #im[~mask]=0        
            fft.append(convolver.FFT(im,mask,reverse=flag[0]>0));
            flag[0] *= -1
        vy,vx,max_corr = cam.cloud_motion_fft(convolver,fft[-2],fft[-1],ratio=0.8); 
        if vx is None or vy is None: #####invalid cloud motion
            img.dump_img(tmpfs+f[-23:-4]+'.pkl');
            q.popleft(); fft.popleft();
            continue
        vy*=flag[0]; vx*=flag[0];
        img.v+=[[vy,vx]]
        img.dump_img(tmpfs+f[-23:-4]+'.pkl');
        print(camera.camID+', ', f[-18:-4]+',  first layer:',len(fft),max_corr,vy,vx)
        
        if SAVE_FIG:
            fig,ax=plt.subplots(2,2,figsize=(12,6),sharex=True,sharey=True);
            ax[0,0].imshow(q[-2].rgb); ax[0,1].imshow(q[-1].rgb); 
            ax[1,0].imshow(q[-2].cm); ax[1,1].imshow(q[-1].cm); 
            plt.tight_layout();
            plt.show();        
        
        fft.popleft();
        q.popleft();             

if __name__ == "__main__":  
    cameras=[];
    for camID in camIDs:
        cameras += [cam.camera(camID,max_theta=70,nx=1000,ny=1000)]

    p = multiprocessing.Pool(len(camIDs)) 
 
    for day in days:
        args=[[camera,day] for camera in cameras]                   
        p.map(preprocess,args)
            
 
            



