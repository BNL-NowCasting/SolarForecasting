import numpy as np
import os,sys, glob
from matplotlib import pyplot as plt
import camera as cam
import stat_tools as st
from scipy.ndimage import morphology####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque
import multiprocessing,subprocess,pickle

SAVE_FIG=True
REPROCESS=False  ####reprocess already processed file?
MAX_INTERVAL = 179 ####max allowed interval between two frames for cloud motion estimation
deg2km=6367*np.pi/180

camIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];

days=['201809{:02d}1[3-9]'.format(iday) for iday in range(1,31)]
# days=['201809151[3,7,8,9]','2018091520','2018092213','2018092220'] 

inpath='~/data/images/' 
# tmpfs='/dev/shm/'
tmpfs='~/ldata/tmp/'

cameras={};
for camID in camIDs:
    cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000) 

def mask_motion(args):
    camera,day=args 
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+camera.camID+'/'+ymd+'/'+camera.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return
    
    q=deque();      
    for f in flist:
        print('Processing', f)
        if (~REPROCESS) and os.path.isfile(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.pkl'):
            continue

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

        r1=q[-2].red.astype(np.float32); r1[r1<=0]=np.nan
        r2=q[-1].red.astype(np.float32); r2[r2<=0]=np.nan
        err0 = r2-r1;
       
        dif=np.abs(err0); 
        dif=st.rolling_mean2(dif,20)
        semi_static=(abs(dif)<10) & (r1-127>100)
        semi_static=morphology.binary_closing(semi_static,np.ones((10,10)))
        semi_static=remove_small_objects(semi_static,min_size=200, in_place=True)
        q[-1].rgb[semi_static]=0;
        r2[semi_static]=np.nan

        cam.cloud_mask(camera,q[-1],q[-2]); ###one-layer cloud masking        
        if (q[-1].cm is None):
            q.popleft();             
            continue
        if (np.sum((q[-1].cm>0))<2e-2*img.nx*img.ny):   ######cloud free case
            q[-1].layers=0; 
        else:               
            dilated_cm=morphology.binary_dilation(q[-1].cm,np.ones((15,15))); dilated_cm &= (r2>0)
            vy,vx,max_corr = cam.cloud_motion(r1,r2,mask1=r1>0,mask2=dilated_cm, ratio=0.7, threads=4);
            if np.isnan(vy):  
                q[-1].layers=0;
            else:
                q[-1].v += [[vy,vx]]; q[-1].layers=1;        
       
        q[-1].dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.pkl');
        q.popleft();             

if __name__ == "__main__":  
    p = multiprocessing.Pool(len(camIDs)) 
    for day in days:
        if not os.path.isdir(tmpfs+day[:8]):
            try:
                subprocess.call(['mkdir', tmpfs+day[:8]]) 
            except:
                print('Cannot create directory,',tmpfs+day[:8])
                continue  
        args=[[cameras[camID], day] for camID in camIDs]                   
        p.map(mask_motion,args)  
            
            



