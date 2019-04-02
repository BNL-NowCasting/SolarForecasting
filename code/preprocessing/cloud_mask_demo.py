import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
import time, sys
import stat_tools as st
from scipy.ndimage import morphology,filters, measurements  ####more efficient than skimage
from scipy import signal
from skimage.morphology import remove_small_objects
# from skimage import measure
from collections import deque
import pickle,multiprocessing
import subprocess
# import cv2,of_dis

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation
SAVE_FIG=False

cameraIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
ixy={'HD5A':[0,0],'HD5B':[1,0],'HD4A':[2,0],'HD4B':[3,0],'HD3A':[4,0],'HD3B':[0,2], \
     'HD2B':[1,2],'HD2C':[2,2],'HD1B':[3,2],'HD1C':[4,2]}
# cameraIDs=['HD2C', 'HD2B', 'HD3B'];
# cameraIDs=['HD2B'];

if len(sys.argv)>=2:
    days=[sys.argv[1]]
else:
    days=['2018082312[0,2,4]','20180825161','20180829165','2018082112[0,1]','20180830181','20180824171','20180821132'] 
    # days=['20180825161']; ####multilayer cloud
#     days=['20180829165']    #####scattered cloud
    # days=['20180821121']    #####thin, overcast cloud
#     days=['20180821120']    #####overcast cloud
#     days=['20180830181']    #####blue sky
    # days=['20180824171']   ###gray sky
#     days=['20180821132']  ##### partial cloud
    
    days=['20180829162'];
    # days=['20180825161','20180823124','20180829165','20180821132','20180830181','20180824171'];
    days=['20181001141']    #####scattered cloud
    days=['20180911151']    #####scattered cloud

inpath='~/data/images/' 
outpath='~/ldata/results/cm/' 
# tmpfs='/dev/shm/'
tmpfs='~/ldata/tmp/'

def visualize(camIDs, dates):
    for day in dates:
        flist = sorted(glob.glob(inpath+camIDs[0]+'/'+day[:8]+'/'+camIDs[0]+'_'+day+'*.jpg')) 
        if len(flist)<=1:
            continue
        for f in flist[1:]:
            ymdhms=f[-18:-4]
            print(ymdhms)
            counter=0;
            for counter in range(10):
                try:
                    pkls=sorted(glob.glob(tmpfs+day[:8]+'/HD*_'+ymdhms+'.pkl'));
                except:
                    time.sleep(5);
                    continue
                if len(pkls)>=max(1,len(camIDs)-2):
                    break;
                time.sleep(5);
                
            fig,ax=plt.subplots(4,5,figsize=(12,10),sharex=True,sharey=True);
            for pkl in pkls:
                camID=pkl[-23:-19]
                if camID not in camIDs:
                    continue;
                ix, iy=ixy[camID]
                ax[iy,ix].set_title(camID);
                ax[iy,ix].set_xticks([]); ax[iy,ix].set_yticks([])
                ax[iy+1,ix].set_xticks([]); ax[iy+1,ix].set_yticks([])                
                img=None
                with open(pkl,'rb') as input:
                    img=pickle.load(input);
                if img is None:
                    continue
                ax[iy,ix].imshow(img.rgb);
                ax[iy+1,ix].imshow(img.cm,vmax=2);

            plt.tight_layout();
            plt.show();
#             fig.savefig(outpath+ymdhms); plt.close(); 

def motion(args):
    camera,day=args  
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+camera.camID+'/'+ymd+'/'+camera.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return None
    
    q=deque();
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
        q[-1].dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.msk');
        q.popleft(); 
        
if __name__ == "__main__":  
    cameras=[];
    for camID in cameraIDs:
        cameras += [cam.camera(camID,max_theta=70,nx=1000,ny=1000)]

    p0=multiprocessing.Process(target=visualize, args=(cameraIDs, days,))
    p0.start(); 
    
#     p = multiprocessing.Pool(len(cameraIDs))      
#     for day in days:
#         if not os.path.isdir(tmpfs+day[:8]):
#             try:
#                 subprocess.call(['mkdir', tmpfs+day[:8]]) 
#             except:
#                 print('Cannot create directory,',tmpfs+day[:8])
#                 continue       
#         args=[[camera,day] for camera in cameras]   
#                         
#         p.map(motion,args)
    p0.join();
