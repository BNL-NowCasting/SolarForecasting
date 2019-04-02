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
# import cv2,of_dis

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation
SAVE_FIG=True

cameraIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
ixy={'HD5A':[0,0],'HD5B':[1,0],'HD4A':[2,0],'HD4B':[3,0],'HD3A':[4,0],'HD3B':[0,2], \
     'HD2B':[1,2],'HD2C':[2,2],'HD1B':[3,2],'HD1C':[4,2]}
# cameraIDs=['HD1C', 'HD2B'];

# days=['20180823124025','20180823124055','20180823124125','20180823124155'] #,'20180824???1','20180829???1'];
days=['20180823124','20180829165'];
days=['20180825161']; ####multilayer cloud
# days=['20180821121']    #####hin, overcast cloud
# days=['20180821120']    #####overcast cloud
# days=['20180830181']    #####blue sky
# days=['20180824171']   ###gray sky

inpath='~/data/images/' 
outpath='~/ldata/results_parallel/' 
tmpfs='/dev/shm/'

def visualize(camIDs, dates):
    for day in dates:
        if not os.path.isdir(tmpfs+day[:8]):
            os.makedirs(tmpfs+day[:8])
        flist = sorted(glob.glob(inpath+camIDs[0]+'/'+day[:8]+'/'+camIDs[0]+'_'+day+'*.jpg')) 
        if len(flist)<=1:
            continue
        for f in flist[1:]:
            ymdhms=f[-18:-4]
            print(ymdhms)
            counter=0;
            for counter in range(8):
                pkls=sorted(glob.glob(tmpfs+day[:8]+'/HD*_'+ymdhms+'.pkl'));
                if len(pkls)>=max(1,len(camIDs)-3):
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
                ax[iy+1,ix].set_title(str(img.v));
                ax[iy,ix].imshow(img.rgb);
                ax[iy+1,ix].imshow(img.cm,vmax=2);

            plt.tight_layout();
            plt.show();
#            fig.savefig(outpath+ymdhms); 
	    plt.close(); 

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
#         gray_img = cv2.cvtColor(img.rgb, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(tmpfs+f[-23:],gray_img);

        if len(q)<=1: 
            continue
        ####len(q) is always 2 beyond this point
        if (q[-1].time-q[-2].time).seconds>=MAX_INTERVAL:
            q.popleft(); q.popleft();
            continue;
        
        cam.cloud_mask(camera,q[-1],q[-2]); ###one-layer cloud masking        
        
        r1=q[-2].red.astype(np.float32); r1[r1<=0]=np.nan
        r2=q[-1].red.astype(np.float32); r2[r2<=0]=np.nan
        err0 = r2-r1;
        
        dilated_cm=morphology.binary_dilation(q[-1].cm,np.ones((15,15))); dilated_cm &= (r2>0)
        vy,vx,max_corr = cam.cloud_motion(r1,r2,mask1=r1>0,mask2=dilated_cm, ratio=0.7, threads=4);
        if vy is None:
            q.popleft(); 
            continue
        q[-1].v += [[vy,vx]]; q[-1].layers+=1;        
        
        err = r2-st.shift2(r1,-vx,-vy);  err[(r2==0) | (st.shift2(r1,-vx,-vy)==0)]=np.nan; 
        print(f[-23:],', first layer:',vy,vx,max_corr)

        mask2=st.rolling_mean2(np.abs(err)-np.abs(err0),40)<0
#         mask2 = (np.abs(err)-np.abs(err0)<0) #| (np.abs(err)<1);
        mask2=remove_small_objects(mask2,min_size=900, in_place=True)
        mask2=morphology.binary_dilation(mask2,np.ones((15,15)))
        mask2 = (~mask2) & (r2>0) & (np.abs(r2-127)<30) & (np.abs(err)>=1) #& (q[-1].cm>0)
#         print(np.sum(mask2 & (q[-1].cm>0))/(img.nx*img.ny))
        if np.sum(mask2 & (q[-1].cm>0))>2e-2*img.nx*img.ny:
            vy,vx,max_corr = cam.cloud_motion(r1,r2,mask1=r1>0,mask2=mask2, ratio=0.7, threads=4);
            if vy is None:
                q.popleft(); 
                continue
            vdist = np.sqrt((vy-q[-1].v[-1][0])**2+(vx-q[-1].v[-1][1])**2)
            if vdist>=5 and np.abs(vy)+np.abs(vx)>2.5 and vdist>0.3*np.sqrt(q[-1].v[-1][0]**2+q[-1].v[-1][1]**2):
                score1=np.nanmean(np.abs(err[mask2])); 
                err2=r2-st.shift2(r1,-vx,-vy); err2[(r2==0) | (st.shift2(r1,-vx,-vy)==0)]=np.nan;  
                score2=np.nanmean(np.abs(err2[mask2]));
                print("Score 1 and score 2: ", score1,score2);
                if score2<score1:
                    q[-1].v += [[vy,vx]]; q[-1].layers=2;
                    dif=st.rolling_mean2(np.abs(err)-np.abs(err2),40)>0
                    dif=remove_small_objects(dif,min_size=300, in_place=True)
                    q[-1].cm[dif & (q[-1].cm>0)]=q[-1].layers; 
                    print(f[-23:],', second layer:',vy,vx,max_corr)
       
        q[-1].dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.pkl');

        if SAVE_FIG:
            fig,ax=plt.subplots(2,2,figsize=(9,9),sharex=True,sharey=True);
            ax[0,0].imshow(mask2); ax[0,1].imshow(q[-1].red); 
            ax[1,0].imshow(q[-1].cm); ax[1,1].imshow(q[-1].rgb);
            plt.tight_layout(); 
	    plt.show();     
           # fig.savefig(outpath+ymdhms); 
        q.popleft();             

if __name__ == "__main__":  
    cameras=[];
    for camID in cameraIDs:
        cameras += [cam.camera(camID,max_theta=70,nx=1000,ny=1000)]

    p0=multiprocessing.Process(target=visualize, args=(cameraIDs, days,))
    p0.start(); 
    
    p = multiprocessing.Pool(len(cameraIDs)) 
 
    for day in days:
        args=[[camera,day] for camera in cameras]                   
        p.map(motion,args)
    p0.join();
