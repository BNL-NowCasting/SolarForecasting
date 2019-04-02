import numpy as np
import sys,os,glob
from matplotlib import pyplot as plt
import camera as cam
import time, mncc
import stat_tools as st
from scipy.ndimage import morphology,filters  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque

camID='HD815_1'
days=['20180302','20180304','20180305','20180306','20180307','20180312','20180316','20180320','20180324','20180328','20180401','20180403','20180405','20180409','20180411','10180415']

inpath='~/data/images/'     
outpath='~/data/cloud_mask/'     

camera=cam.camera(camID,max_theta=70)

for day in days:
####set up the output path as well as appropriate permissions
    dest=outpath+camID
    if not os.path.isdir(dest):
        os.makedirs(dest)
        os.chmod(dest,0o755)
    dest=outpath+camID+'/'+day+'/'
    if not os.path.isdir(dest):
        os.makedirs(dest)
        os.chmod(dest,0o755)

#####get the list of files to be processed
    flist = sorted(glob.glob(inpath+camID+'/'+day+'/'+camID+'*'+day+'*jpg'))

    convolver=None; flag=-1;
    v1=[]; v2=[];  #####the cloud motion vector for the dominant and secondary layers
###queues. q: img sequence; err: error image sequence; fft: fft object sequence    
    q=deque(); err=deque(); fft=deque();  
    for i, f in enumerate(flist):
####visualize the cloud mask     
        if len(q)>=2:    
            fig,ax=plt.subplots(1,2,sharex=True,sharey=True);
            ax[0].imshow(q[-2].rgb); ax[0].axis('off') #####original 
            ax[1].imshow(q[-2].cm); ax[1].axis('off')   ####cloud mask    
            fig.savefig(dest+os.path.basename(f))
            plt.close(fig);
            q.popleft();

        img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
        img.undistort(rgb=True);  ###undistortion
        if img.rbr is None:
            q.clear(); err.clear(); fft.clear();
            continue
#     ims = Image.fromarray(img.rgb); ims.save(outpath+camID+'/'+os.path.basename(f), "PNG"); continue
        img.cloud_mask();    ###cloud masking
        q.append(img)       
        if len(q)<=1: continue  

#####cloud motion for the dominant layer    
        if convolver is None:
            convolver = mncc.Convolver(q[-2].red.shape, img.red.shape, threads=4, dtype=img.red.dtype)  # 
        for ii in range(len(fft)-2,0):
            im=q[ii].red.copy();
            mask = im>-254; im[~mask]=0        
            fft.append(convolver.FFT(im,mask,reverse=flag>0));
            flag = -flag
        vy,vx,max_corr = cam.cloud_motion_fft(convolver,fft[-2],fft[-1],ratio=0.7); vy*=flag; vx*=flag; 
        fft.popleft();
        print(f[-18:-4]+',  First layer:',max_corr,vy,vx) 
        
#####put the error image into the queue, for use in the multi-layer cloud algorithm
        v1+=[[vy,vx]]     
        red1=st.shift_2d(q[-1].rgb[:,:,0].astype(np.float32),-vx,-vy); red1[red1<=0]=np.nan
        red2=q[-2].rgb[:,:,0].astype(np.float32); red2[red2<=0]=np.nan #red2-=np.nanmean(red2-q[-1].rgb[:,:,0])
        er=red2-red1;   ###difference image after cloud motion adjustment
        er[(red1==0)|(red2==0)]=np.nan; 
        a=er.copy(); a[a>0]=0; er-=st.rolling_mean2(a,500);
        err.append(-st.shift_2d(er,vx,vy))
        if len(err)<=1: ####secondar layer processing requires three frames
            continue
        if vy**2+vx**2>=50**2:  ######The motion of the dominant layer is fast, likely low clouds. Do NOT trigger the second layer algorithm 
            v2+=[[np.nan,np.nan]]
            err.popleft(); 
            continue

#####process the secondary layer 
        ert=er+err[-2];    ####total error  
        scale=red2/np.nanmean(red2); nopen=max(5,int(np.sqrt(vx**2+vy**2)/3))  
        cm2=(ert>15*scale) & (q[-2].cm); 
        cm2=morphology.binary_opening(cm2,np.ones((nopen,nopen)))  ####remove line-like structures
        cm2=remove_small_objects(cm2, min_size=500, connectivity=4);    ####remove small objects
        sec_layer=np.sum(cm2)/len(cm2.ravel())  ###the amount of clouds in secondary layer
        if sec_layer<5e-3:   ###too few pixels, no need to process secondary cloud layer
            v2+=[[np.nan,np.nan]]
            err.popleft();        
            continue
        elif sec_layer>1e-1: ####there are significant amount of secondary layer clouds, we may need to re-run
            pass;            ####the cloud motion algorithm for the dominant cloud layer by masking out the secondary layer
            
#####obtain the mask for secondar cloud layer using a watershed-like algorithm    
        mred = q[-2].rgb[:,:,0].astype(np.float32) - st.fill_by_mean2(q[-2].rgb[:,:,0],200,mask=~cm2)
        mrbr = q[-2].rbr - st.fill_by_mean2(q[-2].rbr,200,mask=~cm2)
        merr = st.rolling_mean2(ert,200,ignore=np.nan); var_err=(st.rolling_mean2(ert**2,200,ignore=np.nan)-merr**2)
#     mk=(np.abs(q[-2].rgb[:,:,0].astype(np.float32)-mred)<3) & ((total_err)>-2) & (np.abs(q[-2].rbr-mrbr)<0.05)
        mk=(np.abs(mred)<3) & (ert>-15) & (np.abs(mrbr)<0.05) & (var_err>20*20)
        cm2=morphology.binary_opening(mk|cm2,np.ones((nopen,nopen)))  ####remove line objects produced by cloud deformation
        cm2=remove_small_objects(cm2, min_size=500, connectivity=4)
        q[-2].cm[cm2]=2;  #####update the cloud mask with secondary cloud layer    

#####cloud motion for the secondary layer   
        mask2=np.abs(err[-2])>5;
        mask2=remove_small_objects(mask2, min_size=500, connectivity=4)
        mask2=filters.maximum_filter(mask2,20)   
        vy,vx,max_corr = cam.cloud_motion(err[-1],err[-2],mask1=None,mask2=mask2, ratio=None, threads=4) 
        v2+=[[vy,vx]]
        print(f[-18:-4]+',  second layer:',max_corr,vy,vx) 

        err.popleft(); 
