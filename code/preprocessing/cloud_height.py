import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
import time, mncc
import stat_tools as st
from scipy.ndimage import morphology,filters  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque

MAX_INTERVAL = 180 ####max allowed interval between two frames for cloud motion estimation

camID='HD4B'
camID2='HD4A'
camID3='HD1A'

inpath='~/data/images/' 
outpath='~/data/undistorted/'    

camera=cam.camera(camID,max_theta=70)
camera2=cam.camera(camID2,max_theta=70)

def preprocess(camera,f,q,err,fft,convolver,flag):    
    img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
    img.undistort(rgb=True);  ###undistortion
    if img.red is None:
        return
#     ims = Image.fromarray(img.rgb); ims.save(outpath+camID+'/'+os.path.basename(f), "PNG"); continue
    img.cloud_mask();    ###one-layer cloud masking
    q.append(img)       
    if len(q)<=1: 
        return
    ####len(q) is always 2 beyond this point
    if (q[-1].time-q[-2].time).seconds>=MAX_INTERVAL:
        q.popleft();
        return;
#####cloud motion for the dominant layer    
#     im1=q[-2].red.copy().astype(np.float32); im2=q[-1].red.copy().astype(np.float32); 
#     vy,vx,max_corr = cam.cloud_motion(im1,im2,mask1=im1>5,mask2=np.abs(im1-im2)>15, ratio=0.7, threads=4) 
#     print(camera.camID+', ', f[-18:-4]+',  first layer2:',max_corr,vy,vx) 
    if convolver is None:
        shape=(camera.nx,camera.ny)
        convolver = mncc.Convolver(shape, shape, threads=4, dtype=np.float32)  # 
    for ii in range(len(fft)-2,0):
        im=q[ii].red.astype(np.float32);
        mask = im>0; #im[~mask]=0        
        fft.append(convolver.FFT(im,mask,reverse=flag[0]>0));
        flag[0] *= -1
    vy,vx,max_corr = cam.cloud_motion_fft(convolver,fft[-2],fft[-1],ratio=0.8); 
    if vx is None or vy is None: #####invalid cloud motion
        q.popleft(); fft.popleft();
        return
    vy*=flag[0]; vx*=flag[0]; 
    fft.popleft();
    print(camera.camID+', ', f[-18:-4]+',  first layer:',max_corr,vy,vx) 
#     plt.figure(); plt.imshow(q[-2].red); plt.colorbar(); plt.show();   
    return;

    q[-2].v+=[[vy,vx]]     
    red1=st.shift_2d(q[-1].rgb[:,:,0].astype(np.float32),-vx,-vy); red1[red1<=0]=np.nan
    red2=q[-2].rgb[:,:,0].astype(np.float32); red2[red2<=0]=np.nan #red2-=np.nanmean(red2-q[-1].rgb[:,:,0])
    er=red2-red1;   ###difference image after cloud motion adjustment
    er[(red1==0)|(red2==0)]=np.nan;
    a=er.copy(); a[a>0]=0; er-=st.rolling_mean2(a,500);
    if len(err) <= 0:
        err += [-st.shift_2d(er,vx,vy)] 
        return
    err_2=err[0].copy();  err[0] = (-st.shift_2d(er,vx,vy))
#     if vy**2+vx**2>=50**2:  ######The motion of the dominant layer is fast, likely low clouds. Do NOT trigger the second layer algorithm 
#        return 

#####process the secondary layer 
    ert=er+err_2;    
#     cm2=(er>15) | (er_p>15); cm2=remove_small_objects(cm2, min_size=500, connectivity=4);  
    scale=red2/np.nanmean(red2); nopen=max(5,int(np.sqrt(vx**2+vy**2)/3))
    cm2=(ert>15*scale) & (q[-2].cm); 
    cm2=morphology.binary_opening(cm2,np.ones((nopen,nopen)))
    cm2=remove_small_objects(cm2, min_size=500, connectivity=4);  
#    sec_layer=np.sum(cm2)/len(cm2.ravel())  ###the amount of clouds in secondary layer
    sec_layer=np.sum(cm2)/np.sum(q[-2].cm)  ###the amount of clouds in secondary layer
    if sec_layer<5e-2:   ###too few pixels in second layer, ignore the second cloud layer
        print('Second layer is small:', sec_layer*100, '%')
        return

#####cloud motion for the secondary layer   
    mask2=np.abs(err_2)>5;
    mask2=remove_small_objects(mask2, min_size=500, connectivity=4)
    mask2=filters.maximum_filter(mask2,20) 
    vy,vx,max_corr = cam.cloud_motion(err[0],err_2,mask1=None,mask2=mask2, ratio=None, threads=4) 
    if vx is None or vy is None:
        return
    q[-2].v+=[[vy,vx]]
    print(camera.camID+', ', f[-18:-4]+',  second layer:',max_corr,vy,vx) 
    
    if np.abs(vy-q[-2].v[0][0])+np.abs(vx-q[-2].v[0][1])>10:   
#     if np.abs(vy)+np.abs(vx)>0:
    #####obtain the mask for secondar cloud layer using a watershed-like algorithm    
        mred=q[-2].rgb[:,:,0].astype(np.float32)-st.fill_by_mean2(q[-2].rgb[:,:,0],200,mask=~cm2)
        mrbr=q[-2].rbr-st.fill_by_mean2(q[-2].rbr,200,mask=~cm2)
        merr=st.rolling_mean2(ert,200,ignore=np.nan); var_err=(st.rolling_mean2(ert**2,200,ignore=np.nan)-merr**2)
    #     mk=(np.abs(q[-2].rgb[:,:,0].astype(np.float32)-mred)<3) & ((total_err)>-2) & (np.abs(q[-2].rbr-mrbr)<0.05)
        mk=(np.abs(mred)<3) & (ert>-15) & (np.abs(mrbr)<0.05) & (var_err>20*20)
        cm2=morphology.binary_opening(mk|cm2,np.ones((nopen,nopen)))  ####remove line objects produced by cloud deformation
        cm2=remove_small_objects(cm2, min_size=500, connectivity=4)
        q[-2].layers=2; q[-2].cm[cm2]=2;  #####update the cloud mask with secondary cloud layer 
                
#         fig,ax=plt.subplots(2,2, sharex=True,sharey=True);  ####visualize the cloud masking results
#         ax[0,0].imshow(q[-2].rgb); ax[0,1].imshow(q[-2].cm)
#         ax[1,0].imshow(st.shift_2d(q[-1].rgb,-vx,-vy))  
#         ax[1,1].imshow(er,vmin=-25,vmax=25); plt.show();          
    return;

doy='20180825'
#####get the list of files to be processed
flist = sorted(glob.glob(inpath+camID+'/'+doy[:8]+'/'+'*_'+doy+'155*jpg'))
q=deque(); fft=deque();  ###queues 
q2=deque(); fft2=deque();  ###queues 
err=[]; err2=[];

convolver=None; 
flag=[-1]; flag2=[-1];

t0=time.time()
for i, f in enumerate(flist):
    f2=f.replace(camID,camID2);
    if len(q)>=2 and len(q2)>=2:                
        #####determine cloud height using the previous frame
#         h=cam.cloud_height(q[-2],err[0],q2[-2],err2[0])
#         print('Cameras used: ', q[-2].cam.camID, q2[-2].cam.camID)
#         print(q[-2].time, '. Cloud height and max correlation:',h)
        q.popleft();
        q2.popleft();
    preprocess(camera,f,q,err,fft,convolver,flag)
    preprocess(camera2,f2,q2,err2,fft2,convolver,flag2)
    

print(time.time()-t0) 

