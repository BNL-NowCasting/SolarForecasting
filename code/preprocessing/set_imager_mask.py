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

camID='HD5A' if len(sys.argv)<=1 else sys.argv[1];

days=[20180829];


###HD3A: 201808221321  


inpath='~/data/images/' 
outpath='~/data/results/' 

camera=cam.camera(camID,max_theta=70)


def preprocess(camera,f):    
    img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
    img.undistort(rgb=True);  ###undistortion
    if img.rgb is None:
        return
#     plt.imshow(img.rbr,vmin=-0.7,vmax=0.2); plt.show();
    
    
    mask=(img.rgb[:,:,2]>0) & ((img.rgb[:,:,2]<76))  ####all other cameras
#     mask=(img.rgb[:,:,2]>0) & ((img.rgb[:,:,2]<80) | ((img.rgb[:,:,1]<img.rgb[:,:,0]-5) & (img.rgb[:,:,1]<img.rgb[:,:,2]-5)))  ####HD5A

    mask=morphology.binary_closing(mask,np.ones((9,9)))
    mask=remove_small_objects(mask, min_size=15, connectivity=4)
    mask=morphology.binary_dilation(mask,np.ones((21,21))) 
    mask=morphology.binary_closing(mask,np.ones((17,17))) 
    mask=remove_small_objects(mask, min_size=1000, connectivity=4)

    fig,ax=plt.subplots(2,2,sharex=True,sharey=True);     
    ax[0,0].imshow(img.rgb); ax[0,1].imshow(img.rbr,vmin=-0.2,vmax=0.1); 
    ax[1,0].imshow(mask); ax[1,1].imshow(img.rgb[:,:,2]) 
#     plt.figure(); plt.hist(img.rbr[img.rbr>-1],bins=100);
    plt.show()

    np.save(camID+'_mask',mask);   

  

for day in days:
    ymd=str(day)
#     flist = sorted(glob.glob(inpath+camID+'/'+ymd+'/'+camID+'_'+ymd+'2121[0-2]?.jpg'))
    flist = sorted(glob.glob(inpath+camID+'/'+ymd+'/'+camID+'_'+ymd+'1341[0-2]?.jpg'))

    for f in flist[:]: 
        print(f[-23:])
        preprocess(camera,f)
        break

      
# print(time.time()-t0) 

