import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
from PIL import Image
import pyfftw

camID='HD4B'

inpath='~/data/images/' 
outpath='~/data/undistorted/'    

camera=cam.camera(camID,max_theta=70)

def preprocess(camera,f, imgs):    
    img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
    img.undistort(rgb=True);  ###undistortion
    if img.red is None:
        return
    imgs +=[img.red[500:1500,500:1500]];

doy='20180825'
#####get the list of files to be processed
flist = sorted(glob.glob(inpath+camID+'/'+doy[:8]+'/'+'*_'+doy+'155[0-2]*jpg'))

imgs=[];
for i, f in enumerate(flist):
    preprocess(camera,f, imgs)
ims=np.array(imgs,dtype=np.float32);
print(ims.shape)
ims[np.isnan(ims)]=0;
np.save('ims',ims);

b = np.abs(pyfftw.interfaces.numpy_fft.fft(ims))
fig,ax=plt.subpots(2,2); 
ax[0,0].imshow(b[0]);
ax[0,1].imshow(b[2]);
ax[1,0].imshow(b[4]);
ax[1,1].imshow(b[...,50]);
plt.show();

