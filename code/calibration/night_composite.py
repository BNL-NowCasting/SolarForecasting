import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
import stat_tools as st
from PIL import ImageChops
from PIL import Image

camera='HD815_1'
# camera='HD490'
# camera='HD20'
inpath='d:/data/images/'

flist=[]
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_20180309???5'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802272[1-3][1,4]3[2,3]'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802270[0-9][2,5]8[2,3,4]'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802280[0-9][1,4]029'+'*jpg')) 

# finalimage=plt.imread(flist[0])
# for f in flist[1:]:      
#     currentimage=plt.imread(f)
#     finalimage=np.maximum(finalimage, currentimage)

# finalimage=Image.open(flist[0])
# for f in flist[1:]:      
#     currentimage=Image.open(f)
# #     1720,2600
#     finalimage=ImageChops.lighter(finalimage, currentimage)
# plt.figure(frameon=False); plt.imshow(finalimage);

win=1    
finalimage=np.zeros((2944,2944,3)).astype(np.float32)
im1=np.zeros((2944,2944,3)).astype(np.float32); 
for i,f in enumerate(flist):
    tmp = plt.imread(f)
    if tmp[1831,2636,0]>210:
        continue
    im1 += tmp
    if i%win>=win-1:
        im1 /= win
        finalimage=np.maximum(finalimage,im1)
        im1-=im1    
 
plt.figure(frameon=False); plt.imshow(finalimage.astype(np.uint8));
np.save(inpath+camera+'_moon',finalimage);


    
