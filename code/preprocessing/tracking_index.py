import numpy as np
import os,sys, glob
from matplotlib import pyplot as plt
import camera as cam
import stat_tools as st
from scipy.ndimage import morphology,filters, sobel  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque
import multiprocessing,subprocess,pickle
import time, geo
from functools import reduce
from operator import concat

SAVE_FIG=True
deg2km=6367*np.pi/180
deg2rad=np.pi/180
WIN_SIZE = 50
INTERVAL = 0.5 ####30 sec

inpath='~/data/images/' 
outpath='~/data/results_parallel/'
# tmpfs='/dev/shm/'
tmpfs='~/ldata/tmp/'
stitch_path='~/ldata/stitch/'

GHI_Coor = {1:   [40.868972, -72.852225],
            2:   [40.868116, -72.851999],
            3:   [40.867261, -72.851758],
            4:   [40.866331, -72.851655],
            5:   [40.865166, -72.851309],
            6:   [40.863690, -72.851217],
            7:   [40.867781, -72.849226],
            8:   [40.866068, -72.849014],
            9:   [40.864829, -72.849031],
            10:   [40.862745, -72.850047],
            11:   [40.858548, -72.846888],
            12:   [40.857791, -72.848877],
            13:   [40.857964, -72.850339],
            14:   [40.859147, -72.852050],
            15:   [40.857746, -72.851804],
            16:   [40.858624, -72.854309],
            17:   [40.857478, -72.854772],
            18:   [40.857970, -72.856379],
            19:   [40.857982, -72.857620],
            20:   [40.857826, -72.859741],
            21:   [40.858323, -72.863776],
            22:   [40.859174, -72.864268],
            23:   [40.859951, -72.864739],
            24:   [40.860966, -72.865434],
            25:   [40.862072, -72.865909]}
GHI_loc=[GHI_Coor[key] for key in sorted(GHI_Coor)]
GHI_loc=np.array(GHI_loc)[:1];

lead_minutes=[1,3,5,10,15,30,45,60]; 
lead_minutes=[1]; 
lead_times=[lt/INTERVAL for lt in lead_minutes]

days=['201809{:02d}15'.format(iday) for iday in range(1,31)]
days=['201809021[3-9]','201809031[3-5]','201809051[5-7]','201809111[7-8]','201809151[4-6]','201809161[6-8]','201809171[4-6]'\
      ,'201809191[6-9]','201809192[0-1]','201809201[5-6]','201809221[4-9]','201810011[3-7]','201810021[7-9]'\
      ,'2018100220','2018100320','201810031[3-9]','201810041[3-6]','201810051[3-5]']

cover_ratio = np.zeros((len(GHI_loc), len(lead_times)));
cnt = 0
for day in days:
    flist = sorted(glob.glob(stitch_path+day[:8]+'/'+day+'*sth'))
    print(day[:8],len(flist))
    for f in flist:
        with open(f,'rb') as input:
            img=pickle.load(input);
            ny,nx = img.cm.shape
    #         print(img.time,img.lon,img.lat,img.sz,img.saz,img.pixel_size,img.v,img.height);
            y, x = (img.lat-GHI_loc[:,0])*deg2km, (GHI_loc[:,1]-img.lon)*deg2km*np.cos(GHI_loc[0,0]*deg2rad);
            iys = (0.5 + (y + img.height*np.tan(img.sz)*np.cos(img.saz))/img.pixel_size).astype(np.int32)
            ixs = (0.5 + (x - img.height*np.tan(img.sz)*np.sin(img.saz))/img.pixel_size).astype(np.int32)
            
            for iGHI in range(len(GHI_loc)):
                iy0, ix0 = iys[iGHI], ixs[iGHI]
                slc=np.s_[max(0,iy0-WIN_SIZE):min(ny-1,iy0+WIN_SIZE),max(0,ix0-WIN_SIZE):min(nx-1,ix0+WIN_SIZE)]
                if img.cm[slc].size>=1:
                    for ilt, lead_time in enumerate(lead_times):
                        iy = int(0.5 + iy0 + img.v[0][0]*lead_time); 
                        ix = int(0.5 + ix0 - img.v[0][1]*lead_time);   #####need to revert vx since the image is flipped
                        slc=np.s_[max(0,iy-WIN_SIZE):min(ny-1,iy+WIN_SIZE),max(0,ix-WIN_SIZE):min(nx-1,ix+WIN_SIZE)]
                        if img.cm[slc].size>=1:
                            cover_ratio[iGHI,ilt] += 1;
          
            cnt += 1;    
#             fig,ax=plt.subplots(1,2,sharex=True,sharey=True);
#             ax[0].imshow(img.rgb); ax[0].scatter(ix,iy,s=3,marker='o');
#             ax[1].imshow(img.cm); 
#             plt.tight_layout(); 
#             plt.show();                     
       
cover_ratio /= (cnt)
print(cover_ratio)
plt.figure(); 
for iGHI in range(len(GHI_loc)):
    plt.plot(lead_minutes,cover_ratio[iGHI],label='GHI'+str(iGHI+1));
plt.xlabel("Forecast lead time, minutes",fontsize=16); plt.ylabel("Tracking index",fontsize=16);
plt.legend(); plt.show();
            
