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

ixy={'HD5A':[0,0],'HD5B':[1,0],'HD4A':[0,1],'HD4B':[1,1],'HD3A':[0,2],'HD3B':[1,2], \
     'HD2B':[0,3],'HD2C':[1,3],'HD1B':[0,4],'HD1C':[1,4]}

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
            try:
                pkls = sorted(glob.glob(inpath+'*/'+day[:8]+'/HD*'+ymdhms+'*jpg')) 
            except:
                time.sleep(5);
                
            fig,ax=plt.subplots(5,2,figsize=(5,12),sharex=True,sharey=True);
            for pkl in pkls:
                camID=pkl[-23:-19]
                print(camID)
                if camID not in camIDs:
                    continue;
                ix, iy=ixy[camID]
                ax[iy,ix].set_title(camID);
                ax[iy,ix].set_xticks([]); ax[iy,ix].set_yticks([])
                with open(pkl,'rb') as input:
                    img=plt.imread(pkl);
                    ax[iy,ix].imshow(img);

            plt.tight_layout();
            plt.show();
#             fig.savefig(outpath+ymdhms); plt.close(); 

        
if __name__ == "__main__":  
    p0=multiprocessing.Process(target=visualize, args=(cameraIDs, days,))
    p0.start(); 
    
    p0.join();
