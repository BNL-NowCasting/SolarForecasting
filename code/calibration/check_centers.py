import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from scipy import stats
import mncc

deg2rad=np.pi/180
cameras=['HD815_2','HD490']; distance=1130
# cameras=['HD815_1','HD490']; distance=1153
# cameras=['HD815_2','HD17']; distance=2260
# cameras=['HD490','HD17']; distance=2688
cameras=['HD815_2','HD815_1']; distance=23


####set up paths, constantsand initial parameters
inpath='d:/data/images/'

cnt=0
for icam, camera in enumerate(cameras):        
    for f in sorted(glob.glob(inpath+camera+'*npy')): 
        print(f)
      
#         ######read the image to array
        ts=np.load(camera+'.npy').item()
        im0=np.load(f)
        fig,ax=plt.subplots(); ax.imshow(im0/255)
        for t in ts:
            circle=plt.Circle((ts[t][1],ts[t][0]),2,color='blue')
            ax.add_artist(circle)
    cnt+=1;

