import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from scipy.ndimage.morphology import binary_opening
from scipy import ndimage
import ephem

deg2rad=np.pi/180

camera='HD815_1'
# camera='HD17'
inpath='d:/data/images/'

params = {'HD815_1':[2821.0000,1440.6892,1431.0000,0.1701,0.0084,-0.2048,0.3467,-0.0041,-0.0033],\
          'HD815_2':[2821.0000,1423.9111,1459.000,0.0311,-0.0091,0.1206,0.3455,-0.0035,-0.0032],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD17':[2817.249,1478.902,1462.346,-0.099,0.012,0.867,2]}
rot=params[camera][3]
nx0=ny0=params[camera][0]
nr0=(nx0+ny0)/4
cy,cx=params[camera][1:3]
c1,c2,c3=params[camera][6:9]
x0,y0=np.meshgrid(np.linspace(-nx0//2,nx0//2,nx0),np.linspace(-ny0//2,ny0//2,ny0)); 
r0=np.sqrt(x0**2+y0**2);
nx0=int(nx0+0.5); ny0=int(ny0+0.5)
mask=r0>1320 
xstart=int(params[camera][2]-nx0/2+0.5); ystart=int(params[camera][1]-ny0/2+0.5)

gatech = ephem.Observer(); 
gatech.lat, gatech.lon = '40.88', '-72.87'
sun=ephem.Moon() 

f=sorted(glob.glob(inpath+camera+'/'+camera+'_201802281[6-6]1'+'*jpg'))[0];  
img=plt.imread(f).astype(np.float32); 
fig,ax=plt.subplots(); ax.imshow(img.astype(np.uint8))
# flist=sorted(glob.glob(inpath+camera+'/'+camera+'_201802281[7-9]08[2,3,4]'+'*jpg')); 
flist=sorted(glob.glob(inpath+camera+'/'+camera+'_201802281[4-9]1'+'*jpg'));  
for h in range(24):     
    doy='20180309'+'{:02d}'.format(h)+'1010';
    
    gatech.date = datetime.strptime(doy,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
    sun.compute(gatech) 
    if sun.alt<0: continue    
    sz=np.pi/2-sun.alt; saz=(rot+sun.az-np.pi)%(2*np.pi);   
    

#     rref=c1*sz+c2*sz**3+c3*sz**5
#     xref,yref=cx+nx0*rref*np.sin(saz),cy+ny0*rref*np.cos(saz)
        
    rref=np.sin(sz/2)*np.sqrt(2)*nr0
    xref,yref=cx+rref*np.sin(saz),cy+rref*np.cos(saz)       
    
    circle=plt.Circle((xref,yref),30,color='blue')
    ax.add_artist(circle)





    
