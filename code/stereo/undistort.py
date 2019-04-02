import numpy as np
import os,glob
from matplotlib import pyplot as plt
import stat_tools as st
from PIL import Image

deg2rad=np.pi/180

camera='HD815_2'
day='20180308'

params = {'HD815_1':[2821.0000,1442.8231,1421.0000,0.1700,-0.0135,-2.4368,0.3465,-0.0026,-0.0038],\
          'HD815_2':[2821.0000,1424,1449.0000,0.0310,-0.0114,-0.9816,0.3462,-0.0038,-0.0030 ],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD17':[2817.249,1478.902,1462.346,-0.099,0.012,0.867,2]}

####set up paths, constantsand initial parameters
inpath='~/data/images/'
outpath='~/data/undistorted/'

lat,lon=40.88,-72.87
min_scatter_angle = 8
dark_threshold = 25      #### threshold of dark DN value (i.e., shadow band)
var_threshold = 4        #### threshold for cloud spatial variation
rbr_clear = -0.15     ### upper bound of clear sky red/blue index
rbr_cloud = -0.05     ### lower bound of cloud red/blue index
ndilate=19
####dimension of the valid portion of the original image, i.e., the disk with elevation_angle>0
####they need to be tuned for each camera           

nx,ny=2001,2001          #### size of the undistorted image 
max_theta=70*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

dest=outpath+camera
if not os.path.isdir(dest):
    os.makedirs(dest)
    os.chmod(dest,0o755)
dest=outpath+camera+'/'+day+'/'
if not os.path.isdir(dest):
    os.makedirs(dest)
    os.chmod(dest,0o755)

xbin,ybin=np.linspace(-max_tan,max_tan,nx), np.linspace(-max_tan,max_tan,ny)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2   
invalid = xgrid**2+ygrid**2 > (max_tan-1e-2)**2 

nx0=ny0=params[camera][0]
nr0=(nx0+ny0)/4
xstart=int(params[camera][2]-nx0/2+0.5); ystart=int(params[camera][1]-ny0/2+0.5)
nx0=int(nx0+0.5); ny0=int(ny0+0.5)
#####compute the zenith and azimuth angles for each pixel
x0,y0=np.meshgrid(np.linspace(-nx0//2,nx0//2,nx0),np.linspace(-ny0//2,ny0//2,ny0)); 
r0=np.sqrt(x0**2+y0**2)/nr0;
#     theta0=2*np.arcsin(r0/np.sqrt(2))

roots=np.zeros(51)
rr=np.arange(51)/100.0
c1,c2,c3=params[camera][6:9]
for i,ref in enumerate(rr):
    roots[i]=np.real(np.roots([c3,0,c2,0,c1,-ref])[-1])
theta0=np.interp(r0/2,rr,roots)
              
phi0 = np.arctan2(x0,y0) - params[camera][3]  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
phi0=phi0%(2*np.pi)

beta,azm=params[camera][4:6]
theta=theta0; phi=phi0
#####correction for the mis-pointing error
k=np.array((np.sin(azm),np.cos(azm),0))
a=np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]); 
a = np.transpose(a,[1,2,0])
b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a,axisb=2) \
  + np.reshape(np.outer(np.dot(a,k),k),(ny0,nx0,3))*(1-np.cos(beta))
theta=np.arctan(np.sqrt(b[:,:,0]**2+b[:,:,1]**2)/b[:,:,2])
phi=np.arctan2(b[:,:,1],b[:,:,0])%(2*np.pi)

theta_filter = (theta>max_theta) | (theta<=0); theta[theta_filter]=np.nan;

#####coordinate system for the undistorted space
r=np.tan(theta); 
x,y=r*np.sin(phi), r*np.cos(phi)        
for f in sorted(glob.glob(inpath+camera+'/'+day+'/'+camera+'_'+day+'*jpg')):  ###8200
    print(f)   
#         ######read the image to array
    im0=plt.imread(f).astype(np.float32);        
    im0=im0[ystart:ystart+ny0,xstart:xstart+nx0,:]
    im0[theta_filter,:]=np.nan           
   
    im=np.zeros((ny,nx,3))
    for i in range(3):
        im[:,:,i]=st.bin_average2_reg(im0[:,:,i],x,y,xbin,ybin,mask=valid);    
        im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, mask=(np.isnan(im[:,:,i])) & valid )  

    ims = Image.fromarray(im.astype(np.uint8))
    ims.save(outpath+camera+'/'+day+'/'+os.path.basename(f)[:-3]+'png', "PNG")

