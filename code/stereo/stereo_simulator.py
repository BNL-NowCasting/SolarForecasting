import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from scipy.ndimage import rotate

deg2rad=np.pi/180
camera='HD2'
rot={'HD1':8*deg2rad, 'HD2':0*deg2rad};
di0_dj0={'HD1':(15,15), 'HD2':(55,5)};

##cx1,cy1=[0,0]
cx2,cy2, cz2=[0,20,0]

h=1000

####set up paths, constantsand initial parameters
inpath='~/data/HD_calib/'
inpath='~/data/HD20_calib/'
inpath='~/data/HD/'

lat,lon=40.88,-72.87
min_scatter_angle = 8
dark_threshold = 25      #### threshold of dark DN value (i.e., shadow band)
var_threshold = 4        #### threshold for cloud spatial variation
rbr_clear = -0.15     ### upper bound of clear sky red/blue index
rbr_cloud = -0.05     ### lower bound of cloud red/blue index
ndilate=19
####dimension of the valid portion of the original image, i.e., the disk with elevation_angle>0
####they need to be tuned for each camera        
ni0,nj0=2823,2823
nr0=(ni0+nj0)/4.0        ##### radius of the valid image 
#### displacement of the true image center, i.e., (ix, iy) of the pixel corresponding to the zenith view
#### need to be tuned for each camera and each time the camera is installed
di0,dj0 = di0_dj0[camera]
#### rotation angle in rad, the angle between north and image north, reckoned clockwisely from north
#### need to be tuned for each camera and each time the camera is installed
rotation=rot[camera]      

ni,nj=2000,2000          #### size of the undistorted image 
max_theta=70*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

#####compute the zenith and azimuth angles for each pixel
j0,i0=np.meshgrid(np.linspace(-ni0//2,ni0//2,ni0),np.linspace(-nj0//2,nj0//2,nj0)); 
r0=np.sqrt(i0**2+j0**2);   
theta0=2*np.arcsin(r0/(np.sqrt(2)*nr0))
phi0 = np.arctan2(j0/nr0,i0/nr0) - rotation  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
phi0=phi0%(2*np.pi)

theta_filter = theta0>max_theta; 
theta0[theta_filter]=np.nan; phi0[theta_filter]=np.nan;

l0=h*np.tan(theta0);
x1,y1=l0*np.cos(phi0), l0*np.sin(phi0)

x2=x1-cx2; y2=y1-cy2; 
theta2=np.arctan(np.sqrt(x2**2+y2**2)/h)
phi2=np.arctan2(y2,x2)%(2*np.pi)

r2=np.sin(theta2/2)*np.sqrt(2)*nr0;
i2,j2=r2*np.cos(phi2),r2*np.sin(phi2)

# #####coordinate system for the undistorted space
r=np.tan(theta0);       
x,y=r*np.cos(phi0), r*np.sin(phi0)
r2=np.tan(theta2);       
x2,y2=r2*np.cos(phi2), r2*np.sin(phi2)

xbin,ybin=np.linspace(-max_tan,max_tan,ni), np.linspace(-max_tan,max_tan,nj)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2
# theta=np.arctan(np.sqrt(xgrid**2+ygrid**2))
# plt.figure(); plt.imshow(j2-j0);

# print(sth)

cnt=0;
# for f in sorted(glob.glob(inpath+camera+'*2018022117*jpg')):
for f in sorted(glob.glob(inpath+camera+'*20180219171940*jpg')):
# for f in sorted(glob.glob(inpath+camera+'*20180214192235*jpg')):
    print(f)
    cnt+=1;          
    
    ######read the image to array
    im0=plt.imread(f).astype(np.float32); 
    im0=st.shift_2d(im0,-di0,-dj0);  im0=im0[:nj0,:ni0];  ####cut the appropriate subset of the original image
    im0[theta_filter,:]=np.nan   
   
    ####perform undistortion
    im=np.zeros((nj,ni,3),dtype=np.float32)
    for ic in range(3):
        im[:,:,ic]=st.bin_average2_reg(im0[:,:,ic],x,y,xbin,ybin,mask=valid);    
        im[:,:,ic]=st.fill_by_mean2(im[:,:,ic],7, mask=(np.isnan(im[:,:,ic])) & valid )

    im0=st.shift_2d(im0,5,0);  
    im2=np.zeros((nj,ni,3),dtype=np.float32)
    for ic in range(3):
        im2[:,:,ic]=st.bin_average2_reg(im0[:,:,ic],x2,y2,xbin,ybin,mask=valid);    
        im2[:,:,ic]=st.fill_by_mean2(im2[:,:,ic],7, mask=(np.isnan(im2[:,:,ic])) & valid )        
    
    plt.figure(); plt.imshow(im/255);  ###undistored image    
    plt.figure(); plt.imshow(im2/255);  ###undistored image   



    
