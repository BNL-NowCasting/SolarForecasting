import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from scipy import stats
from scipy import ndimage
import mncc

deg2rad=np.pi/180
# cameras=['HD815_2','HD490']; distance=1130
# cameras=['HD815_1','HD490']; distance=1153
# cameras=['HD815_2','HD17']; distance=2260
# cameras=['HD490','HD17']; distance=2688
cameras=['HD815_2','HD815_1']; distance=23

#####params: nx0,cy,cx,rotation,beta,azm
params = {
#           'HD815_1':[2821.0000,1440.6892,1431.0000,0.1701,0.0084,-0.2048,0.3467,-0.0041,-0.0033],\
          'HD815_1':[2821.0000,1443,1421.0000,0.1698,-0.0137,-2.4165,0.3463,-0.0031,-0.0035],\
#           'HD815_2':[2821.0000,1423.9111,1459.000,0.0311,-0.0091,0.1206,0.3455,-0.0035,-0.0032],\
          'HD815_2':[2821.0000,1424,1449.0000,0.0310,0.0122,2.2050,0.3459,-0.0043,-0.0027],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD17':[2817.249,1478.902,1462.346,-0.099,0.012,0.867,2]}
params = {'HD815_1':[2821.0000,1442.8231,1421.0000,0.1700,-0.0135,-2.4368,0.3465,-0.0026,-0.0038],\
          'HD815_2':[2821.0000,1424,1449.0000,0.0310,-0.0114,-0.9816,0.3462,-0.0038,-0.0030 ],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD17':[2817.249,1478.902,1462.346,-0.099,0.012,0.867,2]}

####set up paths, constantsand initial parameters
inpath='d:/data/images/'

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
max_theta=30*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

xbin,ybin=np.linspace(-max_tan,max_tan,nx), np.linspace(-max_tan,max_tan,ny)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2   
invalid = xgrid**2+ygrid**2 > (max_tan-1e-2)**2 

cnt=0
im=np.zeros((2,ny,nx,3),dtype=np.float32)
for icam, camera in enumerate(cameras):    
    nx0=ny0=params[camera][0]
    nr0=(nx0+ny0)/4
    xstart=int(params[camera][2]-nx0/2+0.5); ystart=int(params[camera][1]-ny0/2+0.5)
#     dy0,dx0=int(params[camera][1]-ny0/2+0.5), int(params[camera][2]-nx0/2+0.5)
    nx0=int(nx0+0.5); ny0=int(ny0+0.5)
    #####compute the zenith and azimuth angles for each pixel
    x0,y0=np.meshgrid(np.linspace(-nx0//2,nx0//2,nx0),np.linspace(-ny0//2,ny0//2,ny0)); 
    r0=np.sqrt(x0**2+y0**2)/nr0;
#     theta0=2*np.arcsin(r0/np.sqrt(2))
#     theta0=params[camera][6]*np.arcsin(r0/np.sqrt(params[camera][7]))
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
    for f in sorted(glob.glob(inpath+camera+'*npy')): 

#         ######read the image to array
        im0=np.load(f)
#         im0=plt.imread(f).astype(np.float32);        
        im0=im0[ystart:ystart+ny0,xstart:xstart+nx0,:]
        im0[theta_filter,:]=np.nan   
        
        im0_m=st.rolling_mean2(im0[:,:,0],100,fill=np.nan)
        im0[:,:,0]-=im0_m        

        
        for i in range(1):
            im[cnt,:,:,i]=st.bin_average2_reg(im0[:,:,i],x,y,xbin,ybin,mask=valid);    
            im[cnt,:,:,i]=st.fill_by_mean2(im[cnt,:,:,i],7, mask=(np.isnan(im[cnt,:,:,i])) & valid )  
    cnt+=1;

# fig,ax=plt.subplots(1,2,sharex=True,sharey=True); 
# ax[0].imshow(im[0,:,:,0]);  ###undistored image
# ax[1].imshow(im[1,:,:,0]);  ###undistored image    

im[np.isnan(im)]=-255; im[:,invalid,:]=-255;
corr=mncc.mncc(im[0,:,:,0],im[1,:,:,0],mask1=im[0,:,:,0]>-250,mask2=im[1,:,:,0]>-250)
deltay,deltax=np.nanargmax(corr)//len(corr)-ny+1,np.nanargmax(corr)%len(corr)-nx+1
deltar=np.sqrt(deltax**2+deltay**2)
height=distance/deltar*nx/(2*max_tan)
print(np.nanmax(corr),height,deltay, deltax)

# plt.figure(); plt.imshow(corr)

im[im<-250]=np.nan
fig,ax=plt.subplots(1,2,sharex=True,sharey=True);  ax[0].imshow(im[0,:,:,0],vmin=-20,vmax=30);  ###undistored image    
ax[1].imshow(st.shift_2d(im[1,:,:,0],deltax,deltay),vmin=-20,vmax=30);
# fig,ax=plt.subplots(1,2,sharex=True,sharey=True);  ax[0].imshow(im[0,:,:,0],vmin=20);  ###undistored image    
# ax[1].imshow(st.shift_2d(im[1,:,:,0],deltax,deltay),vmin=20);  ###undistored image   
