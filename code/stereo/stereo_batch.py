import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from skimage import morphology
import mncc
import time

deg2rad=np.pi/180
cameras=['HD815_2','HD490']; distance=1130
# cameras=['HD815_1','HD490']; distance=1153
# cameras=['HD815_2','HD17']; distance=2260
# cameras=['HD490','HD17']; distance=2688
# cameras=['HD815_2','HD815_1']; distance=23

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
inpath='~/data/images/'
day='20180309'

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

xbin,ybin=np.linspace(-max_tan,max_tan,nx), np.linspace(-max_tan,max_tan,ny)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2   
invalid = xgrid**2+ygrid**2 > (max_tan-1e-2)**2 

x,y,theta_filter={},{},{};
x0,y0={},{};
roi={};
nx0,ny0={},{};
theta,phi={},{};
cnt=0
for camera in cameras:    
    nx0[camera]=ny0[camera]=params[camera][0]
    nr0=(nx0[camera]+ny0[camera])/4
    xstart=int(params[camera][2]-nx0[camera]/2+0.5); ystart=int(params[camera][1]-ny0[camera]/2+0.5)
    nx0[camera]=int(nx0[camera]+0.5); ny0[camera]=int(ny0[camera]+0.5)
    roi[camera]=np.s_[ystart:ystart+ny0[camera],xstart:xstart+nx0[camera]]
    #####compute the zenith and azimuth angles for each pixel
    x0[camera],y0[camera]=np.meshgrid(np.linspace(-nx0[camera]//2,nx0[camera]//2,nx0[camera]),np.linspace(-ny0[camera]//2,ny0[camera]//2,ny0[camera])); 
    r0=np.sqrt(x0[camera]**2+y0[camera]**2)/nr0;
    roots=np.zeros(51)
    rr=np.arange(51)/100.0
    c1,c2,c3=params[camera][6:9]
    for i,ref in enumerate(rr):
        roots[i]=np.real(np.roots([c3,0,c2,0,c1,-ref])[-1])
    theta0=np.interp(r0/2,rr,roots)
                  
    phi0 = np.arctan2(x0[camera],y0[camera]) - params[camera][3]  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
    phi0=phi0%(2*np.pi)
    
    beta,azm=params[camera][4:6]
    
    #####correction for the mis-pointing error
    k=np.array((np.sin(azm),np.cos(azm),0))
    a=np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]); 
    a = np.transpose(a,[1,2,0])
    b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a,axisb=2) \
      + np.reshape(np.outer(np.dot(a,k),k),(ny0[camera],nx0[camera],3))*(1-np.cos(beta))
    theta0=np.arctan(np.sqrt(b[:,:,0]**2+b[:,:,1]**2)/b[:,:,2])
    phi0=np.arctan2(b[:,:,1],b[:,:,0])%(2*np.pi)
    
    theta_filter[camera] = (theta0>max_theta) | (theta0<=0); 
    theta0[theta_filter[camera]]=np.nan;
    
    #####coordinate system for the undistorted space
    r=np.tan(theta0); 
    x[camera],y[camera]=r*np.sin(phi0), r*np.cos(phi0) 
    
    theta[camera]=theta0; phi[camera]=phi0

for f in sorted(glob.glob(inpath+cameras[0]+'/'+day+'/'+cameras[0]+'_'+day+'*jpg')):  ###8200
# for f in sorted(glob.glob(inpath+cameras[0]+'/'+cameras[0]+'*2018030817202*jpg')): 
      #####get the image acquisition time, this need to be adjusted whenever the naming convention changes 
    t_cur=datetime.strptime(f[-18:-4],'%Y%m%d%H%M%S');     
    t_std = t_cur-timedelta(hours=5)     #####adjust UTC time into daylight saving time or standard time
    #####solar zenith and azimuth angles. NOTE: the azimuth from pysolar is
    #####reckoned with 0 corresponding to south, increasing counterclockwise              
    sz, saz = 90-ps.get_altitude(lat,lon,t_std), 360-ps.get_azimuth(lat,lon,t_std)
    sz*=deg2rad; saz=(saz%360)*deg2rad;    
    
    cnt=0
    im=np.zeros((2,ny,nx,3),dtype=np.float32)
    for icam, camera in enumerate(cameras):   
        if icam>=1:
            f=f.replace(cameras[0],camera)
#         ######read the image to array   
        try:    
            im0=plt.imread(f).astype(np.float32);        
        except:
            break
        im0=im0[roi[camera]]
        im0[theta_filter[camera],:]=np.nan   
#         plt.figure(); plt.imshow(im0/255)
         
        cx,cy=params[camera][2:0:-1]
        c1,c2,c3=params[camera][6:9]
        rref=c1*sz+c2*sz**3+c3*sz**5
        xsun,ysun=np.int(cx+nx0[camera]*rref*np.sin(saz+params[camera][3])+0.5),np.int(cy+ny0[camera]*rref*np.cos(saz+params[camera][3])+0.5) ####image coordinates of the sun
        sun_roi=np.s_[max(0,ysun-250):min(ny0[camera],ysun+250),max(0,xsun-250):min(nx0[camera],xsun+250)]
        cos_g=np.cos(sz)*np.cos(theta[camera][sun_roi])+np.sin(sz)*np.sin(theta[camera][sun_roi])*np.cos(phi[camera][sun_roi]-saz); 
        im0[sun_roi][cos_g>0.97]=np.nan
         
#         ###detrend the data
        im0_m=st.rolling_mean2(im0[:,:,0],100,fill=np.nan).astype(np.float32)
        im0[:,:,0]-=im0_m
         
        sun_line=np.abs(x0[camera]*np.cos(saz+params[camera][3] )-y0[camera]*np.sin(saz+params[camera][3] ))<40
        std=np.nanstd(im0[sun_line,0]);
        mk1=(np.abs(im0[:,:,0])>3*std) & sun_line              
        mk2=morphology.remove_small_objects(mk1, min_size=400, connectivity=4)
        im0[mk1!=mk2]=np.nan
                
        for i in range(1):
            im[cnt,:,:,i]=st.bin_average2_reg(im0[:,:,i],x[camera],y[camera],xbin,ybin,mask=valid);    
#             im[cnt,:,:,i]=st.fill_by_mean2(im[cnt,:,:,i],7, mask=(np.isnan(im[cnt,:,:,i])) & valid )  
#         print(time.time()-t0)  
        cnt+=1;  

    if cnt<=1: continue
            
    im[np.isnan(im)]=-255; im[:,invalid,:]=-255;
    corr=mncc.mncc(im[0,:,:,0],im[1,:,:,0],mask1=im[0,:,:,0]>-250,mask2=im[1,:,:,0]>-250)
    deltay,deltax=np.nanargmax(corr)//len(corr)-ny+1,np.nanargmax(corr)%len(corr)-nx+1
    deltar=np.sqrt(deltax**2+deltay**2)
    height=distance/deltar*nx/(2*max_tan)
    
    print(f[-18:-4], np.nanmax(corr),height,deltay, deltax)
    
#    im[im<-250]=np.nan
#    fig,ax=plt.subplots(1,2,sharex=True,sharey=True);  ax[0].imshow(im[0,:,:,0],vmin=-20,vmax=30);  ###undistored image    
#    ax[1].imshow(st.shift_2d(im[1,:,:,0],deltax,deltay),vmin=-20,vmax=30);
 
