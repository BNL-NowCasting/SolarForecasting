import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps

####set up paths, constants, and initial parameters
inpath='~/data/HD20/'
deg2rad=np.pi/180
lat,lon=40.88,-72.87
min_scatter_angle = 8
dark_threshold = 25      #### threshold of dark DN value (i.e., shadow band)
var_threshold = 4        #### threshold for cloud spatial variation
rbr_clear = -0.15     ### upper bound of clear sky red/blue index
rbr_cloud = -0.05     ### lower bound of cloud red/blue index
ndilate=19
####dimension of the valid portion of the original image, i.e., the disk with elevation_angle>0
####they need to be tuned for each camera
nx0,ny0=2790,2790          
# nx0,ny0=2944,2944
nr0=(nx0+ny0)/4.0        ##### radius of the valid image 
#### displacement of the true image center, i.e., (ix, iy) of the pixel corresponding to the zenith view
#### need to be tuned for each camera and each time the camera is installed
dx0,dy0 = 45, 65 
#### rotation angle in rad, the angle between north and image north, reckoned clockwisely from north
#### need to be tuned for each camera and each time the camera is installed
rotation=8*deg2rad      

nx,ny=1000,1000          #### size of the undistorted image 
max_theta=80*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

#####compute the zenith and azimuth angles for each pixel
x0,y0=np.meshgrid(np.arange(nx0),np.arange(ny0))
r0=np.sqrt((x0-nx0/2)**2+(y0-ny0/2)**2);  
theta=np.pi/2*(r0/nr0); 
phi=rotation+np.arctan2(1-x0/nr0,y0/nr0-1)  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
phi=phi%(2*np.pi)
theta_filter = theta>max_theta; 
theta[theta_filter]=np.nan;

#####coordinate system for the undistorted space
r=np.tan(theta);       
x,y=-r*np.sin(phi), r*np.cos(phi)
xbin,ybin=np.linspace(-max_tan,max_tan,nx), np.linspace(-max_tan,max_tan,ny)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2

cnt=0;
for f in sorted(glob.glob(inpath+'HD*2018*jpg')):
    print(f)
    cnt+=1;
    ####get the image acquisition time, this need to be adjusted whenever the naming convention changes 
    t_cur=datetime.strptime(f[-18:-4],'%Y%m%d%H%M%S');     
    t_std = t_cur-timedelta(hours=5)     #####adjust UTC time into daylight saving time or standard time
    #####solar zenith and azimuth angles. NOTE: the azimuth from pysolar is
    #####reckoned with 0 corresponding to south, increasing counterclockwise              
    sz, saz = 90-ps.get_altitude(lat,lon,t_std), ps.get_azimuth(lat,lon,t_std)
    sz*=deg2rad; saz=(saz%360)*deg2rad;

    ####get the spatial pattern of sky radiance from an empirical sky radiance model           
    cos_g=np.cos(sz)*np.cos(theta)+np.sin(sz)*np.sin(theta)*np.cos(phi-saz); 
    gamma = np.arccos(cos_g);
    denom=(0.91+10*np.exp(-3*sz)+0.45*np.cos(sz)**2)*(1-np.exp(-0.32))
    rad = (0.91+10*np.exp(-3*gamma)+0.45*cos_g**2)*(1-np.exp(-0.32/np.cos(theta)))/denom
    
    ######read the image to array
    im0=plt.imread(f).astype(np.float32); 
    im0=st.shift_2d(im0,-dx0,-dy0);  im0=im0[:ny0,:nx0];  ####cut the appropriate subset of the original image
    im0[theta_filter,:]=np.nan   
#     
#     ####the next two lines are for parameter tuning purpose only, you can comment them out once you finished tuning; 
#     ####if you set the rotation, center, and shifting parameters correctly the black dot will coincide with the sun.
    im0[cos_g>0.997,:]=0;
    fig,ax=plt.subplots(1,1,sharex=True,sharey=True); ax.imshow(im0/255); 
    if cnt>1: break
#     continue
    
    #####sun glare removal
    glare = rad>(np.nanmean(rad)+np.nanstd(rad)*3); 
    im0[glare,:]=0; 
    
    ######cloud masking
    invalid=im0[:,:,2]<dark_threshold
    im0[invalid,:]=np.nan; 
    mn=st.rolling_mean2(im0[:,:,0],ndilate); 
    var=st.rolling_mean2(im0[:,:,0]**2,ndilate)-mn**2; var[var<0]=0; var=np.sqrt(var);
    rbr=(im0[:,:,0]-im0[:,:,2])/(im0[:,:,0]+im0[:,:,2])
    if np.sum(var>var_threshold)>1e3:
        rbr_threshold = max(rbr_clear, min(rbr_cloud,np.nanmean(rbr[var>var_threshold])));
        
    ####cloud mask (cmask) is the 10-pixel dilated region where var>threshold or rbr>threshold 
    coef=np.sqrt(np.minimum(1,np.nanmean(rad)/rad)); 
    cmask=(var*coef>var_threshold) | (rbr>rbr_threshold)
        
    ####perform undistortion
    im=np.zeros((ny,nx,3),dtype=np.float32)
    for i in range(3):
        im[:,:,i]=st.bin_average2_reg(im0[:,:,i],x,y,xbin,ybin,mask=valid);    
        im[:,:,i]=st.fill_by_mean2(im[:,:,i],20, mask=(np.isnan(im[:,:,i])) & valid )
   
    ####visualize the cloud mask, calculated sun position is marked as a blue circle 
    fig,ax=plt.subplots(1,2);
    ax[0].imshow(im0/255); ax[0].axis('off') #####original 
#     ax[1].imshow(cmask); ax[1].axis('off')   ####cloud mask
    ax[1].imshow(im/255); ax[1].axis('off')  ###undistored image
#     ax.imshow(im/255); ax.axis('off')  ###undistored image
#     fig.suptitle(f[-18:]); plt.show(); 
    
    if cnt>1:   ####plot the first five images only, do not want to crash the system
        break


    
