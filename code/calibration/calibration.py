import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps

deg2rad=np.pi/180
camera='HD2'
rot={'HD1':9.1*deg2rad, 'HD2':1.8*deg2rad};
dxy0={'HD1':(3,30), 'HD2':(55,10)};

####set up paths, constants, and initial parameters
inpath='d:/data/HD20_calib/'
inpath='d:/data/HD/'
lat,lon=40.88,-72.87
dark_threshold = 25      #### threshold of dark DN value (i.e., shadow band)

nx,ny=1500,1500          #### size of the undistorted image 
max_theta=75*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

def cost_sun_match(params,dx0,dy0,nx0,ny0):
    cost=0;
    rotation = params
    dx0=int(dx0); dy0=int(dy0);
    nr0=(nx0+ny0)/4.0        ##### radius of the valid image    
    #####compute the zenith and azimuth angles for each pixel
    x0,y0=np.meshgrid(np.arange(nx0),np.arange(ny0))
    r0=np.sqrt((x0-nx0/2)**2+(y0-ny0/2)**2);  
#     theta=np.pi/2*(r0/nr0); 
    theta=2*np.arcsin(r0/(np.sqrt(2)*nr0))
    phi=rotation+np.arctan2(1-x0/nr0,y0/nr0-1)  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
    phi=phi%(2*np.pi)
    theta_filter = theta>max_theta; 
    theta[theta_filter]=np.nan;

#     for f in sorted(glob.glob(inpath+'HD*2018010317*jpg')):
#     for f in sorted(glob.glob(inpath+'HD1*201802141908*jpg')):
#     for f in sorted(glob.glob(inpath+camera+'*20180214185005*jpg')):   
    for f in sorted(glob.glob(inpath+camera+'*20180219173840*jpg')):        
        ####get the image acquisition time, this need to be adjusted whenever the naming convention changes 
        t_cur=datetime.strptime(f[-18:-4],'%Y%m%d%H%M%S');     
        t_std = t_cur-timedelta(hours=5)     #####adjust UTC time into daylight saving time or standard time
        #####solar zenith and azimuth angles. NOTE: the azimuth from pysolar is
        #####reckoned with 0 corresponding to south, increasing counterclockwise              
        sz, saz = 90-ps.get_altitude(lat,lon,t_std), ps.get_azimuth(lat,lon,t_std)
        sz*=deg2rad; saz=(saz%360)*deg2rad;        
        
        im0=plt.imread(f).astype(np.float32);      
    
        ####get the spatial pattern of sky radiance from an empirical sky radiance model           
        cos_g=np.cos(sz)*np.cos(theta)+np.sin(sz)*np.sin(theta)*np.cos(phi-saz); 
        gamma = np.arccos(cos_g);
        denom=(0.91+10*np.exp(-3*sz)+0.45*np.cos(sz)**2)*(1-np.exp(-0.32))
        rad = (0.91+10*np.exp(-3*gamma)+0.45*cos_g**2)*(1-np.exp(-0.32/np.cos(theta)))/denom
        
        ######read the image to array   
        im0=st.shift_2d(im0,-dx0,-dy0);  im0=im0[:ny0,:nx0];  ####cut the appropriate subset of the original image
        im0[theta_filter,:]=np.nan   
    
#     #####sun glare removal
        glare = rad>(np.nanmean(rad)+np.nanstd(rad)*2.5); 
#         im0[glare,:]=np.nan;
#         plt.figure(); plt.imshow(im0[:,:,0])
        cost += np.nanmean(im0[glare,0])
    print(dx0,dy0,rotation/deg2rad,cost)
    return -cost

from scipy.optimize import fmin
if __name__ == "__main__":
    x0= [5*deg2rad]
    nx0=ny0=2820
    dx0,dy0 = dxy0[camera]
#     xopt = fmin(cost_sun_match, x0, args=(-3,60,nx0,ny0), maxiter=50,xtol=1e-1) ###HD20
#     xopt = fmin(cost_sun_match, x0, args=(3,30,nx0,ny0), maxiter=50,xtol=1e-3) ###HD1, 815
    xopt = fmin(cost_sun_match, x0, args=(dx0,dy0,nx0,ny0), maxiter=50,xtol=1e-3) ###HD2, 815
#         print(xopt)
#         for rat in np.arange(-10,10):
#             cost_sun_match(im0,sz,saz,rotation=rat*deg2rad);
