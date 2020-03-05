import numpy as np
from matplotlib import pyplot as plt
import tools.stat_tools as st
from datetime import datetime,timedelta,timezone
import os,ephem
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology,sobel
from scipy.ndimage.filters import maximum_filter,gaussian_filter,laplace,median_filter
import tools.mncc as mncc
import tools.geo as geo
from scipy import interpolate, stats
import glob,pickle
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Camera:
    def __init__(self, c_id, config, max_theta=7*np.pi/18.):
        site = config["site_id"]
        self.lat, self.lon = config["cameras"][site][c_id]["latlon"]
        self.height_group = config["cameras"][site][c_id]["group"]
        params = config["cameras"][site][c_id]["calibration_coef"]

        nx = config["cameras"]["img_w"]
        ny = config["cameras"]["img_h"]

        #### size of the undistorted image 
        if nx>=2000:
            nx=ny=2000
        else:
            nx=ny=1000

        self.c_id = c_id

        nx0=ny0=params[0]
        nr0=(nx0+ny0)/4
        xstart=int(params[2]-nx0/2+0.5); ystart=int(params[1]-ny0/2+0.5)
        self.nx0=int(nx0+0.5); self.ny0=int(ny0+0.5)
#         self.cx,self.cy=params[2:0:-1]
        self.max_theta=max_theta

        #####compute the zenith and azimuth angles for each pixel
        x0,y0=np.meshgrid(np.linspace(-self.nx0//2,self.nx0//2,self.nx0),np.linspace(-self.ny0//2,self.ny0//2,self.ny0));
        r0=np.sqrt(x0**2+y0**2)/nr0;
        self.roi=np.s_[ystart:ystart+self.ny0,xstart:xstart+self.nx0]
        self.rotation,self.beta,self.azm=params[3:6]

        roots=np.zeros(51)
        rr=np.arange(51)/100.0
        self.c1,self.c2,self.c3=params[6:9]
        for i,ref in enumerate(rr):
            roots[i]=np.real(np.roots([self.c3,0,self.c2,0,self.c1,-ref])[-1])
        theta0 = np.interp(r0/2,rr,roots)

        phi0 = np.arctan2(x0,y0) - self.rotation  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
        phi0 = phi0%(2*np.pi)

        #####correction for the mis-pointing error
        k=np.array((np.sin(self.azm),np.cos(self.azm),0))
        a=np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]);
        a = np.transpose(a,[1,2,0])
        b=np.cos(self.beta)*a + np.sin(self.beta)*np.cross(k,a,axisb=2) \
          + np.reshape(np.outer(np.dot(a,k),k),(self.ny0,self.nx0,3))*(1-np.cos(self.beta))
        theta0=np.arctan(np.sqrt(b[:,:,0]**2+b[:,:,1]**2)/b[:,:,2])
        phi0=np.arctan2(b[:,:,1],b[:,:,0])%(2*np.pi)

        self.valid0 = (theta0<max_theta) & (theta0>0);
#         theta0[self.valid0]=np.nan;
        self.theta0,self.phi0=theta0,phi0
        self.nx,self.ny=nx,ny
        max_tan = np.tan(max_theta)
        xbin,ybin=np.linspace(-max_tan,max_tan,self.nx), np.linspace(-max_tan,max_tan,self.ny)
        xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
        rgrid =xgrid*xgrid+ygrid*ygrid
        self.valid = rgrid <= max_tan*max_tan
        self.cos_th=1/np.sqrt(1+rgrid)
        rgrid=np.sqrt(rgrid)
        self.cos_p=ygrid/rgrid;
        self.sin_p=xgrid/rgrid;
        self.max_tan=max_tan

        x,y=theta0+np.nan, theta0+np.nan
        r=np.tan(theta0[self.valid0]);
        x[self.valid0],y[self.valid0]=r*np.sin(phi0[self.valid0]), r*np.cos(phi0[self.valid0])

        try:
            self.invalid=np.load(static_mask_path+self.camID+'_mask.npy')
            if (self.nx<=1000):
                tmp=st.block_average2(self.invalid.astype(np.float32),2)
                self.valid &= (tmp<0.2);
        except:
            self.invalid = np.zeros((ny, nx), dtype = bool)

        self.weights=st.prepare_bin_average2(x,y,xbin,ybin)
