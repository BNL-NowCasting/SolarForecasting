import numpy as np
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta,timezone
import os,ephem
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology,sobel
from scipy.ndimage.filters import maximum_filter,gaussian_filter,laplace,median_filter
import mncc, geo
from scipy import interpolate, stats
import glob,pickle
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pytz

BND_WIN = 20;
BND_RED_THRESH, BND_RBR_THRESH  = 2/1.5, 0.012/2
DRED_THRESH, DRBR_THRESH = 150, 157
STD_RED_THRESH, STD_RBR_THRESH = 1.2, 0.012

#static_mask_path='/home/dhuang3/ldata/masks/'  

coordinate = {'HD2C':[40.87203321,-72.87348295],'HD815_2':[40.87189059,-72.873687],\
               'HD490':[40.865968816,-72.884647222], 'HD1B':[40.8575056,-72.8547344], \
               'HD1A':[40.8580088,-72.8575717], 'HD1C':[40.85785,-72.8597],
               'HD5A':[40.947353, -72.899617], 'HD5B':[40.948044,-72.898372],
               'HD3A':[40.897122,-72.879053], 'HD3B':[40.8975,-72.877497],
               'HD4A':[40.915708,-72.892406],'HD4B':[40.917275,-72.891592],
               'HD2B':[40.872341,-72.874354]}
params = {'HD2C':[2821.0000,1442.8231,1421.0000,0.1700,-0.0135,-2.4368,0.3465,-0.0026,-0.0038],\
          'HD815_2':[2821.0000,1424,1449.0000,0.0310,-0.0114,-0.9816,0.3462,-0.0038,-0.0030 ],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD1B':[2830.0007,1473.2675,1459.7203,-0.0986,-0.0106,-1.2440,0.3441,-0.0015,-0.0042], \
          'HD1A':[2826.5389,1461.0000,1476.6598,-0.0097,0.0030,2.9563,0.3415,0.0004,-0.0044], \
          'HD1C':[2812.7874,1475.1453,1415.0000,0.1410,-0.0126,0.4769,0.3441,0.0004,-0.0046],
          'HD4A':[ 2815.9408,1411.8050,1500.0000,-0.0300,-0.0341,-1.4709,0.3555,-0.0136,0.0005 ], \
          'HD4B':[ 2832.5996,1429.9573,1465.0000,-0.0340,-0.0352,0.4037,0.3468,-0.0111,-0.0003 ], \
          'HD5A':[2813.7462,1472.2066,1446.3682,0.3196,-0.0200,-1.9636,0.3444,-0.0008,-0.0042], \
          'HD5B':[2812.1208,1470.1824,1465.0000,-0.1228,-0.0020,-0.5258,0.3441,-0.0001,-0.0042],\
#           'HD3A':[2807.8902,1436.1619,1439.3879,-0.3942,0.0527,2.4658,0.3334,0.0129,-0.0085],\
          'HD3A':[ 2826.5457,1461.8204,1465.0000,-0.4073,0.0054,1.9957,0.3571,-0.0177,0.0009 ],\
          'HD3B':[ 2821.2941,1469.8294,1465.0000,0.1918,-0.0149,-0.7192,0.3619,-0.0248,0.0043 ],\
          'HD2B':[2810.0000,1428.1154,1438.3745,0.1299,0.0167,2.0356,0.3480,-0.0049,-0.0025]}

deg2rad=np.pi/180

def localToUTC(t, local_tz):
    t_local = local_tz.localize(t, is_dst=None)
    t_utc = t_local.astimezone(pytz.utc)
    return t_utc
    
def UTCtimestampTolocal(ts, local_tz):
    t_utc = dt.datetime.fromtimestamp(ts,tz=pytz.timezone("UTC"))
    t_local = t_utc.astimezone(local_tz)
    return t_local


class camera:
    ###variable with the suffix '0' means it is for the raw, undistorted image
    def __init__(self, camID, max_theta=70,nx=2000,ny=2000,cam_tz=pytz.timezone("UTC"),static_mask_path=None):      
        #### size of the undistorted image 
        if nx>=2000:
            nx=ny=2000
        else:
            nx=ny=1000
        
        try:   #####check if the camera object is precomputed  
            with open(static_mask_path+camID+'.'+str(nx)+'.pkl','rb') as input:
                self.__dict__=pickle.load(input).__dict__;
                self.cam_tz = cam_tz
#             print(self.camID,self.nx)
            return 
        except:
            pass;
         
        self.cam_tz = cam_tz
        self.camID=camID
        self.lat, self.lon=coordinate[camID]
        nx0=ny0=params[camID][0]
        nr0=(nx0+ny0)/4
        xstart=int(params[camID][2]-nx0/2+0.5); ystart=int(params[camID][1]-ny0/2+0.5)
        self.nx0=int(nx0+0.5); self.ny0=int(ny0+0.5)
#         self.cx,self.cy=params[camID][2:0:-1]
        self.max_theta=max_theta
        
        #####compute the zenith and azimuth angles for each pixel
        x0,y0=np.meshgrid(np.linspace(-self.nx0//2,self.nx0//2,self.nx0),np.linspace(-self.ny0//2,self.ny0//2,self.ny0)); 
        r0=np.sqrt(x0**2+y0**2)/nr0;
        self.roi=np.s_[ystart:ystart+self.ny0,xstart:xstart+self.nx0]
        self.rotation,self.beta,self.azm=params[camID][3:6]
        
        roots=np.zeros(51)
        rr=np.arange(51)/100.0
        self.c1,self.c2,self.c3=params[camID][6:9]
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

        max_theta *= deg2rad 
        valid0 = (theta0<max_theta) & (theta0>0); 
#         theta0[valid0]=np.nan;
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
        r=np.tan(theta0[valid0]); 
        x[valid0],y[valid0]=r*np.sin(phi0[valid0]), r*np.cos(phi0[valid0])
        
        try:
            invalid=np.load(static_mask_path+self.camID+'_mask.npy')
            if (self.nx<=1000):
                tmp=st.block_average2(invalid.astype(np.float32),2)
                self.valid &= (tmp<0.2);
        except:
            pass;
        
        self.weights=st.prepare_bin_average2(x,y,xbin,ybin); 
    
        with open(static_mask_path+camID+'.'+str(nx)+'.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class image:
    ###image class
    def __init__(self, cam, fn):        
        self.camID=cam.camID
        self.nx,self.ny=cam.nx,cam.ny
        self.lon, self.lat, self.max_theta = cam.lon, cam.lat, cam.max_theta
        self.t_local=None
        self.fn=fn
        self.layers=0
        self.v=[]
        self.height=[]
        self.rgb=None
        self.sz=None
        self.saz=None
        self.red=None    #####spatial structure/texture of the red image, used by the cloud motion and height routines
        self.cm=None     #####cloud mask
        self.cam_tz = cam.cam_tz
#         self.cos_g=None

    def dump_img(self,filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    def undistort(self, cam, rgb=True, day_only=True):    
        """
        Undistort the raw image, set rgb, red, rbr, cos_g
        Input: rgb and day_only flags
        Output: rgb, red, rbr, cos_g will be specified.
        """           
        #####get the image acquisition time, this need to be adjusted whenever the naming convention changes 
        #ts=localToUTCtimestamp(datetime.strptime(self.fn[-18:-4],'%Y%m%d%H%M%S'),cam.cam_tz)    #get UTC timestamp
        #t_std=UTCtimestampTolocal(ts, pytz.timezone("UTC"))                                      #create datetime in UTC
        t_std = localToUTC(datetime.strptime(self.fn[-18:-4],'%Y%m%d%H%M%S'),self.cam_tz)
        #t_std=datetime.strptime(self.fn[-18:-4],'%Y%m%d%H%M%S');
        #t_std = t_local + timedelta(hours=5) #replace(tzinfo=timezone(-timedelta(hours=5)))       
        #print("\tUndistortion->t_local=%s\t\tt_std=%s\n" % (str(t_local),str(t_std)))
        print("\tUndistortion->t_std=%s\n" % (str(t_std)))
        gatech = ephem.Observer(); 
        gatech.date = t_std.strftime('%Y/%m/%d %H:%M:%S')
        gatech.lat, gatech.lon = str(self.lat),str(self.lon)
        sun=ephem.Sun()  ; sun.compute(gatech);        
        sz = np.pi/2-sun.alt; 
        self.sz = sz
        if day_only and sz>75*deg2rad:
            print("Night time (sun angle = %f), skipping\n" % sz)
            return
             
        saz = 180+sun.az/deg2rad; saz=(saz%360)*deg2rad;
        self.saz = saz

        try:
            im0=plt.imread(self.fn);
        except:
            print('Cannot read file:', self.fn)
            return None     
        im0=im0[cam.roi]

        cos_sz=np.cos(sz)        
        cos_g=cos_sz*np.cos(cam.theta0)+np.sin(sz)*np.sin(cam.theta0)*np.cos(cam.phi0-saz);   
        
        red0=im0[:,:,0].astype(np.float32); red0[red0<=0]=np.nan
#         rbr0=(red0-im0[:,:,2])/(im0[:,:,2]+red0)     
        if np.nanmean(red0[(cos_g>0.995) & (red0>=1)])>230: 
            mk=cos_g>0.98
            red0[mk]=np.nan 

        xsun, ysun = np.tan(sz)*np.sin(saz), np.tan(sz)*np.cos(saz)
        self.sun_x,self.sun_y = int(0.5*self.nx*(1+xsun/cam.max_tan)), int(0.5*self.ny*(1+ysun/cam.max_tan))

        invalid=~cam.valid
        
        red=st.fast_bin_average2(red0,cam.weights);
        red=st.fill_by_mean2(red,7, mask=(np.isnan(red)) & cam.valid)        
        red[invalid]=np.nan; 
#         plt.figure(); plt.imshow(red); plt.show();
        red -= st.rolling_mean2(red,int(self.nx//6.666))
        red[red>50]=50; red[red<-50]=-50
        red=(red+50)*2.54+1;         
        red[invalid]=0;        
        self.red=red.astype(np.uint8)
#         plt.figure(); plt.imshow(self.red); plt.show();

        if rgb:             
            im=np.zeros((self.ny,self.nx,3),dtype=im0.dtype)   
            for i in range(3):
                im[:,:,i]=st.fast_bin_average2(im0[:,:,i],cam.weights); 
                im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=(im[:,:,i]==0) & (cam.valid))
#                 im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=np.isnan(red))   
            im[self.red<=0]=0
            plt.figure(); plt.imshow(im); plt.show()
            self.rgb=im               

def cloud_mask(cam,img,img0):
    """
    Set cloud mask
    Input: None
    Output: cloud mask will be specified.
    """            
    cos_s=np.cos(img.sz); sin_s=np.sin(img.sz)
    cos_sp=np.cos(img.saz); sin_sp=np.sin(img.saz)
    cos_th=cam.cos_th; sin_th=np.sqrt(1-cos_th**2)
    cos_p=cam.cos_p; sin_p=cam.sin_p
    cos_g=cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p);  ###cosine of the angle between illumination and view directions    

    r0=img0.rgb[...,0].astype(np.float32); r0[r0<=0]=np.nan
    r1=img.rgb[...,0].astype(np.float32); r1[r1<=0]=np.nan
    rbr_raw=(r1-img.rgb[:,:,2])/(img.rgb[:,:,2]+r1)    
    rbr=rbr_raw.copy();
    rbr -= st.rolling_mean2(rbr,int(img.nx//6.666))
    rbr[rbr>0.08]=0.08; rbr[rbr<-0.08]=-0.08;
    rbr=(rbr+0.08)*1587.5+1;    ####scale rbr to 0-255
    mblue=np.nanmean(img.rgb[(cos_g<0.7) & (r1>0) & (rbr_raw<-0.01),2].astype(np.float32));
    err = r1-r0; err-=np.nanmean(err)    
    dif=st.rolling_mean2(abs(err),100)
    err=st.rolling_mean2(err,5)
    dif2=maximum_filter(np.abs(err),5)
    
    sky=(rbr<126) & (dif<1.2); sky|=dif<0.9; sky |= (dif<1.5) & (err<3) & (rbr<105)
    sky|=(rbr<70) ; sky &=  (img.red>0); 
    cld=(dif>2) & (err>4); cld |= (img.red>150) & (rbr>160) & (dif>3); 
    cld |= (rbr>180);   ####clouds with high rbr
    cld[cos_g>0.7]|=(img.rgb[cos_g>0.7,2]<mblue) & (rbr_raw[cos_g>0.7]>-0.01);  #####dark clouds
    cld &= dif>3
    total_pixel=np.sum(r1>0)

    min_size=50*img.nx/1000
    cld=remove_small_objects(cld, min_size=min_size, connectivity=4, in_place=True)
    sky=remove_small_objects(sky, min_size=min_size, connectivity=4, in_place=True)

    ncld=np.sum(cld); nsky=np.sum(sky)
    
    # these threshholds don't strictly need to match those used in forecasting / training
    if (ncld+nsky)<=1e-2*total_pixel:
        return;
    elif (ncld < nsky) and (ncld<=5e-2*total_pixel):   #####shortcut for clear or totally overcast conditions        
        img.cm=cld.astype(np.uint8)
        img.layers=1
        return
    elif (ncld > nsky) and (nsky<=5e-2*total_pixel):   
        img.cm=((~sky)&(r1>0)).astype(np.uint8)
        img.layers=1
        return       
    max_score=-np.Inf
    x0=-0.15; 
    ncld,nsky=0.25*nsky+0.75*ncld,0.25*ncld+0.75*nsky
#     ncld=max(ncld,0.05*total_pixel); nsky=max(nsky,0.05*total_pixel)
    for slp in [0.1,0.15]:     
        offset=np.zeros_like(r1);
        mk=cos_g<x0; offset[mk]=(x0-cos_g[mk])*0.05;
        mk=(cos_g>=x0) & (cos_g<0.72); offset[mk]=(cos_g[mk]-x0)*slp
        mk=(cos_g>=0.72); offset[mk]=slp*(0.72-x0)+(cos_g[mk]-0.72)*slp/3;
        rbr2=rbr_raw-offset;
        minr,maxr=st.lower_upper(rbr2[rbr2>-1],0.01)            
        rbr2 -= minr; rbr2/=(maxr-minr);
        
        lower,upper,step=-0.1,1.11,0.2
        max_score_local=-np.Inf
        for iter in range(3):
            for thresh in np.arange(lower,upper,step):
                mk_cld=(rbr2>thresh) #& (dif>1) & (rbr>70)
                mk_sky=(rbr2<=thresh) & (r1>0)
                bnd=st.get_border(mk_cld,10,thresh=0.2,ignore=img.red<=0)
#                 bnd=st.rolling_mean2(mk_cld.astype(np.float32),10,ignore=img.red<=0)
#                 bnd=(bnd<0.8) & (bnd>0.2)
                sc=[np.sum(mk_cld & cld)/ncld,np.sum(mk_sky & sky)/nsky,np.sum(dif2[bnd]>4)/np.sum(bnd), \
                    -5*np.sum(mk_cld & sky)/nsky,-5*np.sum(mk_sky & cld)/ncld,-5*np.sum(dif2[bnd]<2)/np.sum(bnd)]         
                score=np.nansum(sc)                
                if score>max_score_local:  
                    max_score_local=score
                    thresh_ref=thresh
                    if score>max_score:
                        max_score=score
                        img.cm=mk_cld.astype(np.uint8);
#                         rbr_ref=rbr2.copy();                        
#                         sc_ref=sc.copy()     
#                 print(slp,iter,lower,upper,step,score)
            lower,upper=thresh_ref-0.5*step,thresh_ref+0.5*step+0.001
            step/=4;        
          
#         lower,upper=st.lower_upper(rbr2[img.red>0],0.02)  
##         print(lower,upper)
#         for thresh in np.arange(lower-0.02,max(upper,lower+0.1),0.02):            
#             mk_cld=(rbr2>thresh) #& (dif>1) & (rbr>70)
#             mk_sky=(rbr2<=thresh) & (r1>0)
#             bnd=st.rolling_mean2(mk_cld.astype(np.float32),10,ignore=img.red<=0)
#             bnd=(bnd<0.8) & (bnd>0.2)
#             sc=[np.sum(mk_cld & cld)/ncld,np.sum(mk_sky & sky)/nsky,np.sum(dif2[bnd]>4)/np.sum(bnd),-5*np.sum(mk_cld & sky)/max(nsky,0.02*total_pixel),-5*np.sum(mk_sky & cld)/max(ncld,0.02*total_pixel),-5*np.sum(dif2[bnd]<2)/np.sum(bnd)]         
#             score=np.nansum(sc)
#             if score>max_score:
#                 max_score=score
#                 img.cm=mk_cld.astype(np.uint8); 
#                 rbr_ref=rbr2.copy();
#                 sc_ref=sc.copy() 
# #             print(slp,score)    
    
    img.layers=1;

#     print(img.camID,img.fn[-18:-4],np.round(max_score,3),np.round(sc_ref,3))
#     mblue=np.round(mblue,2)
#     fig,ax=plt.subplots(2,3,sharex=True,sharey=True);  ax[0,0].set_title(str(mblue));
#     ax[0,0].imshow(img.rgb); ax[0,1].imshow(rbr_raw,vmax=0.05); ax[0,2].imshow(img.cm); 
#     ax[1,0].imshow(err,vmin=-8,vmax=8);  ax[1,1].imshow(sky); ax[1,2].imshow(cld); plt.show()         
    
def cloud_motion_fft(convolver,fft1,fft2,ratio=0.75):
    """
    Determine cloud motion using existing convolver and fft objects
    Input: Convolver, and two fft objects 
    Output: Cloud motion vector, and max correlation
    """    
####use this routine if convolver and fft objects are ready    
    ny,nx=fft2[-2]
    try:
        corr=mncc.mncc_fft(convolver, fft1, fft2, ratio_thresh=ratio) 
#        plt.figure(); plt.imshow(corr); plt.show()    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx] 
    except:
        return np.nan, np.nan, np.nan

def cloud_motion(im1,im2,mask1=None, mask2=None,ratio=0.7, threads=1):
    """
    Determine cloud motion 
    Input: Images and masks for two frames
    Output: Cloud motion vector, and max correlation
    """        
####use this routine if the inputs are raw images   
    ny,nx=im2.shape 
#     if im1.dtype == np.uint8:
#         im1 = im1.astype(np.float32);
#     if im2.dtype == np.uint8:
#         im2 = im2.astype(np.float32);
    try:
        corr=mncc.mncc(im1,im2,mask1=mask1,mask2=mask2,ratio_thresh=ratio,threads=threads)     
#         plt.figure(); plt.imshow(corr); plt.show()    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx]
    except:
        return np.nan, np.nan, np.nan
        
            
def cloud_motion_fast(im1,im2,mask1=None, mask2=None,ratio=0.7, threads=1):
    """
    Determine cloud motion 
    Input: Images and masks for two frames
    Output: Cloud motion vector, and max correlation
    """        
####use this routine if the inputs are raw images   
    ny,nx=im2.shape 
#     if im1.dtype == np.uint8:
#         im1 = im1.astype(np.float32);
#     if im2.dtype == np.uint8:
#         im2 = im2.astype(np.float32);
    try:
        corr=mncc.mncc(im1,im2,mask1=mask1,mask2=mask2,ratio_thresh=ratio,threads=threads)     
#         plt.figure(); plt.imshow(corr); plt.show()    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx]
    except:
        return np.nan, np.nan, np.nan

def cloud_height(img1,img2,layer=0,distance=None):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image objects 
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0 or layer<=0:
        return []
        
    if img1.max_theta != img2.max_theta:
        print("The max_theta of the two cameras is different.");
        return np.nan, np.nan;
    if distance is None:
        distance = 6367e3*geo.distance_sphere(img1.lat,img1.lon,img2.lat,img2.lon)  

    max_tan=np.tan(img1.max_theta*deg2rad)         
    
    im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
#     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)

    
#     mask_tmpl=(img1.cm==layer) 
    mask_tmpl=(img1.cm==1) if layer==1 else (~(img1.cm==1) & (im1>0))           
        
    res = np.nan;
    try:
        corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,ratio_thresh=0.5)       
        if np.any(corr>0):
            max_idx=np.nanargmax(corr)
            deltay,deltax=max_idx//len(corr)-img2.ny+1,max_idx%len(corr)-img2.nx+1            
            deltar=np.sqrt(deltax**2+deltay**2)            
            height=distance/deltar*img1.nx/(2*max_tan)
            score=st.shift_2d(im1,deltax,deltay); score[score<=0]=np.nan; 
            score-=im2; score=np.nanmean(np.abs(score[(im2>0)]))
            score0=np.abs(im2-im1); score0=np.nanmean(score0[(im2>0) & (im1>0)])
#             print('Height',img1.camID,img2.camID,deltay,deltax,height,score0,score)
#             fig,ax=plt.subplots(1,2,sharex=True,sharey=True);  ax[0].set_title(str(deltax)+','+str(deltay));
#             ax[0].imshow(im2); ax[1].imshow(im1); plt.show();            
            if score0-score<=0.3*score0:
                res=np.nan
            else:
                res = min(13000,height)

    except:
        print('Cannot determine cloud height.');
#     print(np.nanmax(corr),height,deltay, deltax)
    return res             
         

def preprocess(cam,fn,outpath):
    if not os.path.isdir(outpath+fn[-18:-10]):
        os.makedirs(outpath+fn[-18:-10])
    t=localToUTC(datetime.strptime(fn[-18:-4],'%Y%m%d%H%M%S'),cam.cam_tz); 
    t_prev=t-timedelta(seconds=30);
    t_prev=t_prev.strftime('%Y%m%d%H%M%S');
    fn_prev=fn.replace(fn[-18:-4],t_prev);
    if len(glob.glob(fn_prev))<=0:
        return None

    print("\tpreprocess->fn=%s\n\t\tt=%s\t\tt_prev=%s\n" % (fn, str(t), str(t_prev)))
    flist=[fn_prev,fn]
    q=deque();      
    for f in flist:
        img=image(cam,f)  ###img object contains four data fields: rgb, red, rbr, and cm 
        img.undistort(cam,rgb=True)  ###undistortion
        print("undistorted: "+f)
        if img.rgb is None:
            return None
        q.append(img)  

        if len(q)<=1: 
            continue
        ####len(q) is always 2 beyond this point
        print("deque two sequential images ",len(q))
        r1=q[-2].red.astype(np.float32); r1[r1<=0]=np.nan
        r2=q[-1].red.astype(np.float32); r2[r2<=0]=np.nan
        err0 = r2-r1;
       
        dif=np.abs(err0); 
        dif=st.rolling_mean2(dif,20)
        semi_static=(abs(dif)<10) & (r1-127>100)
        semi_static=morphology.binary_closing(semi_static,np.ones((10,10)))
        semi_static=remove_small_objects(semi_static,min_size=200, in_place=True)
        q[-1].rgb[semi_static]=0;
        r2[semi_static]=np.nan

        cloud_mask(cam,q[-1],q[-2]); ###one-layer cloud masking        
        if (q[-1].cm is None):
            q.popleft();             
            continue
        if (np.sum((q[-1].cm>0))<2e-2*img.nx*img.ny):   ######cloud free case
            q[-1].layers=0; 
        else:               
            dilated_cm=morphology.binary_dilation(q[-1].cm,np.ones((15,15))); dilated_cm &= (r2>0)
            vy,vx,max_corr = cloud_motion(r1,r2,mask1=r1>0,mask2=dilated_cm, ratio=0.7, threads=4);
            if np.isnan(vy):  
                q[-1].layers=0;
            else:
                q[-1].v += [[vy,vx]]; q[-1].layers=1;        
        #         err = r2-st.shift2(r1,-vx,-vy); err[(r2+st.shift2(r1,-vx,-vy)==0)]=np.nan;  
        # 
        #         mask2=st.rolling_mean2(np.abs(err)-np.abs(err0),40)<-2
        #         mask2=remove_small_objects(mask2,min_size=300, in_place=True)
        #         mask2=morphology.binary_dilation(mask2,np.ones((15,15)))
        #         mask2 = (~mask2) & (r2>0) & (np.abs(r2-127)<30) & (err>-100) #& (q[-1].cm>0)
        #         if np.sum(mask2 & (q[-1].cm>0))>200e-2*img.nx*img.ny:
        #             vy,vx,max_corr = cloud_motion(r1,r2,mask1=r1>0,mask2=mask2, ratio=0.7, threads=4);
        #             if np.isnan(vy):
        #                 q.popleft(); 
        #                 continue
        #             vdist = np.sqrt((vy-q[-1].v[-1][0])**2+(vx-q[-1].v[-1][1])**2)
        #             if vdist>=5 and np.abs(vy)+np.abs(vx)>2.5 and vdist>0.3*np.sqrt(q[-1].v[-1][0]**2+q[-1].v[-1][1]**2):
        #                 score1=np.nanmean(np.abs(err[mask2])); 
        #                 err2=r2-st.shift2(r1,-vx,-vy); err2[(r2==0) | (st.shift2(r1,-vx,-vy)==0)]=np.nan;  
        #                 score2=np.nanmean(np.abs(err2[mask2]));
        #                 if score2<score1:
        #                     q[-1].v += [[vy,vx]]; q[-1].layers=2;
        #                     dif=st.rolling_mean2(np.abs(err)-np.abs(err2),40)>0
        #                     dif=remove_small_objects(dif,min_size=300, in_place=True)
        #                     q[-1].cm[dif & (q[-1].cm>0)]=q[-1].layers;
        outpkl=os.path.join(outpath,f[-18:-10],f[-23:-4]+'.pkl')
        print("Dumping "+outpkl)
        try:
            q[-1].dump_img(outpkl)
        except:
            print("Failed dumping "+outpkl)
        q.popleft()             

    return q[-1]   


class stitch:
    ###image class
    def __init__(self, time):        
        t_local=time
        self.rgb=None
        self.cm=None

    def dump_stitch(self,filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("writing: %s" % filename)
