import numpy as np
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from skimage.morphology import remove_small_objects
from scipy.ndimage.filters import maximum_filter
import mncc, geo
from scipy import interpolate

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
          'HD4A':[2813.3741,1435.1706,1453.7087,-0.0119,-0.0857,-1.8675,0.3499,-0.0033,-0.0027], \
          'HD4B':[2809.2813,1446.4900,1438.0777,-0.0237,-0.0120,-1.3384,0.3479,-0.0024,-0.0037], \
          'HD5A':[2813.7462,1472.2066,1446.3682,0.3196,-0.0200,-1.9636,0.3444,-0.0008,-0.0042], \
          'HD5B':[2812.1208,1470.1824,1465.0000,-0.1228,-0.0020,-0.5258,0.3441,-0.0001,-0.0042],\
          'HD3A':[2807.8902,1436.1619,1439.3879,-0.3942,0.0527,2.4658,0.3334,0.0129,-0.0085],\
          'HD3B':[2814.3693,1473.3718,1445.8960,0.1977,-0.0350,-1.3646,0.3473,-0.0031,-0.0031],\
          'HD2B':[2810.0000,1428.1154,1438.3745,0.1299,0.0167,2.0356,0.3480,-0.0049,-0.0025]}

deg2rad=np.pi/180

class camera:
    ###variable with the suffix '0' means it is for the raw, undistorted image
    def __init__(self, camID, max_theta=70,nx=2000,ny=2000):        
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
        self.valid0 = (theta0<max_theta) & (theta0>0); 
#         theta0[self.valid0]=np.nan;
        self.theta0,self.phi0=theta0,phi0
        
        #### size of the undistorted image 
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
        
#         from sklearn.neighbors import KDTree
#         tree = KDTree(np.column_stack((x[self.valid0],y[self.valid0])))
#         nearest_dist, nearest_ind = tree.query(np.column_stack((xbin.ravel(),ybin.ravel())), k=2)
        self.weights=st.prepare_bin_average2(x,y,xbin,ybin);    

class image:
    ###image class
    def __init__(self, cam, fn):        
        self.cam=cam
        self.fn=fn
        self.layers=0
        self.v=[]
        self.rgb=None
        self.sz=None
        self.saz=None
        self.red=None    #####spatial structure/texture of the red image, used by the cloud motion and height routines
        self.rbr=None    #####normalized red/blue ratio
        self.cm=None     #####cloud mask
#         self.cos_g=None
    
    def undistort(self, rgb=True, day_only=True):    
        """
        Undistort the raw image, set rgb, red, rbr, cos_g
        Input: rgb and day_only flags
        Output: rgb, red, rbr, cos_g will be specified.
        """           
        #####get the image acquisition time, this need to be adjusted whenever the naming convention changes 
        t_cur=datetime.strptime(self.fn[-18:-4],'%Y%m%d%H%M%S');     
        t_std = t_cur-timedelta(hours=5)     #####adjust UTC time into local standard time            
        sz = 90-ps.get_altitude(self.cam.lat,self.cam.lon,t_std); sz*=deg2rad;
        self.sz = sz
        if day_only and sz>85*deg2rad:
            return
             
        saz = 360-ps.get_azimuth(self.cam.lat,self.cam.lon,t_std); saz=(saz%360)*deg2rad;
        self.saz = saz

        try:
            im0=plt.imread(self.fn);
        except:
            print('Cannot read file:', self.fn)
            return None
        im0=im0[self.cam.roi]
        im0[~self.cam.valid0,:]=0

        cos_sz=np.cos(sz)        
        cos_g=cos_sz*np.cos(self.cam.theta0)+np.sin(sz)*np.sin(self.cam.theta0)*np.cos(self.cam.phi0-saz);   
        
        red0=im0[:,:,0].astype(np.float32); red0[red0<=0]=np.nan
        rbr0=(red0-im0[:,:,2])/(im0[:,:,2]+red0)        

        if np.nanmean(red0[(cos_g>0.995) & (red0>=1)])>230: 
            mk=cos_g>0.98
            red0[mk]=np.nan 
            rbr0[mk]=np.nan 
        
        rbr=st.fast_bin_average2(rbr0,self.cam.weights); 
        rbr=st.fill_by_mean2(rbr,7, mask=(np.isnan(rbr)) & self.cam.valid) 
        self.rbr=rbr
        
        red0-=st.rolling_mean2(red0,300,ignore=np.nan)
        red=st.fast_bin_average2(red0,self.cam.weights); 
        red=st.fill_by_mean2(red,7, mask=(np.isnan(red)) & self.cam.valid)
        red[red>50]=50; red[red<-50]=-50
        red=(red+51)*2.5+0.5;       
        self.red=red.astype(np.uint8)
                
        if rgb:             
            im=np.zeros((self.cam.ny,self.cam.nx,3),dtype=im0.dtype)   
            for i in range(3):
                im[:,:,i]=st.fast_bin_average2(im0[:,:,i],self.cam.weights); 
                im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=(im[:,:,i]==0) & (self.cam.valid))
#                 im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=np.isnan(red))   
            im[red<=0]=0
            self.rgb=im   
        
    def cloud_mask(self):
        """
        Set cloud mask
        Input: None
        Output: cloud mask will be specified.
        """          
        if self.rbr is None or self.sz>85*deg2rad:
            return 

        d_rbr=self.rbr-st.rolling_mean2(self.rbr,50);

        cos_s=np.cos(self.sz); sin_s=np.sin(self.sz)
        cos_sp=np.cos(self.saz); sin_sp=np.sin(self.saz)
        cos_th=self.cam.cos_th; sin_th=np.sqrt(1-cos_th**2)
        cos_p=self.cam.cos_p; sin_p=self.cam.sin_p
        cos_g=cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p);  ###cosine of the angle between illumination and view directions    
#         self.cos_g=((1+cos_g)*127.5).astype(np.uint8);
        thresh_max1=0.3*(1-np.cos(self.sz))-0.2;  
        thresh_max2=0.3*(1-np.cos(self.sz))-0.2; thresh_max2=0.5*thresh_max2+0.5*min(0.1,np.nanmean(self.rbr[(cos_g>0.97) & (d_rbr>0.0)]));   
        thresh1=np.nan+cos_g; thresh2=np.nan+cos_g; 
        thresh1[cos_g>0]=thresh_max1+0.2*cos_g[cos_g>0]**2-0.2;  thresh1[cos_g<=0]=thresh_max1-0.2
#         thresh2[cos_g>0]=thresh_max2+0.15*cos_g[cos_g>0]**2-0.15;  thresh2[cos_g<=0]=thresh_max2-0.15
        thresh2=thresh_max2+0.15*cos_g**2-0.15;  thresh2[cos_g<=0]=0.7*(thresh_max2-0.15)+0.3*thresh2[cos_g<=0]
        
        mask1=((d_rbr>0.02) & (self.rbr>thresh1-0.0) & (self.rbr<0.25)); ####cloud
        mask2=((d_rbr<-0.02) & (self.rbr>-0.6) & (self.rbr<thresh2+0.0)); #####clear       
        mask1=remove_small_objects(mask1, min_size=100, connectivity=4)
        mask2=remove_small_objects(mask2, min_size=100, connectivity=4)   
     
        if np.sum(mask1)>1e3 and np.sum(mask2)>1e3:
            xp=np.array([0.58, 0.85, 1.0]); 
            xc=np.array([-1,0.2]+[0.5*(xp[i]+xp[i+1]) for i in range(len(xp)-1)])
            cloud_thresh=xc+np.nan;  clear_thresh=xc+np.nan
            for i in range(len(xp)):            
                mka= cos_g<xp[0] if i==0 else ((cos_g>=xp[i-1]) & (cos_g<xp[i]));
                mk1=mask1 & mka; mk2=mask2 & mka;    
                mrbr=np.nanmean(self.rbr[mka])
                if np.sum(mk1)>5e2 and np.sum(mk2)>5e2:                   
                    clear_thresh[i+1]=np.nanmean(self.rbr[mk2]);  
                    cloud_thresh[i+1]=min(mrbr+0.2,np.nanmean(self.rbr[mk1]));
                else:
                    if mrbr>np.nanmean(thresh2[mka]):
                        cloud_thresh[i+1]=mrbr
                    else:
                        clear_thresh[i+1]=mrbr                                 
                        
#             print(clear_thresh, cloud_thresh)
            if any(cloud_thresh>-1) and any(clear_thresh>-1) and np.nanmean(cloud_thresh[1:]-clear_thresh[1:])>0.035:                           
                if any(np.isnan(cloud_thresh[1:])) or any(np.isnan(clear_thresh[1:])):
                    fill_gaps_thresh(cloud_thresh,clear_thresh,xc)
#                 print(clear_thresh, cloud_thresh)
                if cloud_thresh[-1]-clear_thresh[-1]<0.12:
                    mrbr=np.nanmean(self.rgb[cos_g>xp[-2]])
                    if mrbr>thresh_max2:
                        clear_thresh[-1]-=0.1
                    elif mrbr<thresh_max1:
                        cloud_thresh[-1]+=0.1     
                clear_thresh[0]=clear_thresh[1]; cloud_thresh[0]=cloud_thresh[1] 
                if np.sum(clear_thresh>-1)>=2 and np.sum(cloud_thresh>-1)>=2:                  
                    f = interpolate.interp1d(xc[cloud_thresh>-1],cloud_thresh[cloud_thresh>-1],fill_value='extrapolate')
                    cloud=f(cos_g)
                    f = interpolate.interp1d(xc[clear_thresh>-1],clear_thresh[clear_thresh>-1],fill_value='extrapolate')
                    clear=f(cos_g)   
                    d1=np.abs(self.rbr-cloud); d2=np.abs(self.rbr-clear)
    #                 fig,ax=plt.subplots(1,2,sharex=True,sharey=True);
    #                 ax[0].imshow(clear); ax[0].axis('off') #####original 
    #                 ax[1].imshow(cloud); ax[1].axis('off')        
                    self.cm=(0.6*d1<=d2).astype(np.uint8);
                    self.layers=1
                    return
        
        if np.nanmean(self.rbr[500:-500,500:-500])>-0.15:
            self.cm=self.cam.valid;
            self.layers=1            
        else:
            self.cm=np.zeros(mask1.shape,dtype=np.uint8)

# def detrend(img,cm,cos_g):
#     xp=np.array([-0.3, 0.18, 0.52, 0.8, 0.92, 1.0]); 
#     xp=(1+xp)*127.5;
#     xc=st.rolling_mean(xp,2)[1:]
#     y=xp[1:]+np.nan
#     for i in range(1,len(xp)):            
#         mk= (cos_g>=xp[i-1]) & (cos_g<xp[i]) & (cm>0);
#         if np.sum(mk)<5e3:
#             continue
#         y[i-1]=np.nanmean(img[mk])
#     valid=y>0
#     if np.sum(valid)<=2:
#         return
#     x=xc[valid]-np.mean(xc[valid]);
#     trend=np.sum(x*(y[valid]-np.mean(y[valid])))/np.sum(x**2)
#     mk=(img>0)
#     img[mk]-=(cos_g[mk]*trend)    
                  
def fill_gaps_thresh(cloud,clear,x):
    diff=cloud-clear
    if np.isnan(cloud[1]):
        inext=np.argmax(diff[2:]>-100)  
        cloud[1]=clear[1]+diff[inext]
    if np.isnan(cloud[-1]):
        iprev=np.argmax(diff[::-1]>-100)  
        cloud[-1]=clear[-1]+diff[::-1][iprev]    
    if np.isnan(clear[1]):
        inext=np.argmax(diff[2:]>-100)  
        clear[1]=cloud[1]-diff[inext]
    if np.isnan(clear[-1]):
        iprev=np.argmax(diff[::-1]>-100)  
        clear[-1]=cloud[-1]-diff[::-1][iprev]         
    if any(np.isnan(cloud[1:])):
        mk=cloud>-1
        if np.sum(mk)>=2:        
            f = interpolate.interp1d(x[mk],cloud[mk],fill_value='extrapolate')
            cloud[~mk]=f(x[~mk])
    if any(np.isnan(clear[1:])):
        mk=clear>-1
        if np.sum(mk)>=2:
            f = interpolate.interp1d(x[mk],clear[mk],fill_value='extrapolate')
            clear[~mk]=f(x[~mk])        

def cloud_motion_fft(convolver,fft1,fft2,ratio=0.7):
    """
    Determine cloud motion using existing convolver and fft objects
    Input: Convolver, and two fft objects 
    Output: Cloud motion vector, and max correlation
    """    
####use this routine if convolver and fft objects are ready    
    ny,nx=fft2[-2]
    try:
        corr=mncc.mncc_fft(convolver, fft1, fft2, ratio_thresh=ratio) 
#     plt.figure(); plt.imshow(corr)  
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx] 
    except:
        return None, None, None

def cloud_motion(im1,im2,mask1=None, mask2=None,ratio=0.7, threads=1):
    """
    Determine cloud motion 
    Input: Images and masks for two frames
    Output: Cloud motion vector, and max correlation
    """        
####use this routine if the inputs are raw images   
    ny,nx=im2.shape 
    try:
        corr=mncc.mncc(im1,im2,mask1=mask1,mask2=mask2,ratio_thresh=ratio,threads=threads)     
#     plt.figure(); plt.imshow(corr)    
        max_idx=np.nanargmax(corr)
        vy,vx=max_idx//len(corr)-ny+1,max_idx%len(corr)-nx+1    
        return vy,vx,corr.ravel()[max_idx]
    except:
        return None, None, None
    
# def cloud_height(img1,img2,distance=None):
#     """
#     Determine the cloud height for each cloud layer in img1
#     Input: Two image object 
#     Output: Cloud height, and max correlation
#     """
#     if img1.layers<=0:
#         return []
#     
#     cam1=img1.cam; cam2=img2.cam
#     
#     if cam1.max_theta != cam2.max_theta:
#         print("The max_theta of the two cameras is different.");
#         return None, None;
#     if distance is None:
#         distance = 6367e3*geo.distance_sphere(cam1.lat,cam1.lon,cam2.lat,cam2.lon)  
# 
#     max_tan=np.tan(cam1.max_theta*deg2rad)         
# 
#     im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
# #     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)
#     
#     res=[]    
#     for ilayer in range(img1.layers):
#         mask_tmpl=img1.cm==ilayer+1           
#         if ilayer>=1:
#             mask_tmpl=maximum_filter(mask_tmpl,10) 
#             plt.figure(); plt.imshow(im1*mask_tmpl)
#             plt.figure(); plt.imshow(im2)
#         elif img1.layers>=2:
#             mask_layer2=img1.cm==2
#             mask_layer2=maximum_filter(mask_layer2,50) 
#             mask_tmpl[mask_layer2]=False
#         corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,show_corr=True)       
#         if np.any(corr>0):
#             max_idx=np.nanargmax(corr)
#             deltay,deltax=max_idx//len(corr)-cam2.ny+1,max_idx%len(corr)-cam2.nx+1
#             print(deltay,deltax)
#             deltar=np.sqrt(deltax**2+deltay**2)
#             height=distance/deltar*cam1.nx/(2*max_tan)
#             res += [height,corr.ravel()[max_idx]]
#         
# #     print(np.nanmax(corr),height,deltay, deltax)
#     return res     
            
def cloud_height(img1,err1,img2,err2,distance=None):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image object 
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0:
        return []
    
    cam1=img1.cam; cam2=img2.cam
    
    if cam1.max_theta != cam2.max_theta:
        print("The max_theta of the two cameras is different.");
        return None, None;
    if distance is None:
        distance = 6367e3*geo.distance_sphere(cam1.lat,cam1.lon,cam2.lat,cam2.lon)  

    max_tan=np.tan(cam1.max_theta*deg2rad)         

    im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
#     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)
    
    res=[]    
    for ilayer in range(img1.layers):
        mask_tmpl=img1.cm==ilayer+1           
        if ilayer>=1:
            im1=err1+30; im2=err2+30;
            mask_tmpl=np.abs(err1)>5
#             mask_tmpl=maximum_filter(mask_tmpl,10) 
#             plt.figure(); plt.imshow(im1)
#             plt.figure(); plt.imshow(im2)
        elif img1.layers>=2:
            mask_layer2=img1.cm==2
            mask_layer2=maximum_filter(mask_layer2,50) 
            mask_tmpl[mask_layer2]=False
        try:
            corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,ratio_thresh=0.5,show_corr=False)       
            if np.any(corr>0):
                max_idx=np.nanargmax(corr)
                deltay,deltax=max_idx//len(corr)-cam2.ny+1,max_idx%len(corr)-cam2.nx+1
#                print(deltay,deltax)
                deltar=np.sqrt(deltax**2+deltay**2)
                height=distance/deltar*cam1.nx/(2*max_tan)
                res += [height,corr.ravel()[max_idx]]
        except:
            res += [None, None];
#     print(np.nanmax(corr),height,deltay, deltax)
    return res             
        
def stitch(img1,img2,height):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image object 
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0:
        return []  
    
    cam1=img1.cam; cam2=img2.cam
    max_tan=np.tan(cam1.max_theta*deg2rad)         

    distance = 6367e3*geo.distance_sphere(cam1.lat,cam1.lon,cam2.lat,cam2.lon)
    distance_y = np.pi*6376e3*(cam1.lat-cam2.lat)/180
    distance_x = np.sqrt(distance**2-distance_y**2);     
    if cam2.lon-cam1.lon:
        distance_x *= -1 
    print(distance,distance_y,distance_x)
    
    dx = distance_x/height*cam1.nx/(2*max_tan)
    dy = distance_y/height*cam1.nx/(2*max_tan)
    
    return dy, dx
    
