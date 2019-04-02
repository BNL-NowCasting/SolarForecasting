import numpy as np
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import ephem
from skimage.morphology import remove_small_objects
from scipy.ndimage import morphology,sobel
from scipy.ndimage.filters import maximum_filter,laplace
import mncc, geo
from scipy import interpolate, stats
import pickle

BND_RED_THRESH, BND_RBR_THRESH  = 2, 0.012
DRED_THRESH, DRBR_THRESH = 150, 157
STD_RED_THRESH, STD_RBR_THRESH = 1.2, 0.012
static_mask_path='~/data/masks/'  

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
        valid0 = (theta0<max_theta) & (theta0>0); 
#         theta0[valid0]=np.nan;
        self.theta0,self.phi0=theta0,phi0
        
        #### size of the undistorted image 
        if nx>=2000:
            nx=ny=2000
        else:
            nx=ny=1000
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

class image:
    ###image class
    def __init__(self, cam, fn):        
        self.camID=cam.camID
        self.nx,self.ny=cam.nx,cam.ny
        self.lon, self.lat, self.max_theta = cam.lon, cam.lat, cam.max_theta
        self.time=None
        self.fn=fn
        self.layers=0
        self.v=[]
        self.height=[]
        self.rgb=None
        self.sz=None
        self.saz=None
        self.red=None    #####spatial structure/texture of the red image, used by the cloud motion and height routines
        self.rbr=None    #####normalized red/blue ratio
        self.cm=None     #####cloud mask
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
        self.time=datetime.strptime(self.fn[-18:-4],'%Y%m%d%H%M%S');     
        gatech = ephem.Observer(); 
        gatech.date = self.time.strftime('%Y/%m/%d %H:%M:%S')
        gatech.lat, gatech.lon = str(self.lat),str(self.lon)
        sun=ephem.Sun()  ; sun.compute(gatech);        
        sz = np.pi/2-sun.alt; 
        self.sz = sz
        if day_only and sz>75*deg2rad:
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
        rbr0=(red0-im0[:,:,2])/(im0[:,:,2]+red0)        
        if np.nanmean(red0[(cos_g>0.995) & (red0>=1)])>30: 
            mk=cos_g>0.98
            red0[mk]=np.nan 
            rbr0[mk]=np.nan 
        
        invalid=~cam.valid
#         rbr0[rbr0>0.1]=0.1
        rbr2=rbr0.copy()
        rbr2-=st.rolling_mean2(rbr0,int(self.nx//4),ignore=np.nan)
        rbr0 -= st.rolling_mean2(rbr0,int(self.nx//6.666),ignore=np.nan)
        mk=rbr0>0.005; rbr0[mk]=np.maximum(rbr0[mk],rbr2[mk]) 
#         mk=rbr0<-0.005; rbr0[mk]=np.minimum(rbr0[mk],rbr2[mk]) 
        rbr=st.fast_bin_average2(rbr0,cam.weights); 
        rbr=st.fill_by_mean2(rbr,7, mask=(np.isnan(rbr)) & cam.valid) 
        rbr[rbr>0.08]=0.08; rbr[rbr<-0.08]=-0.08;
        rbr=(rbr+0.08)*1587.5+1;
        rbr[invalid]=0              
        self.rbr=rbr.astype(np.uint8)
        
        red0 -= st.rolling_mean2(red0,int(self.nx//6.666),ignore=np.nan)
        red=st.fast_bin_average2(red0,cam.weights); 
        red=st.fill_by_mean2(red,7, mask=(np.isnan(red)) & cam.valid)
        red[red>50]=50; red[red<-50]=-50
        red=(red+50)*2.54+1; 
        red[invalid]=0;
        self.red=red.astype(np.uint8)
                
        if rgb:             
            im=np.zeros((self.ny,self.nx,3),dtype=im0.dtype)   
            for i in range(3):
                im[:,:,i]=st.fast_bin_average2(im0[:,:,i],cam.weights); 
                im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=(im[:,:,i]==0) & (cam.valid))
#                 im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=np.isnan(red))   
            im[self.red<=0]=0
            self.rgb=im   

    def cloud_mask(self,cam):
        """
        Set cloud mask
        Input: None
        Output: cloud mask will be specified.
        """            
        cos_s=np.cos(self.sz); sin_s=np.sin(self.sz)
        cos_sp=np.cos(self.saz); sin_sp=np.sin(self.saz)
        cos_th=cam.cos_th; sin_th=np.sqrt(1-cos_th**2)
        cos_p=cam.cos_p; sin_p=cam.sin_p
        cos_g=cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p);  ###cosine of the angle between illumination and view directions    
#         self.cos_g=((1+cos_g)*127.5).astype(np.uint8);
        
        #####determine the cloud edges
        red0=self.rgb[:,:,1].astype(np.float32); red0[cos_g>0.98]=0
        rbr0=(self.rgb[:,:,0].astype(np.float32)-self.rgb[:,:,2])/(self.rgb[:,:,2]+self.rgb[:,:,0].astype(np.float32));  
        red=self.red.astype(np.float32); red[cos_g>0.98]=0
        rbr=self.rbr.astype(np.float32); rbr[(rbr<=0) | (cos_g>0.98)]=np.nan; 
        
        BND_WIN=10 if self.nx<=1000 else 15;            
        mred=st.rolling_mean2(red0,BND_WIN,ignore=0); mred[cos_g>0.95]=np.nan
        vred=st.rolling_mean2(red0**2,BND_WIN,ignore=0)-mred**2; #vred[vred>50]=50
        mrbr=st.rolling_mean2(rbr0,BND_WIN,ignore=0)
        vrbr=st.rolling_mean2(rbr0**2,BND_WIN,ignore=0)-mrbr**2; #vrbr[vrbr>50]=50
        
        total_pixel=np.sum(red>0)
#         bnd = (vred>BND_RED_THRESH)  
        for factor in range(1,5):
            bnd = ((100*np.sqrt(vred)/mred>factor*BND_RED_THRESH) & (np.sqrt(vrbr)>BND_RBR_THRESH/3))  \
                  | ((np.sqrt(vrbr)>BND_RBR_THRESH) & (100*np.sqrt(vred)/mred>BND_RED_THRESH/3))
#             print(factor,np.sum(bnd), 0.1*np.sum(red>0))   
            bnd1 = ((100*np.sqrt(vred)/mred>factor*BND_RED_THRESH/2) & (np.sqrt(vrbr)>BND_RBR_THRESH))      
            if np.sum(bnd)<=0.1*total_pixel:
                break;

        ####classify the cloud boundary pixels into cld or sky, which will serve as seeds for further growth
        mk_cld=(red>=DRED_THRESH) | (rbr>=DRBR_THRESH) | ((red+rbr>DRED_THRESH+DRBR_THRESH-20) & (red>DRED_THRESH-5));
        mk_sky=(rbr<=255-DRBR_THRESH+5) & (rbr>0);
        mk_cld=remove_small_objects(mk_cld, min_size=500*self.nx/2000, connectivity=4)
        mk_sky=remove_small_objects(mk_sky, min_size=500*self.nx/2000, connectivity=4)
#         mk_sky=morphology.binary_dilation(mk_sky,np.ones((3,3)));  mk_sky[mk_cld]=False;
#         mk_cld=morphology.binary_dilation(mk_cld,np.ones((3,3)));  mk_cld[mk_sky]=False;                            
        
        if np.sum(mk_sky)<=3e-4*total_pixel:    
            self.cm=(red>0).astype(np.uint8)
            self.layers=1
            return
        
        if np.sum(mk_cld)<=3e-4*total_pixel:
            self.cm=(red>-1).astype(np.uint8)
            return        
        
        bins=np.arange(50,251,5);  bcs=0.5*(bins[:-1]+bins[1:])
        mk=(mk_sky) & (red>0); 
        sky_rbr = st.bin_average(rbr0[mk],red0[mk],bins);
        if np.sum(~np.isnan(sky_rbr))<=1:
            self.cm=(red>0).astype(np.uint8)
            self.layers=1
            return                
        sky_rbr=st.rolling_mean(sky_rbr,20,fill_nan=True)
        coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
#         print('rbr-red fitting:',coeff, red0.shape)
        if coeff[0]>2e-4:
            rbr2=rbr0-np.polyval(coeff,red0)
        else:
            self.cm=(red>0).astype(np.uint8)
            self.layers=1
            return
#             f_sky = interpolate.interp1d(bcs,sky_rbr,fill_value='extrapolate') 
#             rbr2=rbr0-f_sky(red0)
        
#         bins=np.arange(-0.1,0.951,0.03);  bcs=0.5*(bins[:-1]+bins[1:])
#         mk=(mk_sky) & (red>0); 
#         sky_rbr = st.bin_average(rbr2[mk],cos_g[mk],bins);
#         if np.sum(~np.isnan(sky_rbr))<=1:
#             thr_sky=np.zeros_like(red)+np.nanmean(rbr2[mk]);
#         else:
#             sky_rbr=st.rolling_mean(sky_rbr,10,fill_nan=True)
#             coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
#             print('sky coef:',coeff)
#             if (coeff[0]<=0):
#                 thr_sky=np.zeros_like(red)+np.nanmean(rbr2[mk]);
#             else:
#                 thr_sky=np.polyval(coeff,cos_g)
#                 thr_sky[cos_g>0.95]=np.polyval(coeff,0.95)
#                 thr_sky[cos_g<-0.1]=np.polyval(coeff,-0.1)
#         
#         mk=(mk_cld) & (red>0); 
#         thr_cld=np.zeros_like(red)+np.nanmean(rbr2[mk]);
# #         sky_rbr = st.bin_average(rbr2[mk],cos_g[mk],bins);
# #         sky_rbr=st.rolling_mean(sky_rbr,10,fill_nan=True)
# #         coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
# #         print('cloud coef:',coeff)
# #         if (coeff[0]<=0):
# #             thr_cld=np.zeros_like(red)+np.nanmean(rbr2[mk]);
# #         else:
# #             thr_cld=np.polyval(coeff,cos_g)    
#         
# #         mk_cld[rbr2>0.4*thr_cld+0.6*thr_sky]=True; mk_cld[red<=0]=False
        
        mk=(~mk_cld) & (red>0)
        hist,bins=np.histogram(rbr2[mk],bins=100,range=(-0.1,0.1))
        bcs=0.5*(bins[:-1]+bins[1:])
        total=np.sum(hist);  cumhist=np.cumsum(hist);
        min_rbr=bcs[np.argmax(cumhist>0.02*total)];
        max_rbr=bcs[100-np.argmax(cumhist[::-1]<=0.97*total)];
#         print('min, max:', min_rbr,max_rbr,np.nanmean(rbr2))
        self.cm=mk_cld.astype(np.uint8);
        max_score=0;
        for w in np.arange(-0.1,1.51,0.1):
            cld=mk_cld.copy()
            cld[rbr2>min_rbr+(max_rbr-min_rbr)*(w+0.1)/1.6]=True; cld[red<=0]=False
#             cld[rbr2>w*thr_cld+(1-w)*thr_sky]=True; cld[red<=0]=False
            mcld=st.rolling_mean2(cld.astype(np.float32),3,mask_ignore=red<=0,fill_nan=True);
            bnd_e=((mcld>0.2) & (mcld<0.95))
            bnd_e=remove_small_objects(bnd_e, min_size=150*self.nx/2000, connectivity=4)
            nvalid=np.sum(bnd_e & bnd1)
            score=nvalid**2/np.sum(bnd_e)
#             print(w,nvalid, score/nvalid)
            if score>max_score:
                max_score=score
                self.cm=cld.astype(np.uint8);
                
#         self.cm=mk_cld;
        self.layers=1;
#         fig,ax=plt.subplots(2,3,sharex=True,sharey=True);     
#         ax[0,0].imshow(self.rgb); ax[0,1].imshow(rbr0,vmin=-0.15,vmax=0.1); ax[0,2].imshow(self.cm); 
#         ax[1,0].imshow(rbr2,vmin=-0.1,vmax=0.05); ax[1,1].imshow(bnd1);  ax[1,2].imshow(mk_sky)
#         plt.show()         
      

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
        return None, None, None

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
        return None, None, None
        
            
def cloud_height(img1,img2,err1=None,err2=None,layer=0,distance=None):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image objects 
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0:
        return []
        
    if img1.max_theta != img2.max_theta:
        print("The max_theta of the two cameras is different.");
        return None, None;
    if distance is None:
        distance = 6367e3*geo.distance_sphere(img1.lat,img1.lon,img2.lat,img2.lon)  

    max_tan=np.tan(img1.max_theta*deg2rad)         
    
    im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
#     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)
    
    mask_tmpl=(img1.cm==layer+1)           
    if layer>=1:
        im1=err1+30; im2=err2+30;
        mask_tmpl=np.abs(err1)>5
#             mask_tmpl=maximum_filter(mask_tmpl,10) 
        mask_layer2=img1.cm==2
        mask_layer2=maximum_filter(mask_layer2,50) 
        mask_tmpl[mask_layer2]=False
        
    try:
        corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,ratio_thresh=0.5)       
        if np.any(corr>0):
            max_idx=np.nanargmax(corr)
            deltay,deltax=max_idx//len(corr)-img2.ny+1,max_idx%len(corr)-img2.nx+1            
            deltar=np.sqrt(deltax**2+deltay**2)            
            height=distance/deltar*img1.nx/(2*max_tan)
            print(deltay,deltax,distance,height)
            res = [height,corr.ravel()[max_idx]]
    except:
        res = [];
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
    
