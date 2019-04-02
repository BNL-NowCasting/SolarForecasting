import numpy as np
import glob
import sys
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import pysolar.solar as ps
from scipy.ndimage.morphology import binary_opening
from scipy import ndimage
import ephem
import time

camera='HD815_2'
# camera='HD490'
inpath='d:/data/images/'

#####params: nx0,cy,cx,rotation,beta,azm
params = {
#           'HD815_1':[2821.0000,1440.6892,1431.0000,0.1701,0.0084,-0.2048,0.3467,-0.0041,-0.0033],\
          'HD815_1':[2821.0000,1443,1421.0000,0.1698,-0.0137,-2.4165,0.3463,-0.0031,-0.0035],\
#           'HD815_2':[2821.0000,1423.9111,1459.000,0.0311,-0.0091,0.1206,0.3455,-0.0035,-0.0032],\
          'HD815_2':[2821.0000,1424,1449.0000,0.0310,0.0122,2.2050,0.3459,-0.0043,-0.0027],\
          'HD490':[2843.0000,1472.9511,1482.6685,0.1616,0.0210,-0.5859,0.3465,-0.0043,-0.0030], \
          'HD17':[2817.249,1478.902,1462.346,-0.099,0.012,0.867,2]}

nx0=ny0=params[camera][0]
nr0=(nx0+ny0)/4
xstart=int(params[camera][2]-nx0/2+0.5); ystart=int(params[camera][1]-ny0/2+0.5)
nx0=int(nx0+0.5); ny0=int(ny0+0.5)
roi=np.s_[ystart:ystart+ny0,xstart:xstart+nx0]

gatech = ephem.Observer(); 
gatech.lat, gatech.lon = '40.88', '-72.87'
moon=ephem.Moon() 

# ref=np.load(camera+'.npy').item();

####HD17
# ref['20180227214355']=[1188,305]
# ref['20180227221355']=[1294,363]

####HD815_2
# ref['20180227234029']=[1623,598]
# ref['20180227231038']=[1553,501]
# ref['20180227221038']=[1381,329]
# ref['20180227224022']=[1471,409]

####HD815_1
# ref['20180227231008']=[1689,494]
# ref['20180227221025']=[1548,300]
# ref['20180227233929']=[1743,598]
# ref['20180227223952']=[1623,391]

####HD490
# ref['20180227215855']=[1534,302]
# ref['20180227222934']=[1618,394]
# ref['20180227225857']=[1687,490]
# ref['20180227232829']= [1746, 593]
# ref['20180227235829']=[1795, 706]
 
# np.save(camera,ref);
# sys.exit()

# print(ref)
ref={}
flist=[]
# flist=sorted(glob.glob(inpath+camera+'/'+camera+'_201803101???'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_20180309???5'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802272[1-3][1,4]3[2,3]'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802270[0-9][2,5]8[2,3,4]'+'*jpg'));
flist+=sorted(glob.glob(inpath+camera+'/'+camera+'_201802280[0-9][1,4]029'+'*jpg'))  
for f in flist:     
#     print(f)
    doy=f[-18:-4]
    
    gatech.date = datetime.strptime(doy,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
    moon.compute(gatech) 
    sz=np.pi/2-moon.alt; saz=(params[camera][3]+moon.az-np.pi)%(2*np.pi);     
     
    rref=np.sin(sz/2)*np.sqrt(2)*nr0
    xref,yref=nx0//2+rref*np.sin(saz),ny0//2+rref*np.cos(saz)
    xref=int(xref); yref=int(yref)
    
    img=plt.imread(f).astype(np.float32);
    img=img[roi]
#     plt.figure(); plt.imshow(img/255);
    img=(0.8*img[:,:,2]+0.2*img[:,:,0])
#     img=np.nanmean(img,axis=2); #plt.figure(); plt.imshow(img)
    
    x1,x2=max(0,xref-150),min(nx0,xref+150)
    y1,y2=max(0,yref-150),min(ny0,yref+150)
    img=img[y1:y2,x1:x2]
    
#     img_m=st.rolling_mean2(img,11)
#     thresh=img_m>200
    
    img_m=st.rolling_mean2(img,71,fill=0)
    img_m=img-img_m; img_m-=np.nanmean(img_m)
    std=np.nanstd(img_m)
    thresh=img_m>4*std
    
#     t=time.time()
    s = ndimage.generate_binary_structure(2,2) # iterate structure
    labeled_mask, cc_num = ndimage.label(thresh,s)
    try:
        thresh = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
    except:
        continue
    if  np.sum(thresh)<=9:
        print('Moon not found.')
        continue
    
    # Find coordinates of thresholded image
    [y,x] = np.nonzero(thresh)[:2];
    filter=(np.abs(y-np.mean(y))<2.5*np.std(y)) & (np.abs(x-np.mean(x))<2.5*np.std(x))
    # Find average
    xmean = xstart+x1+np.nanmean(x[filter]);
    ymean = ystart+y1+np.nanmean(y[filter]);
    if xmean>200 and ymean>100:
        ref[doy]=[ymean,xmean]
        print('\''+doy+'\': [',int(ymean+0.5),int(xmean+0.5),'], ')
#     fig,ax=plt.subplots(1,3,sharex=True,sharey=True); 
#     ax[0].imshow(img);    ax[1].imshow(img_m);   ax[2].imshow(thresh);         

np.save(camera,ref)

# deg2rad=np.pi/180
# cameras=['HD815_1','HD815_2']
# rot={'HD1':9.5*deg2rad, 'HD2':1.5*deg2rad};
# dxy0={'HD815_1':(30,5), 'HD815_2':(55,10)};


# ####set up paths, constantsand initial parameters
# # inpath='d:/data/HD/'
# inpath='d:/data/images/'
# 
# # doys=['20180227065311','20180227013131']
# doys=['2018022709','2018022708','2018022707','2018022706','2018022705','2018022704','2018022703','2018022702','2018022701','2018022700']
# 
# ny0,nx0=2944,2944
# x0,y0=np.meshgrid(np.linspace(-nx0//2,nx0//2,nx0),np.linspace(-ny0//2,ny0//2,ny0)); 
# r0=np.sqrt(x0**2+y0**2);
# mask=r0>1300            
# 
# cnt=0;
# im=np.zeros((2,ny0,nx0,3))
# 
# for doy in doys:
#     for camera in cameras:
#         dx0,dy0 = dxy0[camera]        
#         for f in sorted(glob.glob(inpath+camera+'/'+camera+'*'+doy+'*jpg')):    
#             print(f)    
#             ######read the image to array
#             tmp=plt.imread(f).astype(np.float32); tmp[tmp<150]=0
#             im[cnt%2]+=tmp 
#     #         im0=st.shift_2d(im0,-dx0,-dy0);  
#     #     
#     # #         if camera=='HD815_1':
#     # #             im0[np.isnan(im0)]=0
#     # #             im0=rotate(im0[:,:],-7.7, reshape=False)
#     #         if cnt%2==0:
#     #             fig,ax=plt.subplots(1,2,sharex=True,sharey=True); 
#     #         ax[cnt%2].imshow(im0[:,:,:]/255);
#             cnt+=1;
#             break
# 
# im[:,mask,:]=0
# np.save('HD815',im)


# dx0,dy0=(30,5)
# dx1,dy1=(55,10)
# im=np.load('HD815.npy')
# im0,im1=im[0,:,:,0],im[1,:,:,0]
# # im0=st.shift_2d(im0,-dx0,-dy0,constant=0);
# # im1=st.shift_2d(im1,-dx1,-dy1,constant=0);
# # im0=rotate(im0,-8, reshape=False); #im0[~(im0>=0)]=0
# 
# # import mncc
# # corr=mncc.mncc(im0,im1,mask1=im0>20,mask2=im1>20)
# # fig,ax=plt.subplots(1,2,sharex=True,sharey=True);
# # ax[0].imshow(im0[:,:]);
# # ax[1].imshow(im1[:,:]);
# 
# plt.figure(); plt.imshow(im0[600:-900,700:-100],vmin=0,vmax=255);  ###undistored image    
# plt.figure(); plt.imshow(im1[600:-900,700:-100],vmin=0,vmax=255);


    
