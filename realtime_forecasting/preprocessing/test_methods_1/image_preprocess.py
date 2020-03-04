from datetime import datetime
import ephem
from matplotlib import pyplot as plt
import numpy as np
import os
from skimage.morphology import remove_small_objects
import stat_tools_0 as st

BND_WIN = 30;
BND_RED_THRESH, BND_RBR_THRESH  = 2, 0.012
DRED_THRESH, DRBR_THRESH = 150, 157
STD_RED_THRESH, STD_RBR_THRESH = 1.2, 0.012

def undistort_helper( image, _, rgb=True, day_only=True ):
    """
    Undistort the raw image, set rgb, red, rbr, cos_g
    Input: rgb and day_only flags
    Output: rgb, red, rbr, cos_g will be specified.
    """
    #####get the image acquisition time, this need to be adjusted whenever the naming convention changes
    image.time=datetime.strptime(image.fn[-18:-4],'%Y%m%d%H%M%S');
    gatech = ephem.Observer();
    gatech.date = image.time.strftime('%Y/%m/%d %H:%M:%S')
    gatech.lat, gatech.lon = str(image.camera.lat),str(image.camera.lon)
    sun=ephem.Sun()  ; sun.compute(gatech);
    sz = np.pi/2-sun.alt;
    image.sz = sz
    #if day_only and sz>75*deg2rad:
     #   return

    saz = np.pi+sun.az; saz=saz % (2*np.pi);
    image.saz = saz

    try:
        im0=plt.imread(image.fn);
    except:
        print('Cannot read file:', image.fn)
        return None
    im0=im0[image.camera.roi]
    im0[~image.camera.valid0,:]=0

    cos_sz=np.cos(sz)
    cos_g=cos_sz*np.cos(image.camera.theta0)+np.sin(sz)*np.sin(image.camera.theta0)*np.cos(image.camera.phi0-saz);

    red0=im0[:,:,0].astype(np.float32); red0[red0<=0]=np.nan
    rbr0=(red0-im0[:,:,2])/(im0[:,:,2]+red0)

    if np.nanmean(red0[(cos_g>0.995) & (red0>=1)])>30:
        mk=cos_g>0.98
        red0[mk]=np.nan
        rbr0[mk]=np.nan

#     rbr0[rbr0>0.1]=0.1
    rbr2=rbr0.copy()
    rbr2-=st.rolling_mean2(rbr0,int(image.camera.nx//4),ignore=np.nan)
    rbr0 -= st.rolling_mean2(rbr0,int(image.camera.nx//6.666),ignore=np.nan)
    mk=rbr0>0.005; rbr0[mk]=np.maximum(rbr0[mk],rbr2[mk])
#     mk=rbr0<-0.005; rbr0[mk]=np.minimum(rbr0[mk],rbr2[mk])
    rbr=st.fast_bin_average2(rbr0,image.camera.weights);
    rbr=st.fill_by_mean2(rbr,7, mask=(np.isnan(rbr)) & image.camera.valid)
    rbr[rbr>0.08]=0.08; rbr[rbr<-0.08]=-0.08;
    rbr=(rbr+0.08)*1587.5+1;
    rbr[image.camera.invalid]=0
    image.rbr=rbr.astype(np.uint8)
    red0 -= st.rolling_mean2(red0,int(image.camera.nx//6.666),ignore=np.nan)
    red=st.fast_bin_average2(red0,image.camera.weights);
    red=st.fill_by_mean2(red,7, mask=(np.isnan(red)) & image.camera.valid)
    red[red>50]=50; red[red<-50]=-50
    red=(red+50)*2.54+1;
    red[image.camera.invalid]=0;
    image.red=red.astype(np.uint8)

    if rgb:
        im=np.zeros((image.camera.ny,image.camera.nx,3),dtype=im0.dtype)
        for i in range(3):
            im[:,:,i]=st.fast_bin_average2(im0[:,:,i],image.camera.weights);
            im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=(im[:,:,i]==0) & (image.camera.valid))
#             im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=np.isnan(red))
        im[(red<=0) | (image.camera.invalid)]=0
        image.rgb=im


def cloud_mask_helper( image ):
    """
    Set cloud mask
    Input: None
    Output: cloud mask will be specified.
    """
    cam = image.camera

    cos_s=np.cos(image.sz); sin_s=np.sin(image.sz)
    cos_sp=np.cos(image.saz); sin_sp=np.sin(image.saz)
    cos_th=cam.cos_th; sin_th=np.sqrt(1-cos_th**2)
    cos_p=cam.cos_p; sin_p=cam.sin_p
    cos_g=cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p);  ###cosine of the angle between illumination and view directions
#     image.cos_g=((1+cos_g)*127.5).astype(np.uint8);

    #####determine the cloud edges
    red0=image.rgb[:,:,1].astype(np.float32); red0[cos_g>0.98]=0
    rbr0=(image.rgb[:,:,0].astype(np.float32)-image.rgb[:,:,2])/(image.rgb[:,:,2]+image.rgb[:,:,0].astype(np.float32));
    red=image.red.astype(np.float32); red[cos_g>0.98]=0
    rbr=image.rbr.astype(np.float32); rbr[(rbr<=0) | (cos_g>0.98)]=np.nan;

    mred=st.rolling_mean2(red0,BND_WIN,ignore=0); mred[cos_g>0.95]=np.nan
    vred=st.rolling_mean2(red0**2,BND_WIN,ignore=0)-mred**2; #vred[vred>50]=50
    mrbr=st.rolling_mean2(rbr0,BND_WIN,ignore=0)
    vrbr=st.rolling_mean2(rbr0**2,BND_WIN,ignore=0)-mrbr**2; #vrbr[vrbr>50]=50
#     bnd = (vred>BND_RED_THRESH)
    bnd = ((100*np.sqrt(vred)/mred>BND_RED_THRESH) & (np.sqrt(vrbr)>BND_RBR_THRESH/3)) | ((np.sqrt(vrbr)>BND_RBR_THRESH) & (100*np.sqrt(vred)/mred>BND_RED_THRESH/2))
#     bnd = ((100*np.sqrt(vred)/mred>BND_RED_THRESH)) | ((np.sqrt(vrbr)>BND_RBR_THRESH))
    print('bnd:',np.sum(bnd),np.sum(red>0))

    ####classify the cloud boundary pixels into cld or sky, which will serve as seeds for further growth
    mk_cld=(red>=DRED_THRESH) | (rbr>=DRBR_THRESH) | ((red+rbr>DRED_THRESH+DRBR_THRESH-20) & (red>DRED_THRESH-5));
    mk_sky=(rbr<=255-DRBR_THRESH+5) & (rbr>0);
    mk_cld=remove_small_objects(mk_cld, min_size=500, connectivity=4)
    mk_sky=remove_small_objects(mk_sky, min_size=500, connectivity=4)
#     mk_sky=morphology.binary_dilation(mk_sky,np.ones((3,3)));  mk_sky[mk_cld]=False;
#     mk_cld=morphology.binary_dilation(mk_cld,np.ones((3,3)));  mk_cld[mk_sky]=False;

#     bins=np.arange(0,0.96,0.03);  bcs=0.5*(bins[:-1]+bins[1:])
#     mk=(mk_sky) & (red>0);
#     sky_rbr = st.bin_average(rbr0[mk],cos_g[mk],bins);
#     coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
# #     mk_cld=rbr0>=f_sky(cos_g)-0.0
#     cos_g[cos_g<0]=0
#     rbr2=rbr0.copy()-np.polyval(coeff,cos_g)
    if np.sum(mk_sky)>=1000:
        bins=np.arange(50,250,5);  bcs=0.5*(bins[:-1]+bins[1:])
        mk=(mk_sky) & (red>0);
        sky_rbr = st.bin_average(rbr0[mk],red0[mk],bins);
        sky_rbr=st.rolling_mean(sky_rbr,20,fill_nan=True)
#         plt.figure(); plt.plot(bcs,sky_rbr); plt.show()
        coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
        rbr2=rbr0-np.polyval(coeff,red0)
       #      f_sky = interpolate.interp1d(bcs,sky_rbr,fill_value='extrapolate')
       #      rbr2=rbr0-f_sky(red0)
    else:
        image.cm=red>0
        image.layers=1
        return

    bins=np.arange(-0.1,0.951,0.03);  bcs=0.5*(bins[:-1]+bins[1:])
    mk=(mk_sky) & (red>0);
    sky_rbr = st.bin_average(rbr2[mk],cos_g[mk],bins);
    if np.sum(~np.isnan(sky_rbr))<=1:
        thr_sky=np.zeros_like(red)+np.nanmean(rbr2[mk]);
    else:
        sky_rbr=st.rolling_mean(sky_rbr,10,fill_nan=True)
        coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
        print('sky coef:',coeff)
        if (coeff[0]<=0):
            thr_sky=np.zeros_like(red)+np.nanmean(rbr2[mk]);
        else:
            thr_sky=np.polyval(coeff,cos_g)
            thr_sky[cos_g>0.95]=np.polyval(coeff,0.95)
            thr_sky[cos_g<-0.1]=np.polyval(coeff,-0.1)

    mk=(mk_cld) & (red>0);
    sky_rbr = st.bin_average(rbr2[mk],cos_g[mk],bins);
    sky_rbr=st.rolling_mean(sky_rbr,10,fill_nan=True)
    coeff=np.polyfit(bcs[sky_rbr>-1],sky_rbr[sky_rbr>-1],1)
    print('cloud coef:',coeff)
    if (coeff[0]<=0):
        thr_cld=np.zeros_like(red)+np.nanmean(rbr2[mk]);
    else:
        thr_cld=np.polyval(coeff,cos_g)

#     mk_cld[rbr2>0.4*thr_cld+0.6*thr_sky]=True; mk_cld[red<=0]=False

    image.cm=mk_cld;
    max_score=0;
    for w in np.arange(-0.1,1.51,0.1):
        cld=mk_cld.copy()
        cld[rbr2>w*thr_cld+(1-w)*thr_sky]=True; cld[red<=0]=False
        mcld=st.rolling_mean2(cld.astype(np.float32),3,mask_ignore=red<=0,fill_nan=True);
        bnd_e=((mcld>0.2) & (mcld<0.95))
        bnd_e=remove_small_objects(bnd_e, min_size=100, connectivity=4)
        score=np.sum(bnd_e & bnd)/np.sum(bnd_e)
        print(w,np.sum(bnd_e & bnd), score)
        if score>max_score:
            max_score=score
            image.cm=cld;

#     image.cm=mk_cld;
    image.layers=1;
    #fig,ax=plt.subplots(2,3,sharex=True,sharey=True);
    #ax[0,0].imshow(image.rgb); ax[0,1].imshow(rbr0,vmin=-0.15,vmax=0.1); ax[0,2].imshow(image.cm);
    #ax[1,0].imshow(bnd); ax[1,1].imshow(mk_sky);  ax[1,2].imshow(mk_cld)
    #plt.show()

