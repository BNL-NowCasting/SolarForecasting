# /home/tchapman/solar_code/20180920/preprocessing_shm/cloud_height.py~
from collections import deque

from matplotlib import pyplot as plt
import numpy as np

from scipy.ndimage import morphology
from scipy.ndimage.filters import maximum_filter
from skimage.morphology import remove_small_objects

import tools.geo as geo
import tools.mncc as mncc
import tools.stat_tools_0 as st

import traceback

# MAGIC CONSTANT
MAX_INTERVAL = 180

def cloud_height_helper( image_set, cam_id):
	cameras = image_set.camera_dict
	images = image_set.images

	if not cam_id in images:
		return

	neighbors = cameras[cam_id].height_group
	print( "Height for " + cam_id )

	img1 = images[cam_id]
	img1.height = [[]] * img1.layers


	q1 = deque()
	fft1 = deque()
	err1 = None
	flag1 = [-1]

	img1.v = []
	preprocess( cameras[cam_id], img1, q1, err1, fft1, flag1 )
	for n_id in neighbors:
		cam = cameras.get( n_id, None )
		if not cam:
			continue
		if not n_id in images:
			continue

		img2 = images[n_id]
		q2 = deque()
		fft2 = deque()
		err2 = None
		flag2 = [-1]

		preprocess(cameras[n_id], img2, q2, err2, fft2, flag2)

#	       # cloud height for previous frame
#	       h = cloud_height( q1[0],err1,q2[0],err2 )
		heights = cloud_height_math( q1[1], err1, q2[1], err2 )
		print( "Estimated height with neighbor {}: {}".format(n_id, heights) )
		for l,h in enumerate(heights):
			img1.height[l] += [h]

	for l in range(img1.layers):
		img1.height[l] = np.nanmedian( img1.height[l], axis=0 )
		img1.v[l] = [[
		    np.nanmedian([vs[l][0] if l < len(vs) else np.nan for vs in img1.v]),
		    np.nanmedian([vs[l][1] if l < len(vs) else np.nan for vs in img1.v])
		]]

def cloud_height_math(img1,err1,img2,err2,distance=None):
    """
    Determine the cloud height for each cloud layer in img1
    Input: Two image object
    Output: Cloud height, and max correlation
    """
    if img1.layers<=0:
        return []

    cam1=img1.camera; cam2=img2.camera

    res= [None] * img1.layers

    if cam1.max_theta != cam2.max_theta:
        print("The max_theta of the two cameras is different.");
        return res;
    if distance is None:
        distance = 6367e3*geo.distance_sphere(cam1.lat,cam1.lon,cam2.lat,cam2.lon)

    max_tan=np.tan(cam1.max_theta)

    im1=img1.red.astype(np.float32); im2=img2.red.astype(np.float32)
#     im1=img1.rgb[:,:,0].astype(np.float32); im2=img2.rgb[:,:,0].astype(np.float32)

    for ilayer in range(img1.layers):
        mask_tmpl=(img1.cm==ilayer+1)
        if ilayer>=1:
            im1=err1+30; im2=err2+30;
            mask_tmpl=np.abs(err1)>5
#             mask_tmpl=maximum_filter(mask_tmpl,10)
        elif img1.layers>=2:
            mask_layer2=img1.cm==2
            mask_layer2=maximum_filter(mask_layer2,50)
            mask_tmpl[mask_layer2]=False
        try:
            corr=mncc.mncc(im2,im1,mask1=im2>0,mask2=mask_tmpl,ratio_thresh=0.5)
            if np.any(corr>0):
                max_idx=np.nanargmax(corr)
                deltay,deltax=max_idx//len(corr)-cam2.ny+1,max_idx%len(corr)-cam2.nx+1
#                print(deltay,deltax)
                deltar=np.sqrt(deltax**2+deltay**2)
                height=distance/deltar*cam1.nx/(2*max_tan)
                res[ilayer] = height
        except:
            print( traceback.format_exc() )
            res[ilayer] = None
#     print(np.nanmax(corr),height,deltay, deltax)
    return res

def preprocess(camera, img, q, err, fft, flag, convolver=mncc.Convolver((2000,2000),(2000,2000), threads=4, dtype=np.float32)):
    if img.previous_image:
        q.append( img.previous_image )
    q.append(img)

    if len(q)<=1:
        return
    if (q[-1].timestamp-q[-2].timestamp).seconds>=MAX_INTERVAL:
        q.popleft();
        return;
#####cloud motion for the dominant layer    
    for ii in range(len(fft)-2,0):
        im=q[ii].red.astype(np.float32);
        mask = im>0; #im[~mask]=0        
        fft.append(convolver.FFT(im,mask,reverse=flag[0]>0));
        flag[0] *= -1
    vy,vx,max_corr = cloud_motion_fft(convolver,fft[-2],fft[-1],ratio=0.7);
    if vx is None or vy is None: #####invalid cloud motion
        q.popleft(); fft.popleft();
        return
    vy*=flag[0]; vx*=flag[0];
    fft.popleft();
#     print(f[-18:-4]+',  First layer:',max_corr,vy,vx) 

    #q[-2].v+=[[vy,vx]]
    if len(q[-1].v) == 0:
        q[-1].v.append( [] )
    q[-1].v[0].append([vy,vx])

    red1=st.shift_2d(q[-1].rgb[:,:,0].astype(np.float32),-vx,-vy); red1[red1<=0]=np.nan
    red2=q[-2].rgb[:,:,0].astype(np.float32); red2[red2<=0]=np.nan #red2-=np.nanmean(red2-q[-1].rgb[:,:,0])
    er=red2-red1;   ###difference image after cloud motion adjustment
    er[(red1==0)|(red2==0)]=np.nan;
    a=er.copy(); a[a>0]=0; er-=st.rolling_mean2(a,500);
    err_2=err;
    err = (-st.shift_2d(er,vx,vy))
    if err_2 is None:
        return
#     if vy**2+vx**2>=50**2:  ######The motion of the dominant layer is fast, likely low clouds. Do NOT trigger the second layer algorithm 
#         v2+=[[np.nan,np.nan]]
#         err.popleft(); 
#         continue

#####process the secondary layer 
    ert=er+err_2;
#     cm2=(er>15) | (er_p>15); cm2=remove_small_objects(cm2, min_size=500, connectivity=4);  
    scale=red2/np.nanmean(red2); nopen=max(5,int(np.sqrt(vx**2+vy**2)/3))
    cm2=(ert>15*scale) & (q[-2].cm);
    cm2=morphology.binary_opening(cm2,np.ones((nopen,nopen)))
    cm2=remove_small_objects(cm2, min_size=500, connectivity=4);
    sec_layer=np.sum(cm2)/len(cm2.ravel())  ###the amount of clouds in secondary layer
    if sec_layer<5e-3:   ###too few pixels, no need to process secondary cloud layer
#         v2+=[[np.nan,np.nan]]
        return

#####cloud motion for the secondary layer   
    mask2=np.abs(err_2)>5;
    mask2=remove_small_objects(mask2, min_size=500, connectivity=4)
    mask2=filters.maximum_filter(mask2,20)
    vy,vx,max_corr = cloud_motion(err,err_2,mask1=None,mask2=mask2, ratio=None, threads=4)
    if vx is None or vy is None:
        return
    #q[-2].v+=[[vy,vx]]
    if len(q[-1].v) == 1:
        q[-1].v.append( [] )
    q[-1].v[1].append([vy,vx])
#     print(f[-18:-4]+',  second layer:',max_corr,vy,vx) 

    if np.abs(vy-q[-1].v[0][0][0])+np.abs(vx-q[-1].v[0][0][1])>30:
#     if np.abs(vy)+np.abs(vx)>0:
    #####obtain the mask for secondar cloud layer using a watershed-like algorithm    
        mred=q[-2].rgb[:,:,0].astype(np.float32)-st.fill_by_mean2(q[-2].rgb[:,:,0],200,mask=~cm2)
        mrbr=q[-2].rbr-st.fill_by_mean2(q[-2].rbr,200,mask=~cm2)
        merr=st.rolling_mean2(ert,200,ignore=np.nan); var_err=(st.rolling_mean2(ert**2,200,ignore=np.nan)-merr**2)
    #     mk=(np.abs(q[-2].rgb[:,:,0].astype(np.float32)-mred)<3) & ((total_err)>-2) & (np.abs(q[-2].rbr-mrbr)<0.05)
        mk=(np.abs(mred)<3) & (ert>-15) & (np.abs(mrbr)<0.05) & (var_err>20*20)
        cm2=morphology.binary_opening(mk|cm2,np.ones((nopen,nopen)))  ####remove line objects produced by cloud deformation
        cm2=remove_small_objects(cm2, min_size=500, connectivity=4)
        #q[-2].layers=2; q[-2].cm[cm2]=2;  #####update the cloud mask with secondary cloud layer 
        q[-1].layers=2; q[-1].cm[cm2]=2;  #####update the cloud mask with secondary cloud layer 

    #     fig,ax=plt.subplots(2,2, sharex=True,sharey=True);  ####visualize the cloud masking results
    #     ax[0,0].imshow(q[-2].rgb); ax[0,1].imshow(q[-2].cm)
    #     ax[1,0].imshow(st.shift_2d(q[-1].rgb,-vx,-vy))  
    #     ax[1,1].imshow(er,vmin=-25,vmax=25)          
    return;

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

