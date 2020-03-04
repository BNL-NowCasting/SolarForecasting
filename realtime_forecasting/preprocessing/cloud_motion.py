import numpy as np

from scipy.ndimage import morphology
from scipy.ndimage.filters import maximum_filter,gaussian_filter,laplace,median_filter
from skimage.morphology import remove_small_objects

import tools.mncc as mncc
import tools.stat_tools_2 as st
import traceback

def cloud_motion_helper(image_set, img, prev_img):
	img.v = [[np.nan, np.nan]]

	r1 = prev_img.red.astype(np.float32)
	r1[r1<=0] = np.nan
	r2 = img.red.astype(np.float32)
	r2[r2<=0] = np.nan
	err0 = r2-r1

	dif = np.abs(err0)
	# MAGIC CONSTANT 5,6,7,8
	dif = st.rolling_mean2(dif,20)
	semi_static = (abs(dif)<10) & (r1-127>100)
	semi_static = morphology.binary_closing(semi_static, np.ones((10,10)))
	semi_static = remove_small_objects(semi_static, min_size=200, in_place=True)
	img.rgb[semi_static] = 0
	r2[semi_static] = np.nan

	if np.sum(img.cm>0) < 2e-2*img.nx*img.ny:
		# no clouds
		img.layers=0;
	else:
		dilated_cm = morphology.binary_dilation( img.cm, np.ones((15,15)) )
		dilated_cm &= (r2>0)
		[vy,vx,max_corr] = cloud_motion_math(
		    image_set, r1, r2, 
		    mask1=r1>0, mask2=dilated_cm, ratio=0.7, threads=4
		)

		if np.isnan(vy):
			img.layers=0;
		else:
			img.v = [[vy,vx]]
			img.layers=1

def cloud_motion_math(image_set, im1, im2, mask1=None, mask2=None, ratio=0.7, threads=1):
	"""
	Determine cloud motion 
	Input: Images and masks for two frames
	Output: Cloud motion vector, and max correlation
	"""
	# use this routine if the inputs are raw images   
	ny,nx=im2.shape
	try:
		corr = mncc.mncc( im1, im2, mask1=mask1, mask2=mask2, ratio_thresh=ratio, threads=threads )
		max_idx = np.nanargmax(corr)
		vy = max_idx//len(corr) - ny + 1
		vx = max_idx%len(corr) - nx + 1
		return [vy, vx, corr.ravel()[max_idx]]
	except:
		print( "Caught: " + str(traceback.format_exc()) )
		return [np.nan, np.nan, np.nan]
