import numpy as np
import tools.geo as geo
import tools.mncc as mncc
import tools.stat_tools_2 as st
import traceback

def cloud_height_helper( camera_dict, cam_id, images ):
	import warnings
	warnings.filterwarnings('ignore')
	
	cameras = camera_dict
	images = images

	if not cam_id in images:
		return

	neighbors = cameras[cam_id].height_group
	#print( "Height for " + cam_id )

	img1 = images[cam_id]
	img1.height = [[]]*img1.layers

	cam1 = cameras[cam_id]
	for n_id in cameras: #neighbors:
		cam2 = cameras.get( n_id, None )
		if not cam2:
			continue
		if not n_id in images:
			continue

		img2 = images[n_id]
		distance = 6367e3*geo.distance_sphere(cam1.lat, cam1.lon, cam2.lat, cam2.lon)

		for l in range(img1.layers):
			# cameras are too close to get proper parallax for high clouds
			# MAGIC CONSTANT 9
			if l >= 1 and distance < 500:
				break

			h = cloud_height_math( img1, img2, l+1, distance=distance )
			# MAGIC CONSTANT 10, 11
			if not np.isfinite( h ) or h > 20*distance or h < 0.5*distance:
				continue
			img1.height[l].append( h )

	# set each layer height to average of the votes from each neighbor
	for l in range(img1.layers):
		img1.height[l] = np.nanmean( np.array(img1.height[l]) )

def cloud_height_math( img1, img2, layer, distance=None ):
	if img1.layers < layer or img2.layers < layer or layer <= 0:
		print( "Bad layer passed to cloud_height_math" )
		return np.nan

	if img1.max_theta != img2.max_theta:
		print("The max_theta of the two cameras is different. {} {}".format(img1.c_id, img2.c_id))
		return np.nan

	if distance is None:
		distance = 6367e3*geo.distance_sphere(img1.camera.lat, img1.camera.lon, img2.camera.lat, img2.camera.lon)

	max_tan=np.tan(img1.max_theta)

	im1 = img1.red.astype(np.float32)
	im2 = img2.red.astype(np.float32)


	#mask_tmpl=(img1.cm==layer) 
	mask_tmpl=(img1.cm==1) if layer==1 else (~(img1.cm==1) & (im1>0))

	res = np.nan;
	try:
		corr = mncc.mncc( im2, im1, mask1=im2>0, mask2=mask_tmpl, ratio_thresh=0.5 )
		if np.any(corr>0):
			max_idx = np.nanargmax(corr)
			delta_y = max_idx//len(corr) - img2.ny + 1
			delta_x = max_idx%len(corr) - img2.nx + 1
			delta_r = np.sqrt(delta_x**2 + delta_y**2)
			height = distance / delta_r * img1.nx / (2 * max_tan)
			score = st.shift_2d( im1, delta_x, delta_y )
			score[score<=0] = np.nan
			score -= im2
			score = np.nanmean( np.abs(score[(im2>0)]) )
			score0 = np.abs(im2-im1)
			score0 = np.nanmean( score0[(im2>0) & (im1>0)] )

			if score0-score <= 0.3*score0:
				res = np.nan
			else:
				res = min(13000,height)
	except:
		print( 'Cannot determine cloud height.' )
		print( traceback.format_exc() )
	return res
