from pathlib import Path
import numpy as np
import pandas as pd

def extract_features_helper( image_set ):
	img = image_set.stitched_image
	s = image_set.stitched_image

	ghi_xs = ( image_set.ghi_locs[:,1] - img.lon ) * image_set.deg2km * np.cos( image_set.ghi_locs[0,0] * np.pi / 180. )
	ghi_ys = ( img.lat - image_set.ghi_locs[:,0] ) * image_set.deg2km

	# x,y coordinates of the sky directly above each ghi loc's latlon
	img_xs = np.rint(
	    (ghi_xs - img.h * np.tan(img.sz) * np.sin(img.saz)) / img.pixel_size
	).astype(np.int32)
	img_ys = np.rint(
	    (ghi_ys + img.h * np.tan(img.sz) * np.cos(img.saz)) / img.pixel_size
	).astype(np.int32)

	columns = [
	    "lead_time", "time_str", "img.h", "img.sz",
	    "cf", "avg_r", "avg_g", "avg_b", "min_r", "min_g",
	    "min_b", "max_r", "max_g", "max_b", "rbr", "cf2",
	    "avg_r2", "avg_g2", "avg_b2", "min_r2", "min_g2",
	    "min_b2", "max_r2", "max_g2", "max_b2", "rbr2"
	]
	for i in range(len(image_set.ghi_locs)):
		features = pd.DataFrame( columns=columns )

		y = img_ys[i]
		x = img_xs[i]

		[ny, nx] = img.cm.shape

		slc = np.s_[
		    max(0,y-image_set.win_size):min(ny-1,y+image_set.win_size),
		    max(0,x-image_set.win_size):min(nx-1,x+image_set.win_size)
		]

		# if the region over the sensor is outside of the camera's view
		if img.cm[slc].size <= 0:
			print( "img cm slc" )
			print( nx, ny, x, y )
			continue

		rgb0 = img.rgb.astype(np.float32)
		rgb0[rgb0<=0] = np.nan
		rgb = np.reshape(rgb0[slc], (-1,3))
		[avg_r, avg_g, avg_b] = np.nanmean(rgb,axis=0)
		if np.isnan(avg_r):
			print( "avg r" )
			continue

		min_r, min_g, min_b = np.nanmin(rgb,axis=0)
		max_r, max_g, max_b = np.nanmax(rgb,axis=0)
		# red-blue ratio
		rbr = (avg_r - avg_b) / (avg_r + avg_b)
		# cloud fraction
		cf = np.sum(img.cm[slc]) / np.sum(rgb[:,0]>0)

		for ilt, lead_min in enumerate(image_set.lead_minutes):
			lead_steps = lead_min / image_set.interval
			# TODO use whatever layer stitch picked
			y2 = int(round(y + image_set.layer_vels[0][0]*lead_steps))
			# negate vx since the image is flipped
			x2 = int(round(x - image_set.layer_vels[0][1]*lead_steps))
			slc=np.s_[max(0, y2-image_set.win_size):min(ny-1, y2+image_set.win_size), max(0,x2-image_set.win_size):min(nx-1,x2+image_set.win_size)]

			# can't see the lead_min distant clouds
			if img.cm[slc].size <= 0:
				print( "failed lead_min {} b/c cm".format( lead_min ) )
				continue
			rgb = np.reshape(rgb0[slc], (-1,3))
			avg_r2, avg_g2, avg_b2 = np.nanmean(rgb,axis=0)
			if np.isnan(avg_r2):
				continue

			min_r2, min_g2, min_b2 = np.nanmin(rgb,axis=0)
			max_r2, max_g2, max_b2 = np.nanmax(rgb,axis=0)
			rbr2 = (avg_r2 - avg_b2) / (avg_r2 + avg_b2)
			cf2 = np.sum(img.cm[slc]) / np.sum(rgb[:,0]>0)

			time_str = image_set.timestamp.strftime(
			    "%Y%m%d%H%M%S"
			)

			features = features.append( pd.DataFrame( np.array( [[
			    lead_min, time_str, img.h, img.sz,
			    cf, avg_r, avg_g, avg_b, min_r, min_g,
			    min_b, max_r, max_g, max_b, rbr, cf2,
			    avg_r2, avg_g2, avg_b2, min_r2, min_g2,
			    min_b2, max_r2, max_g2, max_b2, rbr2
			]] ), columns=columns ) )

		features.reset_index(drop=True, inplace=True)
		print( features )
		fdir = "{}{}/".format( image_set.feature_path, image_set.day )
		if image_set.KEY:
			fdir = "{}{}/{}/".format(
			    image_set.feature_path, image_set.KEY, image_set.day
			)
		fn = "{}/{}_{}.csv".format(
		    fdir, i, image_set.timestamp
		)
			
		print( "FN: " + str(fn))
		Path( fdir ).mkdir( parents=True, exist_ok=True )
		features.to_csv(fn)
