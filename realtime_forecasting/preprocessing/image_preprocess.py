from datetime import datetime, timedelta, timezone
import ephem
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.ndimage.filters import maximum_filter, gaussian_filter, laplace, median_filter
from skimage.morphology import remove_small_objects
import tools.stat_tools_2 as st
import traceback

def image_init(self, camera, fn, previous_image, config, reprocess={}, KEY=""):
	bn = os.path.basename( fn )
	self.timestamp = datetime.strptime( bn[-18:-4], "%Y%m%d%H%M%S" )
	self.day = bn[-18:-10]
	site = config["site_id"]
	self.pickle_path = config["paths"][site]["pickle_path"]
	self.pic_dir = "{}preprocessed_imgs/{}/".format(
		self.pickle_path, self.day
	)
	self.pic_path = "{}{}.pkl".format(
	    self.pic_dir, os.path.basename(fn)
	)

	if KEY:
		old_pic_path = self.pic_path
		self.pic_dir = "{}{}/preprocessed_imgs/{}/".format(
			self.pickle_path, KEY, self.day
		)
		self.pic_path = "{}{}.pkl".format(
		    self.pic_dir, os.path.basename(fn)
		)

	if os.path.exists( self.pic_path ) and not reprocess["image"]:
		with open( self.pic_path, 'rb' ) as fh:
			self.__dict__ = pickle.load(fh).__dict__
		return

	if KEY:
		if os.path.exists( old_pic_path ) and not reprocess["image"]:
			with open( old_pic_path, 'rb' ) as fh:
				self.__dict__ = pickle.load(fh).__dict__
			return
	print( "Not loading pickled image" )

	self.fn = fn
	self.camera = camera
	self.previous_image = previous_image
	# I pickle everything, so we can't have this recursing back
	# to the first instance processed.
	if self.previous_image:
		self.previous_image.previous_image = None

	self.c_id = camera.c_id
	self.nx = camera.nx
	self.ny = camera.ny
	self.lon = camera.lon
	self.lat = camera.lat
	self.max_theta = camera.max_theta
	self.t_local = None
	self.fn = fn
	self.layers = 0
	self.v = []
	self.height = []
	self.rgb = None
	self.sz = None
	self.saz = None
	# spatial structure/texture of the red image, 
	# used by the cloud motion and height routines
	self.red = None
	self.cm = None # cloud mask

	self.undistorted = False
	self.cloud_masked = False
	self.skip_bc_night = False
	self.error = ""

	self.dump_image()

def undistort_helper( image, cam, rgb=True, day_only=True ):
	# get image acquisition time
	# ephem only accepts UTC times and internally translates them
	# based on lat and lon
	timestamp = datetime.strptime(
	    image.fn[-18:-4], '%Y%m%d%H%M%S'
	).replace(tzinfo=timezone.utc)

	gatech = ephem.Observer();
	gatech.date = timestamp.strftime('%Y/%m/%d %H:%M:%S')
	gatech.lat = str(image.lat)
	gatech.lon = str(image.lon)
	# print( ephem.localtime( gatech.date ).ctime() )

	sun = ephem.Sun()
	sun.compute(gatech);

	# angle from the vertical to the sun
	image.sz = np.pi / 2 - sun.alt;
	image.saz = (np.pi + sun.az) % (2*np.pi)

#	# TODO: I feel like this doesn't quite match up with the shift
#	#       from day to night image collecting
#	# MAGIC CONSTANT 1
#	#if day_only and sun.alt < np.pi / 12.:
#	if day_only and sun.alt < np.pi / 20.:
#		print( "IS NIGHT and day_only" )
#		image.skip_bc_night = True
#		return

	try:
		im0 = plt.imread(image.fn);
	except Exception as e:
		print('Cannot read file:', image.fn)
		print( traceback.format_exc() )
		image.error = str(e)
		return None

	im0 = im0[cam.roi]

	cos_sz = np.cos(image.sz)
	cos_g = cos_sz*np.cos(cam.theta0) + np.sin(image.sz)*np.sin(cam.theta0)*np.cos(cam.phi0 - image.saz)

	red0 = im0[:,:,0].astype(np.float32)
	red0[red0<=0] = np.nan

	# MAGIC CONSTANT 2, 3, 4
	if np.nanmean( red0[(cos_g>0.995) & (red0>=1)] ) > 230:
		mk = cos_g>0.98
		red0[mk] = np.nan

	image.sun_x = round(
	    image.nx * ( 1 + np.tan(image.sz)*np.sin(image.saz) / cam.max_tan )
	)

	image.sun_y = round(
	    image.ny * ( 1 + np.tan(image.sz)*np.cos(image.saz) / cam.max_tan )
	)

	invalid = ~cam.valid

	red = st.fast_bin_average2( red0,cam.weights )
	red = st.fill_by_mean2( red,7, mask=(np.isnan(red) & cam.valid) )
	red[invalid] = np.nan
	red -= st.rolling_mean2( red, int(image.nx//6.666) )
	red[red>50] = 50
	red[red<-50] = -50
	red = (red + 50)*2.54 + 1
	red[invalid] = 0

	image.red = red.astype(np.uint8)

	if rgb:
		im = np.zeros( (image.ny,image.nx,3), dtype=im0.dtype )
		for i in range(3):
			im[:,:,i] = st.fast_bin_average2(im0[:,:,i], cam.weights);
			im[:,:,i] = st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=(im[:,:,i]==0) & (cam.valid))
			#im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, ignore=0, mask=np.isnan(red))   
		im[image.red<=0] = 0
		image.rgb = im

def cloud_mask_helper( image ):
	image0 = image.previous_image
	cam = image.camera

	cos_s = np.cos(image.sz)
	sin_s = np.sin(image.sz)
	cos_sp = np.cos(image.saz)
	sin_sp = np.sin(image.saz)
	cos_th = cam.cos_th
	sin_th = np.sqrt(1-cos_th**2)
	cos_p = cam.cos_p
	sin_p = cam.sin_p
	# cosine of the angle between illumination and view directions
	cos_g = cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p)

	r0 = image0.rgb[...,0].astype(np.float32)
	r0[r0<=0] = np.nan
	r1 = image.rgb[...,0].astype(np.float32)
	r1[r1<=0] = np.nan
	rbr_raw = (r1-image.rgb[:,:,2])/(image.rgb[:,:,2]+r1)
	rbr = rbr_raw.copy()
	rbr -= st.rolling_mean2(rbr,int(image.nx//6.666))
	rbr[rbr>0.08] = 0.08
	rbr[rbr<-0.08] = -0.08
	rbr = (rbr+0.08)*1587.5+1 # scale rbr to 0-255
	mblue = np.nanmean(image.rgb[(cos_g<0.7) & (r1>0) & (rbr_raw<-0.01),2].astype(np.float32))

	err = r1-r0
	err -= np.nanmean(err)
	dif = st.rolling_mean2(abs(err),100)
	err = st.rolling_mean2(err,5)
	dif2 = maximum_filter(np.abs(err),5)

	sky = (rbr<126) & (dif<1.2)
	sky |= dif<0.9
	sky |= (dif<1.5) & (err<3) & (rbr<105)
	sky |= (rbr<70)
	sky &= (image.red>0)
	cld = (dif>2) & (err>4)
	cld |= (image.red>150) & (rbr>160) & (dif>3)
	cld |= (rbr>180) # clouds with high rbr
	cld[cos_g>0.7] |= (image.rgb[cos_g>0.7,2]<mblue) & (rbr_raw[cos_g>0.7]>-0.01) #dark clouds
	cld &= dif>3

	total_pixel = np.sum(r1>0)

	min_size = 50*image.nx/1000
	cld = remove_small_objects(cld, min_size=min_size, connectivity=4, in_place=True)
	sky = remove_small_objects(sky, min_size=min_size, connectivity=4, in_place=True)

	ncld = np.sum(cld)
	nsky = np.sum(sky)
#	print(ncld/total_pixel,nsky/total_pixel);
	if (ncld+nsky) <= 1e-2*total_pixel:
		self.error = "ncld + nsky is less than 1% of the image"
		return;
	elif (ncld < nsky) & (ncld <= 2e-2*total_pixel):
		# clear sky
		image.cm = cld.astype(np.uint8)
		image.layers = 1
		return
	elif (ncld > nsky) & (nsky <= 2e-2*total_pixel):
		# overcast sky
		image.cm = ((~sky)&(r1>0)).astype(np.uint8)
		image.layers = 1
		return

	max_score = -np.Inf
	x0 = -0.15;
	ncld = 0.25*nsky+0.75*ncld
	nsky = 0.25*ncld+0.75*nsky
#	ncld=max(ncld,0.05*total_pixel); nsky=max(nsky,0.05*total_pixel)
	for slp in [0.1,0.15]:
		offset = np.zeros_like(r1)
		mk = cos_g<x0
		offset[mk] = (x0-cos_g[mk])*0.05
		mk = (cos_g >= x0) & (cos_g < 0.72)
		offset[mk] = (cos_g[mk]-x0)*slp
		mk = (cos_g >= 0.72)
		offset[mk] = slp*(0.72-x0) + (cos_g[mk]-0.72)*slp/3
		rbr2 = rbr_raw-offset
		[minr, maxr] = st.lower_upper( rbr2[rbr2 > -1], 0.01 )
		rbr2 -= minr
		rbr2 /= (maxr-minr)

		lower = -0.1
		upper = 1.11
		step = 0.2
		max_score_local = -np.Inf
		for _ in range(3):
			for thresh in np.arange(lower, upper, step):
				mk_cld = (rbr2>thresh) #&(dif>1)&(rbr>70)
				mk_sky = (rbr2 <= thresh) & (r1 > 0)
				bnd = st.get_border(
				    mk_cld, 10,
				    thresh=0.2, ignore=image.red<=0
				)
				#bnd=st.rolling_mean2(mk_cld.astype(np.float32),10,ignore=image.red<=0)
				#bnd=(bnd<0.8) & (bnd>0.2)
				sc = [
				    np.sum(mk_cld & cld) / ncld,
				    np.sum(mk_sky & sky) / nsky,
				    np.sum(dif2[bnd]>4) / np.sum(bnd),
				    -5*np.sum(mk_cld & sky) / nsky,
				    -5*np.sum(mk_sky & cld) / ncld,
				    -5*np.sum(dif2[bnd]<2) / np.sum(bnd)
				]
				score = np.nansum(sc)
				if score > max_score_local:
					max_score_local = score
					thresh_ref = thresh
					if score > max_score:
						max_score = score
						image.cm = mk_cld.astype(np.uint8);

			lower = thresh_ref - 0.5*step
			upper = thresh_ref + 0.5*step + 0.001
			step /= 4

	image.layers = 1

