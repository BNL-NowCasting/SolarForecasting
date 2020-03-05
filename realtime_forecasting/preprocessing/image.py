from datetime import datetime, timedelta, timezone
import ephem
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle
from scipy.ndimage.filters import maximum_filter, gaussian_filter, laplace, median_filter
from skimage.morphology import remove_small_objects
import tools.stat_tools_2 as st
import traceback

class Image:
	def __init__(self, fn, prev_fn, pickle_path, camera_object=None, camera_id=None, reprocess=False, KEY=""):
		bn = os.path.basename( fn )
		self.timestamp = datetime.strptime( bn[-18:-4], "%Y%m%d%H%M%S" )
		self.day = bn[-18:-10]
		self.pickle_path = pickle_path

		if camera_object:
			camera = camera_object
		elif camera_id:		
			camera = utils.load_camera_object( camera_id, self.pickle_path ) 
		else:
			print( "Image constructor requires camera_object or camera_id" )
			exit(1)

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

		if os.path.exists( self.pic_path ) and not reprocess:
			with open( self.pic_path, 'rb' ) as fh:
				self.__dict__ = pickle.load(fh).__dict__
			return

		if KEY:
			if os.path.exists( old_pic_path ) and not reprocess:
				with open( old_pic_path, 'rb' ) as fh:
					self.__dict__ = pickle.load(fh).__dict__
				return

		self.fn = fn
		self.camera = camera
		self.previous_image = None
		if prev_fn is not None:
			self.previous_image = Image( prev_fn, None, pickle_path, camera_object=camera )
			# I pickle everything, so we can't have this recursing back
			# to the first instance processed.
			self.previous_image.previous_image = None
			self.previous_image.camera = None

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
		self.cloud_masked = False

		self.undistorted = False
		self.skip_bc_night = False
		self.error = ""

#		self.dump_self()

	def undistort( image, rgb=True, day_only=True ):
		print( "Start undistort" )
		import warnings
		#warnings.filterwarnings('ignore')

		#print( "Start undistort" )
		if image.undistorted:
			print( "already undistorted" )
			return

		cam = image.camera
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

		try:
			im0 = plt.imread(image.fn);
		except Exception as e:
			print('Cannot read file:', image.fn)
			print( traceback.format_exc() )
			image.error = str(e)
			return None

		if len(im0.shape) != 3:
			# some corrupted images can be read (and displayed)
			# but for some reason load as empty matrices
			image.error = "Corrupted image; shape = " + str(im0.shape)
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
	
		image.undistorted = True
#		image.dump_self()

	def cloud_mask( self ):
		import warnings
		#warnings.filterwarnings('ignore')
		print( "Start cloud mask" )
		if self.cloud_masked:
			print( "already cloud masked" )
			return

		if not self.previous_image:
			print( "skipping cloud mask; no prev image" )
			return

		image0 = self.previous_image
		cam = self.camera

		self.cloud_masked = True

		cos_s = np.cos(self.sz)
		sin_s = np.sin(self.sz)
		cos_sp = np.cos(self.saz)
		sin_sp = np.sin(self.saz)
		cos_th = cam.cos_th
		sin_th = np.sqrt(1-cos_th**2)
		cos_p = cam.cos_p
		sin_p = cam.sin_p


		# cosine of the angle between illumination and view directions
		cos_g = cos_s*cos_th+sin_s*sin_th*(cos_sp*cos_p+sin_sp*sin_p)

		r0 = image0.rgb[...,0].astype(np.float32)
		r0[r0<=0] = np.nan
		r1 = self.rgb[...,0].astype(np.float32)
		r1[r1<=0] = np.nan
		rbr_raw = (r1-self.rgb[:,:,2])/(self.rgb[:,:,2]+r1)
		rbr = rbr_raw.copy()
		rbr -= st.rolling_mean2(rbr,int(self.nx//6.666))
		rbr[rbr>0.08] = 0.08
		rbr[rbr<-0.08] = -0.08
		rbr = (rbr+0.08)*1587.5+1 # scale rbr to 0-255
		mblue = np.nanmean(self.rgb[(cos_g<0.7) & (r1>0) & (rbr_raw<-0.01),2].astype(np.float32))

		err = r1-r0
		err -= np.nanmean(err)
		dif = st.rolling_mean2(abs(err),100)
		err = st.rolling_mean2(err,5)
		dif2 = maximum_filter(np.abs(err),5)

		sky = (rbr<126) & (dif<1.2)
		sky |= dif<0.9
		sky |= (dif<1.5) & (err<3) & (rbr<105)
		sky |= (rbr<70)
		sky &= (self.red>0)
		cld = (dif>2) & (err>4)
		cld |= (self.red>150) & (rbr>160) & (dif>3)
		cld |= (rbr>180) # clouds with high rbr
		cld[cos_g>0.7] |= (self.rgb[cos_g>0.7,2]<mblue) & (rbr_raw[cos_g>0.7]>-0.01) #dark clouds
		cld &= dif>3

		total_pixel = np.sum(r1>0)

		min_size = 50*self.nx/1000
		cld = remove_small_objects(cld, min_size=min_size, connectivity=4, in_place=True)
		sky = remove_small_objects(sky, min_size=min_size, connectivity=4, in_place=True)

		ncld = np.sum(cld)
		nsky = np.sum(sky)
	#       print(ncld/total_pixel,nsky/total_pixel);
		if (ncld+nsky) <= 1e-2*total_pixel:
			self.error = "ncld + nsky is less than 1% of the image"
			return;
		elif (ncld < nsky) & (ncld <= 2e-2*total_pixel):
			print( "CLEAR SKY" )
			# clear sky
			self.cm = cld.astype(np.uint8)
			self.layers = 0
			return
		elif (ncld > nsky) & (nsky <= 2e-2*total_pixel):
			# overcast sky
			self.cm = ((~sky)&(r1>0)).astype(np.uint8)
			self.layers = 1
			return

		max_score = -np.Inf
		x0 = -0.15;
		ncld = 0.25*nsky+0.75*ncld
		nsky = 0.25*ncld+0.75*nsky
	#       ncld=max(ncld,0.05*total_pixel); nsky=max(nsky,0.05*total_pixel)
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
					    thresh=0.2, ignore=self.red<=0
					)
					#bnd=st.rolling_mean2(mk_cld.astype(np.float32),10,ignore=self.red<=0)
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
							self.cm = mk_cld.astype(np.uint8);

				lower = thresh_ref - 0.5*step
				upper = thresh_ref + 0.5*step + 0.001
				step /= 4

		self.layers = 1

#		self.dump_self()

	def cloud_motion(self):
		prev_self = self.previous_image

		self.v = [[np.nan, np.nan]]
		self.layers = 1

		if prev_img is None:
			# sometimes individual cameras are missing data
			return

		r1 = prev_img.red.astype(np.float32)
		r1[r1<=0] = np.nan
		r2 = self.red.astype(np.float32)
		r2[r2<=0] = np.nan
		err0 = r2-r1

		dif = np.abs(err0)
		# MAGIC CONSTANT 5,6,7,8
		dif = st.rolling_mean2(dif,20)
		semi_static = (abs(dif)<10) & (r1-127>100)
		semi_static = morphology.binary_closing(semi_static, np.ones((10,10)))
		semi_static = remove_small_objects(semi_static, min_size=200, in_place=True)
		self.rgb[semi_static] = 0
		r2[semi_static] = np.nan

		if np.sum(self.cm>0) < 2e-2*self.nx*self.ny:
			# no clouds
			self.layers=0;
		else:
			dilated_cm = morphology.binary_dilation( self.cm, np.ones((15,15)) )
			dilated_cm &= (r2>0)
			[vy,vx,max_corr] = cloud_motion_math(
			    r1, r2, mask1=r1>0, mask2=dilated_cm, ratio=0.7, threads=4
			)

			if np.isnan(vy):
				self.layers=0;
			else:
				self.v = [[vy,vx]]
				self.layers=1

	def dump_self(self):
		Path( self.pic_dir ).mkdir( parents=True, exist_ok=True )
		with open( self.pic_path, 'wb' ) as fh:
			pickle.dump( self, fh, pickle.HIGHEST_PROTOCOL )

