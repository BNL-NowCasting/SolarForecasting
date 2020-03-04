from .cloud_height import cloud_height_helper
from .cloud_motion import cloud_motion_helper
from .features import extract_features_helper
from .image_preprocess import image_init, cloud_mask_helper, undistort_helper
from .stitch import stitch_helper

import tools.utils as utils
from tools.utils import quick_plot

from datetime import datetime
import ephem
import glob
from itertools import repeat, zip_longest
import matplotlib.pyplot as plt
import numpy as np
from numpy import AxisError
import os
from pathlib import Path
import pandas as pd
import pickle
import traceback

def load_image( zip_item ):
	self, cam, previous_set = zip_item
	c_id, camera = cam
	print(c_id)

	previous_image = None
	if previous_set and c_id in previous_set.images:
		previous_image = previous_set.images[c_id]

	f = get_next_image( self.image_path, c_id, self.timestamp, self.file_lists )
	if not f:
		print("Can't find file {} at {}".format(c_id,timestamp))
		return
	image_name = os.path.basename(f)

	# load, undistort, cloud mask
	image = Image(camera, f, previous_image, self.config, reprocess=self.reprocess, KEY=self.KEY)
	if image.error:
		print( "Image load failed: " + str(image.error) )
		return

	self.previous_images[c_id] = previous_image

	if self.reprocess["image"]:
		image.undistorted = False
		image.cloud_masked = None

		image.undistort()
		image.cloud_mask()

	self.images[c_id] = image

# used to override individual methods without messy version management
# obviously, this is an abomination and must be removed once the code
# starts working
def use_test_methods( KEY, image_set ):
	global image_init
	global cloud_height_helper
	global cloud_motion_helper
	global extract_features_helper
	global cloud_mask_helper
	global undistort_helper
	global stitch_helper

	print( "KEY = "  +KEY)
	if KEY:
		script_names = ["image_preprocess", "image_preprocess", "image_preprocess", "cloud_motion", "cloud_height", "stitch", "features"]
		method_names = ["image_init", "undistort_helper", "cloud_mask_helper", "cloud_motion_helper", "cloud_height_helper", "stitch_helper", "extract_features_helper"]

		for script, method in zip( script_names, method_names ):
			try:
				exec( "from .{}.{} import {}".format( KEY, script, method ) )
				print( "Testing " + method )
			except ImportError:
				pass
			except:
				print( script, method )
				print( traceback.format_exc() )

# helper methods
def append_to_csv( df, fp ):
	if not os.path.isfile(fp):
		df.to_csv( fp, mode='a', index=False )
	else:
		df.to_csv( fp, mode='a', index=False, header=False )

def get_next_image( image_path, cam, ts, file_lists ):
	minute = ts.strftime( "%Y%m%d%H%M" )
	second = ts.second
	filtered_files = [
	    f for f in file_lists[cam]
	    if f[-18:-6] == minute
	    and int(f[-6:-4]) - int(second) < 30
	    and int(f[-6:-4]) - int(second) >= 0
	    and os.path.isfile(f)
	]
	if len(filtered_files) == 0:
		print( "No matches found for camera {} at {}{}".format(
		    cam, minute, second
		) )
		return None
	return filtered_files[0]

class Image:
	def __init__(self, cam, fn, prev_image, config, reprocess={}, KEY=""):
		image_init( self, cam, fn, prev_image, config, reprocess=reprocess, KEY=KEY )

	def dump_image(self):
		Path( self.pic_dir ).mkdir( parents=True, exist_ok=True )
		with open( self.pic_path, 'wb' ) as fh:
			pickle.dump( self, fh, pickle.HIGHEST_PROTOCOL )

	def undistort( self, rgb=True, day_only=True ):
		print( "START UNDISTORT" )
		if self.undistorted:
			print( "ALREADY UNDISTORTED" )
			return

		cam = self.camera

		undistort_helper( self, cam, rgb=rgb, day_only=day_only )

		self.undistorted = True
		self.dump_image()

	def cloud_mask(self):
		print( "START CLOUD MASK" )
		if self.cloud_masked:
			print( "ALREADY CLOUD MASKED" )
			return

		if not self.previous_image:
			print( "SKIPPING CLOUD MASK; no prev image" )
			return

		cloud_mask_helper(self)

		### PLOT
		#quick_plot( [self.cm, self.rgb] )
		#save_path = "{}cm_plots/{}.png".format(
		#    self.pickle_path, os.path.basename( self.fn )
		#)
		#quick_plot( [self.cm, self.rgb], save=save_path )
		
		self.cloud_masked = True
		self.dump_image()

class ImageSet:
	# class variables to avoid having to reload these 
	# for every 30 second period considered
	file_lists = {}
	file_lists_day = None

	# create a dictionary with an entry for each camera
	# containing an array of all images taken on the given day
	@staticmethod
	def populate_file_lists( day, cameras, img_path ):
		file_lists = {}
		for cam in cameras:
			p = img_path + "{}/{}/".format(
			    cam, day
			)
			# if there are no records for a day
			# use a blank array
			if os.path.isdir( p ):
				file_lists[cam] = [p + x for x in os.listdir(p)]
			else:
				file_lists[cam] = []
		ImageSet.file_lists = file_lists

	def __init__( self, timestamp, previous_set, camera_dict, config, day_only=True, KEY="" ):
		print( "STARTING IMAGE SET " + str(timestamp) )

		site = config["site_id"]
		self.feature_path = config["paths"][site]["feature_path"]
		self.image_path = config["paths"][site]["img_path"]
		pickle_path = config["paths"][site]["pickle_path"]

		self.timestamp = timestamp
		self.day = timestamp.strftime( "%Y%m%d" )

		# MAGIC CONSTANT
		# I want this to be consistent between runs but also
		# not to break if any given camera is missing data
		arbitrary_camera = camera_dict[min(camera_dict.keys())]

		gatech = ephem.Observer();
		gatech.date = timestamp.strftime('%Y/%m/%d %H:%M:%S')
		gatech.lat = str(arbitrary_camera.lat)
		gatech.lon = str(arbitrary_camera.lon)
		# print( ephem.localtime( gatech.date ).ctime() )

		sun = ephem.Sun()
		sun.compute(gatech)
		self.skip_bc_night = sun.alt < np.pi/20.
		if self.skip_bc_night:
			print( "Skipping because it is night time" )
			return

		### This whole block of code is setting it up so I can rerun
		### stitch (say) and automatically redo feature_extraction
		### without wasting time reprocessing anything else
		pl_c = config["pipeline"]
		self.dependencies = {
			"features": ["stitch"],
			"stitch": ["height", "motion"],
			"motion": ["image"],
			"height": ["image"],
			"image": []
		}
		def fill_dependencies(im_s, key):
			arr = im_s.dependencies[key]
			for val in im_s.dependencies[key]:
				arr += fill_dependencies( im_s, val )
			# no dups
			arr = list(set(arr))
			im_s.dependencies[key] = arr
			return arr
		fill_dependencies(self, "features")
			
		self.reprocess = {
		    "features": pl_c["features"]["reprocess"],
		    "height": pl_c["height"]["reprocess"],
		    "motion": pl_c["motion"]["reprocess"],
		    "stitch": pl_c["stitch"]["reprocess"],
		    "image": pl_c["image_preprocessing"]["reprocess"]
		}
		self.reprocess["all"] = pl_c["reprocess_all"]

		# this code block is because use_test_methods modifies
		# reprocess on the assumption that we have fully processed
		# everything with KEY="" and nothing with the current key
		self.pic_dir = "{}{}/image_sets/{}/".format(pickle_path, KEY, self.day)
		self.pic_path = "{}{}.pkl".format(self.pic_dir, self.timestamp)

		if KEY:
			from copy import copy
			orig_reprocess = copy(self.reprocess)

			use_test_methods( KEY, self )

			if os.path.exists(self.pic_path):
				self.reprocess = orig_reprocess
		# end

		for key in self.dependencies.keys():
			self.reprocess[key] = self.reprocess["all"] or self.reprocess[key] or any(self.reprocess[val] for val in self.dependencies[key])

		self.reprocess["all"] = (self.reprocess["all"] or
					all(v for v in self.reprocess.values()))
		self.reprocess["any"] = any(v for v in self.reprocess.values())

		print( self.dependencies )
		print( self.reprocess )
		### end

		self.config = config
		self.camera_dict = camera_dict

		self.cloud_base_height = None
		self.cloud_base_vel = None
		self.stitched_image = None
		self.extracted_features = False
		self.skip_bc_error = False
		self.skip = False

		self.layer_vels = []
		self.vels = []
		self.layer_heights = []
		self.heights = []

		# I only pickle calculated values like cloud_base_height and vel
		self.KEY = KEY

		if os.path.exists(self.pic_path) and not self.reprocess["all"]:
			with open( self.pic_path, 'rb' ) as fh:
				obj = pickle.load(fh)
				for k, v in obj.__dict__.items():
					self.__dict__[k] = v

		default_pic_path = "{}/image_sets/{}/{}.pkl".format( pickle_path, self.day, self.timestamp )
		if KEY and os.path.exists(default_pic_path) and not self.reprocess["all"] and not os.path.exists(self.pic_path):
				with open( default_pic_path, 'rb' ) as fh:
					obj = pickle.load(fh)
					for k, v in obj.__dict__.items():
						self.__dict__[k] = v

		# the whole point of preprocess is to produce a features file
		# if such a file exists, we're already done
		# (we still have to load this set though in case the next set
		#  has processing left to do and wants to reference this one)
		fn = "{}{}/{}/*_{}.csv".format(
		    self.feature_path, KEY, self.day, self.timestamp
		)
		if not self.reprocess["any"] and len(glob.glob(fn)):
			print( "Skipping ImageSet; features already extracted" )
			self.finished = True
			return

		# MAGIC CONSTANT 0
		self.deg2km=6367*np.pi/180	

		self.min_lat = min( [cam.lat for cam in camera_dict.values()] )
		self.min_lon = min( [cam.lon for cam in camera_dict.values()] )
		self.max_lat = max( [cam.lat for cam in camera_dict.values()] )
		self.max_lon = max( [cam.lon for cam in camera_dict.values()] )
		self.median_lat = np.median( [cam.lat for cam in camera_dict.values()] )

		self.x_cams = (self.max_lon - self.min_lon) * self.deg2km * np.cos( self.min_lat * np.pi / 180 )
		self.y_cams = (self.max_lat - self.min_lat) * self.deg2km 

		locs = config["ghi_sensors"]["coords"]
		self.ghi_locs = np.array( [
		    v for (k,v) in sorted(locs.items(), key=lambda i: i[0])
		] )

		self.lead_minutes = config["pipeline"]["lead_times"]
		self.interval = config["pipeline"]["interval"]

		day = timestamp.strftime( "%Y%m%d" )
		if day != ImageSet.file_lists_day:
			# preload the day's images rather than loading
			# them once for every 30 seconds
			ImageSet.populate_file_lists(
			    day, camera_dict.keys(), self.image_path
			)
			# maintain a local reference to the correct file lists
			# so that we can reference them even if another run
			# changes the class variable
			self.file_lists = ImageSet.file_lists

			# assuming the next instance is for the same day
			# we won't have to run this again
			ImageSet.file_lists_day = day

		# load the images for the given timestamp
		self.images = {}
		self.previous_images = {}

		#pool = multiprocessing.Pool(len(camera_dict.keys()))
		#pool.map( self.load_image, zip( camera_dict.items(), repeat(previous_set) ) )
		#self.pool = pool
		#self.close_pool()
		for c_id, camera in camera_dict.items():
			self.load_image( c_id, camera, previous_set )
	
		self.dump_set()
		print( "FINISHED IMAGE SET" )

	def load_image( self, c_id, camera, previous_set ):
		#cam, previous_set = zip_item
		#c_id, camera = cam
		#print(c_id)

		previous_image = None
		if previous_set and c_id in previous_set.images:
			previous_image = previous_set.images[c_id]

		f = get_next_image( self.image_path, c_id, self.timestamp, self.file_lists )
		if not f:
			print("Can't find file {} at {}".format(c_id,timestamp))
			return
		image_name = os.path.basename(f)

		# load, undistort, cloud mask
		image = Image(camera, f, previous_image, self.config, reprocess=self.reprocess, KEY=self.KEY)
		if image.error:
			print( "Image load failed: " + str(image.error) )
			return

		self.previous_images[c_id] = previous_image

		if self.reprocess["image"]:
			image.undistorted = False
			image.cloud_masked = None

			image.undistort()
			image.cloud_mask()

		self.images[c_id] = image

		
	def cloud_motion(self):
		print( "Start cloud motion" )
		if self.cloud_base_vel is not None and not self.reprocess["motion"]:
			print("Already found velocities " + str(self.vels))
			print( self.cloud_base_vel )
			return self.vels

		if self.skip_bc_night:
			print( "Error: invoked cloud_motion but is night and day_only" )
			return self.vels

		for c_id, img in self.images.items():
			cloud_motion_helper(
			    self, self.images[c_id], self.previous_images[c_id]
			)
		#for img in self.images.values():
		#	for l in range(img.layers):
		#		img.v[l] = img.v[l][0]

		self.vels = [img.v for img in self.images.values()]
		try:
			num_layers = max(len(im) for im in self.vels)
			self.layer_vels = [[]]*num_layers
			for i in range(num_layers):
				self.layer_vels[i] = [
				    np.nanmedian([img[i][0] if i < len(img) else np.nan for img in self.vels]),
				    np.nanmedian([img[i][1] if i < len(img) else np.nan for img in self.vels])
				]
					
			print( self.vels )
			print( self.layer_vels )
			self.cloud_base_vel = self.layer_vels[0]
		except AxisError:
			print( "Failed to find layer_vels because there are no layers" )
			self.layer_vels = []
			self.cloud_base_vel = np.nan

		print("Found velocities " + str(self.vels))
		print("Found layer velocities " + str(self.layer_vels))
		print( self.cloud_base_vel )

		self.dump_set()
		return self.vels

	def cloud_height(self):
		print( "Start cloud height" )
		if self.cloud_base_height is not None and not self.reprocess["height"]:
			print("Already found cloud heights: "+str(self.heights))
			print( self.cloud_base_height )
			return self.heights

		if self.skip_bc_night:
			print( "Error: invoked cloud_height but is night and day_only" )
			return self.heights

		for cam in self.camera_dict:
			if not cam in self.images:
				print( "SKIPPING CLOUD HEIGHT; image does not exist for cam " + str(cam) )
				continue

			cloud_height_helper( self, cam )

		cams = list(self.camera_dict.keys())
		cams.sort()

		self.heights = [
		    self.images[cam].height if cam in self.images else [] for cam in cams
		]
		try:
			self.layer_heights = np.nanmedian( np.array( list(zip_longest(*self.heights, fillvalue=np.nan)) ), axis=1 )
			self.cloud_base_height = self.layer_heights[0]
		except AxisError:
			print( "Failed to find layer_heights because there are no layers" )
			self.layer_heights = []
			self.cloud_base_height = np.nan
		

		### TEMP: log height estimates to csv file
		self.bottom_layer_heights = [self.heights[i][0] if len(self.heights[i]) else None for i in range(len(self.heights))]
		# dtype=float casts None to nan

		print( "Found cloud heights: " + str(self.heights) )
		print( self.cloud_base_height )

		header = ["timestamp"] + cams + ["avg"]
		h_df = pd.DataFrame( [np.concatenate( (
		    np.array([self.timestamp]), 
		    self.bottom_layer_heights, 
		    np.array([self.cloud_base_height])
		) )], columns=header )
		append_to_csv( h_df, "/home/tchapman/root/data/bnl/height_estimation_{}.csv".format( self.KEY ) )

		self.dump_set()
		return self.heights

	def stitch(self):
		print( "Start stitch" )
		if self.stitched_image is not None and not self.reprocess["stitch"]:
			print("Already found stitch")
			return self.stitched_image
		if self.skip_bc_night:
			print("Error: invoked stitch but is night and day_only")
			return None

		if self.cloud_base_height is None:
			self.cloud_height()
		if self.cloud_base_vel is None:
			self.cloud_motion()

		heights = self.layer_heights
		vels = self.layer_vels
		if len(heights) == 0 or len(vels) == 0 or all(np.isnan(h) for h in heights):
			# clear sky
			heights = [15000]
			vels = [[0, 0]]

		heights = np.array(heights)
		vels = np.array(vels)

		stitch_helper( self, heights, vels )
		self.dump_set()

		quick_plot( [self.stitched_image.rgb, self.stitched_image.cm] )
		return self.stitched_image.rgb

	def extract_features(self):
		print( "Start features" )
		if self.extracted_features and not self.reprocess["features"]:
			print( "Already extracted features" )
			return

		if self.skip_bc_night:
			print("Error: invoked features but is night and day_only")
			return

		if self.stitched_image is None:
			self.stitch()

		self.win_size = self.config["pipeline"]["features"]["win_size"]
		# currently creates 25 files; I'd much rather return something
		# and then save it here... TODO
		extract_features_helper(self)

		self.extracted_features = True
		self.dump_set()

	def dump_set(self):
		Path( self.pic_dir ).mkdir( parents=True, exist_ok=True )

		data_obj = self.DataWrapper(self)
		with open( self.pic_path, 'wb' ) as fh:
			pickle.dump( data_obj, fh, pickle.HIGHEST_PROTOCOL )

	class StitchedImage:
		def __init__(self):
			self.sz = None
			self.saz = None
			self.height = None
			self.v = None
			self.pixel_size = None
			self.lat = None
			self.lon = None
			self.rgb = None
			self.cm = None

	class DataWrapper:
		def __init__(self, image_set):
			self.cloud_base_height = image_set.cloud_base_height
			self.cloud_base_vel = image_set.cloud_base_vel
			self.stitched_image = image_set.stitched_image
			self.extracted_features = image_set.extracted_features
			self.layer_vels = image_set.layer_vels
			self.vels = image_set.vels
			self.layer_heights = image_set.layer_heights
			self.heights = image_set.heights
