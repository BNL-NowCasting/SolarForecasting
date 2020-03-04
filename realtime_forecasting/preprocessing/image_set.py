from .cloud_height import cloud_height_helper
from .cloud_motion import cloud_motion_helper
from .features import extract_features_helper
from .image import Image
from .stitch import stitch_helper

from tools.utils import quick_plot

from datetime import datetime
import ephem
import glob
from itertools import repeat, zip_longest
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from numpy import AxisError
import os
from pathlib import Path
import pandas as pd
import pickle
import traceback

def load_image( zip_item ):
	c_id, camera, previous_image, f, reprocess, KEY, pick = zip_item
	print("Load iamge " + str(c_id) )

	image_name = os.path.basename(f)

	# load, undistort, cloud mask
	image = Image(camera, f, previous_image, pick, reprocess=reprocess, KEY=KEY)
	if image.error:
		print( "Image load failed: " + str(image.error) )
		return (c_id, None)

	if reprocess:
		image.undistorted = False
		image.cloud_masked = None

	image.undistort()
	image.cloud_mask()

	return (c_id, image)
	
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
	script_names = [
	    "image", "cloud_motion", "cloud_height", "stitch", "features"
	]
	method_names = [
	    "Image", "cloud_motion_helper", "cloud_height_helper", 
	    "stitch_helper", "extract_features_helper"
	]

	for script, method in zip( script_names, method_names ):
		try:
			exec("from .{}.{} import {}".format(KEY,script,method))
			print( "Testing " + method )
		except ImportError:
			pass
		except:
			print( script, method )
			print( traceback.format_exc() )

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

	def __init__(self, timestamp, previous_set, camera_dict, config, KEY=""):
		print( "Start image set " + str(timestamp) )
		self.loaded = False
		self.complete = False
		self.skip_bc_night = False

		self.config = config
		#self.logger = logger

		site = config["site_id"]
		self.feature_path = config["paths"][site]["feature_path"]
		self.pickle_path = config["paths"][site]["pickle_path"]
		self.image_path = config["paths"][site]["img_path"]

		self.KEY = KEY
		self.previous_set = previous_set

		self.timestamp = timestamp
		self.time_str = timestamp.strftime( "%Y%m%d%H%M%S" )
		self.day = timestamp.strftime( "%Y%m%d" )

		## are we already done processing this ImageSet?
		self.set_reprocess()
		fp = "{}{}/{}/*_{}.csv".format(
		    self.feature_path, KEY, self.day, self.timestamp
		)
		if not self.reprocess["any"] and len(glob.glob(fp)):
			print( "Skipping ImageSet; features already extracted" )
			self.complete = True
			return

		## is it nighttime?
		# I want this camera to be consistent between runs but also
		# not to break if any given camera is missing data
		arbitrary_camera = camera_dict[min(camera_dict.keys())]

		gatech = ephem.Observer();
		gatech.date = timestamp.strftime('%Y/%m/%d %H:%M:%S')
		gatech.lat = str(arbitrary_camera.lat)
		gatech.lon = str(arbitrary_camera.lon)

		sun = ephem.Sun()
		sun.compute(gatech)
		if sun.alt < np.pi/20.:
			self.skip_bc_night = True
			print( "Skipping because it is night time" )
			return

		if KEY:
			use_test_methods( self, KEY )

		self.cloud_base_height = None
		self.cloud_base_vel = None
		self.stitched_image = None
		self.extracted_features = False

		self.layer_vels = []
		self.vels = []
		self.layer_heights = []
		self.heights = []

		self.load_details(KEY=KEY)
		if previous_set is not None and not previous_set.loaded:
			previous_set.load_details(KEY=KEY)
		
		self.camera_dict = camera_dict


		self.pic_dir = "{}{}/image_sets/{}/".format(
		    self.pickle_path, KEY, self.day
		)
		self.pic_path = "{}{}.pkl".format(self.pic_dir, self.time_str)

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

		self.images = {}
		self.previous_images = {}

#		self.load_images()
		for c_id, cam in camera_dict.items():
			self.load_image( c_id, cam )
		print( "Finish init" )
		
	def load_images( self ):
		p = multiprocessing.Pool(10)

		cam_ids = list(self.camera_dict.keys())
		cameras = [self.camera_dict[c_id] for c_id in cam_ids]

		if self.previous_set:
			previous_images = [
			  self.previous_set.images.get(c_id) for c_id in cam_ids
			]
		else:
			previous_images = repeat(None)

		file_paths = [
		    get_next_image(
		      self.image_path, c_id, self.timestamp, self.file_lists
		    ) for c_id in cam_ids
		]

		for (c_id, img) in zip(cam_ids, previous_images):
			self.previous_images[c_id] = img

		args = zip(cam_ids, cameras, previous_images, file_paths, 
		            repeat(self.reprocess["image"]), repeat(self.KEY),
		            repeat(self.pickle_path) )
		images = p.map( 
		    load_image, 
		    args
		)
		p.close()
		p.join()

		for (c_id, img) in images:
			if not img.error:
				self.images[c_id] = img

	def load_image( self, c_id, camera ):
		previous_set = self.previous_set

		print(c_id)

		previous_image = None
		if previous_set and c_id in previous_set.images:
			previous_image = previous_set.images[c_id]

		f = get_next_image( 
		    self.image_path, c_id, self.timestamp, 
		    self.file_lists 
		)
		if not f:
			print("Can't find file {} at {}".format(c_id,timestamp))
			return
		image_name = os.path.basename(f)

		# load, undistort, cloud mask
		image = Image(camera, f, previous_image, self.pickle_path, reprocess=self.reprocess, KEY=self.KEY)

		if image.error:
			print( "Image load failed: " + str(image.error) )
			return

		self.previous_images[c_id] = previous_image

		if self.reprocess["image"]:
			image.undistorted = False
			image.cm = None

		image.undistort()
		image.cloud_mask()

		self.images[c_id] = image

	def cloud_height(self):
		print( "Start cloud height" )
		print( len(self.heights)  )
		print( self.reprocess["height"] )
		if len(self.heights) and not self.reprocess["height"]:
			print("Already found cloud heights: "+str(self.heights))
			print( self.cloud_base_height )
			return self.heights

		if self.skip_bc_night:
			print( "Error: invoked cloud_height but is night" )
			return self.heights

#		p = multiprocessing.Pool(10)
#		p.map( cloud_height_helper, zip(repeat(self.camera_dict), [cam for cam in self.camera_dict if cam in self.images], repeat(self.images)) )
		for cam in self.camera_dict:
			if not cam in self.images:
				print("skipping {}; img not found".format(cam))
				continue

			cloud_height_helper( self.camera_dict, cam, self.images )

		cams = list(self.camera_dict.keys())
		cams.sort()

		self.heights = [ self.images[cam].height if cam in self.images else [] for cam in cams ]
		print( "set heights" )
		try:
			self.layer_heights = np.nanmedian( np.array( list(zip_longest(*self.heights, fillvalue=np.nan)) ), axis=1 )
			self.cloud_base_height = self.layer_heights[0]
		except AxisError:
			print( "Failed to find layer_heights because there are no layers" )
			self.layer_heights = []
			self.cloud_base_height = np.nan

		self.log_heights(cams)

		print( "save ehighs" )
		self.dump_set()
		return self.heights

	def cloud_motion(self):
		print( "Start cloud motion" )
		if self.cloud_base_vel is not None and not self.reprocess["motion"]:
			print("Already found velocities " + str(self.vels))
			print( self.cloud_base_vel )
			return self.vels

		if self.skip_bc_night:
			print( "Error: invoked cloud_motion but is night" )
			return self.vels

		for c_id, img in self.images.items():
			cloud_motion_helper(
			    self, self.images[c_id], self.previous_images[c_id]
			)

		self.vels = [img.v for img in self.images.values()]
		try:
			num_layers = max(len(im) for im in self.vels)
			self.layer_vels = [[]]*num_layers
			for i in range(num_layers):
				self.layer_vels[i] = [
				    np.nanmedian([img[i][0] if i < len(img) else np.nan for img in self.vels]),
				    np.nanmedian([img[i][1] if i < len(img) else np.nan for img in self.vels])
				]
			self.layer_vels_2 = np.nanmedian( np.array( list(zip_longest(*self.vels, fillvalue=np.nan)) ), axis=1 )
			self.cloud_base_vel = self.layer_vels[0]

			print( self.vels )
			print( self.layer_vels )
			print( self.layer_vels_2 )
		except AxisError:
			print( "Failed to find layer_vels because there are no layers" )
			self.layer_vels = []
			self.cloud_base_vel = np.nan

		print("Found velocities " + str(self.vels))
		print("Found layer velocities " + str(self.layer_vels))
		print( self.cloud_base_vel )

		self.dump_set()
		return self.vels

	def extract_features(self):
		print( "Start features" )
		if self.extracted_features and not self.reprocess["features"]:
			print( "Already extracted features" )
			return

		if self.skip_bc_night:
			print("Error: invoked features but is night")
			return

		if self.stitched_image is None:
			self.stitch()

		locs = self.config["ghi_sensors"]["coords"]
		self.ghi_locs = np.array( [
		    v for (k,v) in sorted(locs.items(), key=lambda i: i[0])
		] )

		self.lead_minutes = self.config["pipeline"]["lead_times"]
		self.interval = self.config["pipeline"]["interval"]

		self.win_size = self.config["pipeline"]["features"]["win_size"]
		# currently creates 25 files; I'd much rather return something
		# and then save it here... TODO
		extract_features_helper(self)

		self.extracted_features = True
		self.dump_set()


	def stitch(self):
		print( "Start stitch" )
		if self.stitched_image is not None and not self.reprocess["stitch"]:
			print("Already found stitch")
			return self.stitched_image
		if self.skip_bc_night:
			print("Error: invoked stitch but is night")
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

		self.deg2km = 6367*np.pi/180

		self.min_lat = min( [cam.lat for cam in self.camera_dict.values()] )
		self.min_lon = min( [cam.lon for cam in self.camera_dict.values()] )
		self.max_lat = max( [cam.lat for cam in self.camera_dict.values()] )
		self.max_lon = max( [cam.lon for cam in self.camera_dict.values()] )
		self.median_lat = np.median( [cam.lat for cam in self.camera_dict.values()] )

		self.x_cams = (self.max_lon - self.min_lon) * self.deg2km * np.cos( self.min_lat * np.pi / 180 )
		self.y_cams = (self.max_lat - self.min_lat) * self.deg2km


		stitch_helper( self, heights, vels )
		self.dump_set()

		#quick_plot( [self.stitched_image.rgb, self.stitched_image.cm] )
		return self.stitched_image.rgb


	# load any saved intermediate results
	def load_details(self, KEY=""):
		if self.loaded:
			print( "skipping detail load" )
			return

		pic_path = "{}{}/image_sets/{}/{}.pkl".format(
		    self.pickle_path, KEY, self.day, self.time_str
		)

		if os.path.exists(pic_path) and not self.reprocess["all"]:
			print( "loaded details" )
			with open( pic_path, 'rb' ) as fh:
				obj = pickle.load(fh)
				for k, v in obj.__dict__.items():
					print( k, v )
					self.__dict__[k] = v
			print( len(self.heights) )
		elif KEY and not self.reprocess["all"]:
			# if we haven't processed this timestamp with KEY,
			# check if we have any reusable intermediate values
			# from processing it with KEY=""
			self.load_details()
		self.loaded = True
		

	def fill_dependencies(self, root):
		arr = self.dependencies[root]
		for val in self.dependencies[root]:
			arr += self.fill_dependencies( val )
		# no dups
		arr = list(set(arr))
		self.dependencies[root] = arr
		return arr

	# determine which parts of the processing sequence need to be redone
	def set_reprocess(self):
		self.dependencies = {
		    "features": ["stitch"],
		    "stitch": ["height", "motion"],
		    "motion": ["image"],
		    "height": ["image"],
		    "image": []
		}
		self.fill_dependencies( "features" )

		pl_c = self.config["pipeline"]
		self.reprocess = {
		    "features": pl_c["features"]["reprocess"],
		    "height": pl_c["height"]["reprocess"],
		    "motion": pl_c["motion"]["reprocess"],
		    "stitch": pl_c["stitch"]["reprocess"],
		    "image": pl_c["image_preprocessing"]["reprocess"]
		}
		self.reprocess["all"] = (pl_c["reprocess_all"] or
					all(v for v in self.reprocess.values()))
		self.reprocess["any"] = any(v for v in self.reprocess.values())
		
		for key in self.dependencies.keys():
			self.reprocess[key] = (
			    self.reprocess["all"] or 
			    self.reprocess[key] or 
			    any(self.reprocess[val] 
			      for val in self.dependencies[key])
			)
		print( "reprocess: " + str(self.reprocess) )

	def dump_set(self):
		Path( self.pic_dir ).mkdir( parents=True, exist_ok=True )

		data_obj = self.DataWrapper(self)
		with open( self.pic_path, 'wb' ) as fh:
			pickle.dump( data_obj, fh, pickle.HIGHEST_PROTOCOL )

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

	class StitchedImage:
		def __init__(self):
			self.sz = None
			self.saz = None
			self.h = None
			self.v = None
			self.pixel_size = None
			self.lat = None
			self.lon = None
			self.rgb = None
			self.cm = None
	

	def log_heights(self, cams):
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

