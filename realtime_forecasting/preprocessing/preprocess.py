from . import camera as cam
from .image_set import ImageSet
from .image import Image
import tools.utils as utils

from datetime import datetime, timedelta
from itertools import repeat, zip_longest
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import traceback

last_images = None

def get_next_image( image_path, cam, ts, file_lists ):
	start_ts = ts.strftime( "%Y%m%d%H%M%S" )
	end_ts = (ts+timedelta(seconds=30)).strftime( "%Y%m%d%H%M%S" )

	filtered_files = [
	    f for f in file_lists[cam]
	    if f[-18:-4] >= start_ts
	    and f[-18:-4] < end_ts
	    and os.path.isfile(f)
	]
	if len(filtered_files) == 0:
		print( "No matches found for camera {} at {}".format(
		    cam, ts
		) )
		return None
	return filtered_files[0]

def prepare_image(args):
	c_id, ts, image_file, previous_image_file, config, reprocess = args
	if image_file is None:
		return
	print( "Prepare " + str(image_file) )
	image = Image(image_file, previous_image_file, config, reprocess=reprocess, camera_id=c_id)
	image.preprocess()

def process_image_set(args):
	ts, config, reprocess = args
	#ts, cam_ids, image_files, pickle_path = args

	image_set = ImageSet( ts, config, reprocess )
	image_set.cloud_height()
	image_set.stitch()
	image_set.dump_set()
	image_set.extract_features()

def preprocess( config, logger, KEY="" ):
	global last_images

	site = config["site_id"]
	img_path = config["paths"][site]["img_path"]
	pickle_path = config["paths"][site]["pickle_path"]

	num_cores = config["pipeline"]["cores_to_use"]
	cameras = config["cameras"][site]["all_cameras"]

	reprocess = build_reprocess( config )

	target_ranges = sorted( zip(
		config["pipeline"]["target_ranges"]["start_dates"],
		config["pipeline"]["target_ranges"]["end_dates"]
	), key=lambda x: x[0] )

	def timestamps_to_consider():
		global last_images

		next_t = datetime.strptime( '0001', '%Y' )
		for (start, end) in target_ranges:
			# start a step behind start so we can have last_images set when we start
			start_t = utils.date_from_timestamp( start ) - timedelta(seconds=30)
			end_t = utils.date_from_timestamp( end )

			if start_t > next_t:
				next_t = start_t
				last_images = None
			#next_t = utils.date_from_timestamp( start )
			print( "Next_t =  " + str(start) )

			while next_t < end_t:
				yield next_t
				## TODO: handle night-time case
				next_t = next_t + timedelta( seconds=30 )

	pic_path = pickle_path + "camera_objects.pkl"
#	camera_dict = utils.try_pickle(
#		pic_path,
#		create_camera_objects,
#		[config]
#	)

	pool = multiprocessing.Pool( num_cores )

	# Process in chunks as a compromise between preprocessing all images,
	# then doing all clouds heights, then all feature extraction etc. (efficient & easy)
	# and fully processing each timestamp in order (actually provides forecasts before the whole
	#   process is completed)
	ts_gen = timestamps_to_consider()
	timestamps = []

	file_lists = {}
	last_day = None

	first = True
	for ts in ts_gen:
		day = ts.strftime( "%Y%m%d" )
		if day != last_day:
			file_lists = {}
			for cam in cameras:
				p = img_path + "{}/{}/".format(
				    cam, day
				)
				print( p )
				# if there are no records for a day
				# use a blank array
				if os.path.isdir( p ):
					file_lists[cam] = [p + x for x in os.listdir(p)]
				else:
					file_lists[cam] = []
			last_day = day
		# do preprocessing which only depends on previous images
		# having been preprocessed (undistort, cloud mask, cloud motion)
		image_files = [
		    get_next_image(img_path, cam, ts, file_lists) 
		    for cam in cameras
		]
		previous_image_files = [get_next_image(
		    img_path, cam, ts-timedelta(seconds=30), file_lists
		) for cam in cameras] # breaks at midnight but GHI is 0 at midnight, so eh...
		if first:
			previous_image_files = repeat(None, len(cameras))
#		print( list( zip( image_files, previous_image_files ) ) )

		args = zip_longest(
		    cameras, repeat(ts, len(cameras)),
		    image_files, previous_image_files, repeat(config, len(cameras)), 
		    repeat(reprocess, len(cameras))
		)
		print( "Prepare image set" )
		pool.map( 
		    prepare_image,
		    args
		)

		# do preprocessing which requires all images for the current 
		# timestamp to have been processed 
		# 	(cloud height, features, stitch)
		if len(timestamps) == num_cores:
			pool.map( process_image_set, zip(timestamps, repeat(config), repeat(reprocess)) )
			timestamps = []
		if not first:
			timestamps.append( ts )
		else:
			first = False

	if len(timestamps):
		pool.map( process_image_set, zip(timestamps, repeat(config), repeat(reprocess)) )
	pool.close()
	pool.join()

#		if images.skip_bc_night or len(images.images) == 0:
#			last_images = None
#			continue

#		# don't bother loading the data for timestamps which have 
#		# already been processed unless we need it to process 
#		# the succeeding timestamp
#		if images.complete or not last_images:
#			last_images = images
#			last_images.minimize_memory_load()
#			continue
#
#		# store only the values that are actually needed to process the
#		# succeeding timestamp
#		last_images = images
#		last_images.minimize_memory_load()

def load_camera( c_id, config ):
	return cam.Camera( c_id, config, 7.*np.pi/18, img_w, img_h )

def create_camera_objects( config ):
	camera_ids = config["cameras"][site]["all_cameras"]

	cameras = {
	    c_id: cam.Camera(
	      c_id, config, 7.*np.pi/18, img_w, img_h
	    ) for c_id in camera_ids
	}
	return cameras

# determine which parts of the processing sequence need to be redone
def build_reprocess(config):
	dependencies = {
	    "features": ["stitch"],
	    "stitch": ["height", "motion"],
	    "motion": ["image"],
	    "height": ["image"],
	    "image": []
	}
	fill_dependencies( dependencies, "features" ) # modifies dependencies in place

	pl_c = config["pipeline"]
	reprocess = {
	    "features": pl_c["features"]["reprocess"],
	    "height": pl_c["height"]["reprocess"],
	    "motion": pl_c["motion"]["reprocess"],
	    "stitch": pl_c["stitch"]["reprocess"],
	    "image": pl_c["image_preprocessing"]["reprocess"]
	}
	reprocess["all"] = (pl_c["reprocess_all"] or
				all(v for v in reprocess.values()))
	reprocess["any"] = any(v for v in reprocess.values())

	for key in dependencies.keys():
		reprocess[key] = (
		    reprocess["all"] or
		    reprocess[key] or
		    any(reprocess[val]
		      for val in dependencies[key])
		)
	print( "reprocess: " + str(reprocess) )
	return reprocess

def fill_dependencies(dependencies, root):
	arr = dependencies[root]
	for val in dependencies[root]:
		arr += fill_dependencies( dependencies, val )
	# no dups
	arr = list(set(arr))
	dependencies[root] = arr
	return arr

if __name__ == "__main__":
	try:
		from config_handler import handle_config
		from logger import Logger
	except:
		print( "Failed to import config_handler or logger\n" +
		   "They are located in ../tools/" )
		exit( 1 )

	cp = handle_config(
	    metadata={"invoking_script": "preprocess"},
	    header="pipeline"
	)
	logger = Logger( "forecast_pipeline", cp )
	preprocess( cp, logger )
