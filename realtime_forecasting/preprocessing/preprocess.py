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
                print( "No matches found for camera {} at {}{}".format(
                    cam, minute, second
                ) )
                return None
        return filtered_files[0]

def prepare_image_set(args):
	print( "Prepare " + image_file )
	c_id, ts, image_file, previous_image_file, pickle_path = args
	image = Image( c_id, image_file, previous_image_file, pickle_path )

	print( "Undistort" )
	image.undistort()
	if image.error:
		print( "{} Image undistort failed: {}".format(
		    c_id, image.error
		) )
		# log_bad_image(image)
		return

	print( "Mask" )
	image.cloud_mask()

	print( "Motion" )
	image.cloud_motion()

	image.dump_self()

def process_image_set(args):
	#ts, cam_ids, image_files, pickle_path = args
	return

	image_set = ImageSet( ts, cam_ids, image_files, pickle_path )
	image_set.cloud_height()
	image_set.stitch()
	image_set.dump_self()
	image_set.extract_features()

def preprocess( config, logger, KEY="" ):
	global last_images

	site = config["site_id"]
	img_path = config["paths"][site]["img_path"]
	pickle_path = config["paths"][site]["pickle_path"]

	num_cores = config["pipeline"]["cores_to_use"]
	cameras = config["cameras"][site]["all_cameras"]

	target_ranges = sorted( zip(
		config["pipeline"]["target_ranges"]["start_dates"],
		config["pipeline"]["target_ranges"]["end_dates"]
	), key=lambda x: x[0] )

	def timestamps_to_consider():
		global last_images

		next_t = datetime.strptime( '0001', '%Y' )
		for (start, end) in target_ranges:
			# start a step behind start so we can have last_images set when we start
			start_t = utils.date_from_timestamp( start ) - timedelta(seconds = 30 )
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
	camera_dict = utils.try_pickle(
		pic_path,
		create_camera_objects,
		[config]
	)

	pool = multiprocessing.Pool( num_cores )

	# Process in chunks as a compromise between preprocessing all images,
	# then doing all clouds heights, then all feature extraction etc. and 
	# fully processing each timestamp in order
	ts_gen = timestamps_to_consider()
	timestamps = []

	file_lists = {}
	last_day = None
	for ts in ts_gen:
		day = ts.strftime( "%Y%m%d" )
		if day != last_day:
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
			last_day = day
		# do preprocessing which only depends on previous images
		# having been preprocessed (undistort, cloud mask, cloud motion)
		image_files = [
		    get_next_image(img_path, cam, ts, file_lists) 
		    for cam in cameras
		]
		previous_image_files = [get_next_image(
		    img_path, cam, ts-timedelta(seconds=30), file_lists
		) for cam in cameras] # breaks at midnight but GHI is 0 at midnight
		pool.map( 
		    prepare_image_set,
		    zip_longest(
		      cameras, repeat(ts),
		      image_files, previous_image_files, repeat(pickle_path)
		    )
		)

		# do preprocessing which requires all images for the current 
		# timestamp to have been processed 
		# 	(cloud height, motion, features, stitch)
		if len(timestamps) == num_cores:
			pool.map( process_image_set, zip_longest(timestamps) )
			timestamps = []
		timestamps.append( ts )

	if len(timestamps):
		pool.map( process_image_set, zip_longest(timestamps) )
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
