from . import camera as cam
from datetime import datetime, timedelta
#from .less_horrible_image import ImageSet
#from .image_set import create_set
from .image_set import ImageSet
import matplotlib.pyplot as plt
import numpy as np
import os
import tools.utils as utils

last_images = None

def preprocess( config, logger, KEY="" ):
	global last_images

	site = config["site_id"]
	img_path = config["paths"][site]["img_path"]
	pickle_path = config["paths"][site]["pickle_path"]

	target_ranges = sorted( zip(
		config["pipeline"]["target_ranges"]["start_dates"],
		config["pipeline"]["target_ranges"]["end_dates"]
	), key=lambda x: x[0] )

	cameras = config["cameras"][site]["all_cameras"]

	# img dimensions; really ought to be renamed...
	img_w = config["cameras"]["img_w"]
	img_h = config["cameras"]["img_h"]

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
		[cameras, config, img_w, img_h]
	)

	for ts in timestamps_to_consider():
		### Load, undistort, and cloud mask image
		#images = create_set(ts,last_images,camera_dict,config,KEY=KEY)
		images = ImageSet(ts,last_images, camera_dict, config, KEY=KEY)
		if images.skip_bc_night:
			last_images = None
			continue
		if images.complete:
			continue

		logger.log( "Undistorted images for timestamp {}".format(ts) )

		if not last_images:
			last_images = images
			continue
		last_images = images

		### Estimate cloud layer heights in image
		heights = images.cloud_height()
		height = images.cloud_base_height
		logger.log( "Estimated cloud layer heights as {} from {}".format( height,heights ))

		### Estimate cloud layer velocities in image
		# [[vx, vy, corr]]
#		layer_vels = images.cloud_motion()
#		vel = images.cloud_base_vel
#		logger.log( "Estimated cloud layer velocities as {} from {}".format( vel,layer_vels ))

		### Perform image stitching
#		stitch = images.stitch()

		### Perform feature extraction
#		features = images.extract_features()
	print( "end " )

def create_camera_objects( camera_ids, config, img_w, img_h ):
	cameras = {
	    c_id: cam.Camera(c_id, config, 7.*np.pi/18, img_w, img_h) for c_id in camera_ids
	}
	return cameras

if __name__ == "__main__":
	try:
		from config_handler import handle_config
		from logger import Logger
	except:
		print( "Failed to import config_handler or logger\n" +
		   "They are located in ../tools/\n" )
		exit( 1 )

	cp = handle_config(
	    metadata={"invoking_script": "preprocess"},
	    header="pipeline"
	)
	logger = Logger( "forecast_pipeline", cp )
	preprocess( cp, logger )
