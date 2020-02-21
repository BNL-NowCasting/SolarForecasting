### inventory.py
# every hour count up the new photos taken by each camera 
#    and add an entry to inventory.csv
# TODO: redo inventory for given time range 
# 	(e.g if we sync data from another machine, 
# 	    update inventory for only those hours)
###

from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import time
import traceback

try:
    from config_handler import handle_config
    from logger import Logger
    import utils
except:
    print( "Failed to import config_handler, logger or utils.\n" + 
           "They are located in ../tools/\n" )
    print( traceback.format_exc() )
    exit( 1 )

cp = handle_config(
    metadata={"invoking_script":"hist_inventory"}, 
    header="inventory"
)
logger = Logger( "hist_inventory", cp )

### load configuration
SITE = cp["site_id"]
CAMERAS = cp["cameras"][SITE]["cameras"]
NUM_CAMERAS = len(CAMERAS)

# how long to sleep once we run out of images to log
SLEEP_LEN = cp["inventory"]["hist"]["sleep_time"]

ROOT = cp["paths"][SITE]["img_path"]
CACHE_ROOT = cp["paths"][SITE]["cache_path"]

FOLDER_NAMES = [ROOT + x + "/" for x in CAMERAS]
CACHE_FOLDER_NAMES = [CACHE_ROOT + x + "/" for x in CAMERAS]

# location of the inventory.csv files
INVENTORY_ROOT = cp["paths"][SITE]["inventory_path"] 

# the code as is depends on the timestamp column being 'hour'
HEADER = "hour," + ",".join(CAMERAS).lower()
HEADER_COLS = ["hour"] + CAMERAS

if cp["inventory"]["start_date"]:
	next_t = cp["inventory"]["start_date"]
else:
	# default to the start of the current year
	next_t = datetime.now().year
next_t = utils.date_from_timestamp( next_t )

if cp["inventory"]["end_date"]:
	end_t = cp["inventory"]["end_date"]
	end_t = utils.date_from_timestamp( end_t )
else:
	# default to running forever as a daemon
	end_t = None

### helper methods
def list_files( day ):
	file_lists = []
	for path in FOLDER_NAMES:
		p = path + day + "/"
		# if day hasn't happened yet or 
		# a camera wasn't collecting on day
		# there won't be a folder for day
		if os.path.isdir( p ):
			file_lists.append( os.listdir(p) )
		else:
			file_lists.append( [] )
	return file_lists

def list_cached_files( ):
	file_lists = []
	for path in CACHE_FOLDER_NAMES:
		file_lists.append( os.listdir(path) )
	return file_lists

def full_path( folder, file_n, day, use_cache=False ):
	if use_cache:
		return os.path.join( folder + "/", file_n )
	else:
		return os.path.join( folder + day + "/", file_n )

def finish_inv():
	print( "Finished logging up to {}. Exiting.".format( end_t ) )
	exit(0)

# construct and return an array representing an entry in inventory*.csv
def count_folders( hour, file_lists, use_cache=False ):
	day = hour[:-2]

	folder_names = CACHE_FOLDER_NAMES if use_cache else FOLDER_NAMES
	count = [""] + ['0']*NUM_CAMERAS

	for i, folder in enumerate(folder_names):
		file_list = [
		    f for f in file_lists[i]
		    if f[-18:-8] == hour # fragile
		    and os.path.isfile(full_path(folder, f, day, use_cache)) 
		]

		count[i+1] = str(len(file_list))
		if len(file_list) and not count[0]:
			count[0] = day
	return count


Path( INVENTORY_ROOT ).mkdir( parents=True, exist_ok=True )

# if there is an inventory file for the target year 
# continue logging from the point where we left off earlier
inventory_name = INVENTORY_ROOT + "inventory_{}.csv".format(next_t.year)
if not os.path.exists( inventory_name ):
	with open( inventory_name, "w" ) as inventory:
		 inventory.write( HEADER + "\n" )

old_data = pd.read_csv(inventory_name)
if len(old_data.index):
	t = old_data.loc[len(old_data.index)-1]['hour']
	next_t = datetime.strptime( str(t), "%Y%m%d%H" ) + timedelta(hours=1)

### main loop
last_day = None
last_year = next_t.year

file_lists = []
while True:
	try:
		counts = pd.DataFrame( columns=HEADER_COLS )

		max_t = datetime.now()
		last_success = next_t # the last hour where we found data
		last_counts = 0 # size of counts during last_success

		use_cache = False
		while True:
			if end_t and next_t >= end_t:
				counts = counts[0:last_counts]
				break

			if next_t >= max_t:
				next_t = last_success
				counts = counts[0:last_counts]
				last_day = None
				
				if not use_cache:
					# reload our files list from the cache
					use_cache = True
					continue
				break
			
			# when we close out a year, we switch to a new inv file
			# so, we need to write our counts df before continuing
			if next_t.year != last_year:
				last_year = next_t.year
				break

			day = next_t.strftime( "%Y%m%d" )
			if day != last_day:
				if not use_cache:
					file_lists = list_files( day )
				else:
					file_lists = list_cached_files()

				last_day = next_t.strftime( "%Y%m%d" )

			next_update_str = next_t.strftime( "%Y%m%d%H" )

			# count the number of images taken by each camera
			# during the hour after next_t (UTC)
			count = count_folders(
			    next_update_str, file_lists, use_cache=use_cache
			)
			counts.loc[len(counts.index)] = count

			if count[0]:
				if next_t - last_success > timedelta(hours=1):
					logger.log( 
					    "Data missing starting after " +
					    last_success.strftime( "%Y-%m-%d %H:%M" )
					)
					logger.log( 
					    "Data resumes starting at " +
					    next_t.strftime( "%Y-%m-%d %H:%M" )
					)	

				last_success = next_t
				last_counts = counts.index[-1]
			counts.loc[counts.index[-1]][0] = next_update_str

			next_t += timedelta( hours=1 )

		counts.to_csv(inventory_name,mode='a',header=False,index=False)
	except Exception as e:
		logger.log_exception()
		print( "Caught exception {}".format(e) )
	finally:
		if end_t and next_t >= end_t:
			finish_inv()
		else:
			time.sleep(SLEEP_LEN)
