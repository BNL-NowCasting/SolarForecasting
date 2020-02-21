#!/usr/bin/python3
from datetime import datetime
import os
import pickle

def date_from_timestamp( ts, snap_left=True ):
	default_ts = "00000101000000" if snap_left else "00001231235959"
	ts = str(ts) + default_ts[len(str(ts)):]
	ts = datetime.strptime( ts, "%Y%m%d%H%M%S" )
	return ts

# instead of running an expensive method
# try to load the result from a pickle file
# and if that fails, run the method and save the result to a pickle file
def try_pickle( file_path, call_back, args=[], save_results=True ):
	if os.path.exists( file_path ):
		with open( file_path, 'rb' ) as fh:
			return pickle.load(fh)
	else:
		obj = call_back(*args)
		if save_results:
			with open( file_path, 'wb' ) as fh:
				pickle.dump( obj, fh, pickle.HIGHEST_PROTOCOL )
		return obj
