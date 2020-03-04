#!/usr/bin/python3

def date_from_timestamp( ts, snap_left=True ):
	from datetime import datetime
	default_ts = "00000101000000" if snap_left else "00001231235959"
	ts = str(ts) + default_ts[len(str(ts)):]
	ts = datetime.strptime( ts, "%Y%m%d%H%M%S" )
	return ts

def quick_plot( imgs, save="" ):
	from matplotlib import pyplot as plt
	f = plt.figure()
	for i in range( len(imgs) ):
		f.add_subplot( 1, len(imgs), i+1 )
		plt.imshow( imgs[i] )

	if save:
		plt.savefig( save )
		plt.close()
	else:
		plt.show()

def execute_multi_processing( method, args, num_cores=10 ):
	import multiprocessing
	pool = multiprocessing.Pool( num_cores )
	pool.map(method, args)
	pool.close()

# instead of running an expensive method
# try to load the result from a pickle file
# and if that fails, run the method and save the result to a pickle file
def try_pickle( file_path, call_back, args=[], save_results=True ):
	import pickle
	import os
	if os.path.exists( file_path ):
		with open( file_path, 'rb' ) as fh:
			return pickle.load(fh)
	else:
		obj = call_back(*args)
		if save_results:
			with open( file_path, 'wb' ) as fh:
				pickle.dump( obj, fh, pickle.HIGHEST_PROTOCOL )
		return obj
