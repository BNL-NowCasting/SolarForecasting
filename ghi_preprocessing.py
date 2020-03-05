from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
from itertools import product
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import tools.stat_tools_2 as st
import tools.utils as utils
import traceback

def process_ghi( config, logger ):
	global ghi_path
	global processed_ghi_path

	sensors = list(config["ghi_sensors"]["coords"].keys())
	site = config["site_id"]
	ghi_path = config["paths"][site]["raw_ghi_path"]
	processed_ghi_path = config["paths"][site]["ghi_path"]

	target_ranges = sorted( zip(
		config["pipeline"]["target_ranges"]["start_dates"],
		config["pipeline"]["target_ranges"]["end_dates"]
	), key=lambda x: x[0] )

	def months_to_consider():
		next_t = datetime.strptime( '0001', '%Y' )
		for (start, end) in target_ranges:
			next_t = max(next_t, utils.date_from_timestamp( start ))
			print( "Set next_t to " +str(next_t ) )
			end_t = utils.date_from_timestamp( end )

			while next_t < end_t:
				yield next_t.strftime("%Y-%m")
				next_t += relativedelta( months=1 )

	p = multiprocessing.Pool(10)

	files = product( list(months_to_consider()), sensors )
	p.map( process_sensor, files )

	p.close()
	p.join()

def process_sensor( pairing ):
	(month, sensor) = pairing
	print( month, sensor )
	fn = "{}{}/{}_bps_{}_second.dat".format(
	    ghi_path, month, month, sensor
	)
	print( fn )
	if not os.path.exists( fn ):
		print( "File does not exist" )
		return

	# group into 30 second blocks
	format_timestamp = lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5)).strftime( "%Y%m%d%H%M" ) + ("00" if int(x[17:19]) < 30 else "30" )
	ghi = pd.read_csv( 
	    fn, header=0, skiprows=[0,2,3], 
	    usecols=["TIMESTAMP", "SP2A_H"], 
	    converters={"TIMESTAMP":format_timestamp}
	)

	ghi.loc[ghi["SP2A_H"] <= 5, "SP2A_H"] = np.nan
	ghi = ghi.groupby( "TIMESTAMP" ).mean()
	#ghi = ghi.groupby("TIMESTAMP").agg(np.nanmean)
	
	ghi.reset_index(inplace=True)
	ghi.columns = ["timestamp", "ghi"]
	out_fn = "{}{}/{}.csv".format(
	    processed_ghi_path, month, sensor
	)
	if not os.path.exists( processed_ghi_path + month ):
		os.mkdir( processed_ghi_path + month )
	ghi.to_csv( out_fn )

if __name__ == "__main__":
	from tools.config_handler import handle_config
	from tools.logger import Logger

	cp = handle_config(
	    metadata={"invoking_script": "preprocess"},
	    header="pipeline"
	)
	logger = Logger( "forecast_pipeline", cp )
	print( " attempt to process ghi" )
	process_ghi( cp, logger )

