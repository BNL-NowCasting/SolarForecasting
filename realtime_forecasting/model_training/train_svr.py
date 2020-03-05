from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tools.utils import date_from_timestamp, quick_plot

from sklearn.preprocessing import normalize, scale

def train( config, logger, KEY="" ):
	target_ranges = sorted( zip(
		config["pipeline"]["target_ranges"]["start_dates"],
		config["pipeline"]["target_ranges"][ "end_dates" ]
	), key=lambda x: x[0] )

	def timestamps_to_consider():
		next_t = datetime.strptime( '0001', '%Y' )
		for (start, end) in target_ranges:
			next_t = max(next_t, date_from_timestamp(start) + timedelta(seconds=30))
			end_t = date_from_timestamp( end )

			while next_t < end_t:
				yield next_t
				next_t += timedelta( seconds=30 )

	site = config["site_id"]
	feature_path = config["paths"][site]["feature_path"]
	forecast_path = config["paths"][site]["forecast_path"]
	ghi_path = config["paths"][site]["ghi_path"]

	sensors = list( config["ghi_sensors"]["coords"].keys() )
	lead_times = config["pipeline"]["lead_times"]

	last_month = None
	for sensor in sensors:
		data = None
		# Load extracted features and preferred model
		# Pass image features through model and save prediction
		for timestamp in timestamps_to_consider():
			# the ghi files contain a month of data
			# no need to load them 90000 times each
			month = timestamp.strftime( "%Y-%m" )
			if month != last_month:
				ground_truth = None
				fn = "{}{}/{}.csv".format( ghi_path, month, sensor )
				print( fn )
				if os.path.exists(fn):
					ground_truth = pd.read_csv( fn )
				else:
					print ("nope" )
				last_month = month

			day = timestamp.strftime( "%Y%m%d" )
			fn = feature_path + "{}{}/{}_{}.csv".format(KEY, day, sensor, timestamp)
			if not os.path.exists( fn ):
				print( "Skipping {} because {} does not exist".format( timestamp, fn ) )
				continue
			if data is not None:
				data = pd.concat( [data, pd.read_csv( fn )] )
			else:
				data = pd.read_csv( fn )

		if data is None:
			continue

		data.reset_index( inplace=True, drop=True )
		data.drop(data.columns[0], axis=1, inplace=True) # this should go away eventually; rn the way I do preprocessing, index becomes an unnamed column
		for lt in lead_times:
			print( sensor, lt )
			model = None # pickle.load( ... )
			with open('/home/tchapman/root/data/bnl/pickles/models/optimal_model{:02d}.mod99'.format(lt),'rb') as fmod:
				model = pickle.load(fmod)

			lt_data = data.loc[data["lead_time"] == lt].reset_index(drop=True)
			#predictions = model.run( lt_data )
			print( lt_data )
			if len(lt_data.index) == 0:
				print( "No data for lead time  "+ str(lt) )
				continue
			tmp = lt_data.drop( "lead_time", axis=1 )
			#print( ground_truth.columns )
			tmp = pd.merge( tmp, ground_truth[["timestamp", "ghi"]], how='left', left_on=["time_str"], right_on=["timestamp"] )
			tmp.drop( ["time_str", "timestamp"], inplace=True, axis=1 )
			tmp["ghi"] /= 400
			tmp[tmp.columns[1:]] = scale( tmp[tmp.columns[1:]] )
			cols = list(tmp.columns[-1:]) + list(tmp.columns[:-1])
			tmp = tmp[cols]
			#print( tmp.columns )

			forecast = model.predict( tmp )
			final_timestamps = list(lt_data["time_str"])
			lead_time_col = list(lt_data["lead_time"])
			for i, ts in enumerate( final_timestamps ):
				final_timestamps[i] = ( datetime.strptime( str(ts), "%Y%m%d%H%M%S" ) + timedelta( minutes=int(lead_time_col[i]) ) ).strftime( "%Y%m%d%H%M%S" )
			lt_data = lt_data.assign(final_timestamp=final_timestamps)
			
			
			#print( forecast )
			#print( list(lt_data["final_timestamp"]) )

			truth = np.empty( (len(lt_data.index)) )
			truth.fill(np.nan) 
			if ground_truth is not None:
				match = ground_truth["timestamp"].map(
				    lambda ts: str(ts) in list(lt_data["final_timestamp"])
				)
				truth = ground_truth.loc[match]["ghi"].reset_index(drop=True)
			print( truth )

			print( lt_data )	
			predictions = pd.DataFrame()
			predictions = predictions.assign(pred_ghi=forecast)
			predictions = predictions.assign(timestamp=lt_data["time_str"])
			#predictions = predictions.assign(target_timestamp=lt_data["final_timestamp"])
			predictions = predictions.assign(truth=truth)

			output_file = forecast_path + day + "/{}_{}_{}.csv".format(sensor, lt, timestamp)
			predictions.to_csv( output_file )

			fig = plt.figure()
			ax = fig.add_subplot(111)

			### This whole block is setting up properly displaying dates on the x-axis
			### The automatic formatting for dates get bad if you zoom in, so I overrode it
			### I sort of tried to make it less fragile, but it'll likely break if we update
			plt.xticks(rotation=15)
			x_locator = dates.AutoDateLocator()
			ax.xaxis.set_major_locator( x_locator )
			# print( dates.AutoDateFormatter( x_locator ).__dict__ )
			formatter = dates.AutoDateFormatter( x_locator )
			for k,v in formatter.scaled.items():
				if v == "%H:%M:%S":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
				elif v == "%H:%M:%S.%f":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
				elif v == "%d %H:%M":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
				elif v == "%M:%S.%f":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
				elif v == "%m-%d %H":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
				elif v == "%H:%M:%S.%f":
					formatter.scaled[k] = "%b %d %Y %H:%M:%S"
			ax.xaxis.set_major_formatter( formatter )
			### end

			xs = pd.to_datetime(predictions.loc[:,'timestamp'].astype(int),format="%Y%m%d%H%M%S").tolist()
			preds = forecast
			print("xs " + str(xs) )
			print("preds " + str(preds) )
			print("truth " + str(list(truth) ))
			ax.scatter(
			    xs, preds, c='r',
			    edgecolors='r',  s=float(20)
			)
			xs = pd.to_datetime(ground_truth['timestamp'].astype(str),format="%Y%m%d%H%M%S").tolist() 
			slc = [a.strftime( "%Y%m%d" ) == "20191205" for a in xs]
			xs = [ x-timedelta(minutes=lt) for [i,x] in enumerate(xs) if slc[i]]
			truth = [ ghi for [i,ghi] in enumerate(ground_truth["ghi"]) if slc[i]]
			ax.scatter(
			    xs, truth, c='b',
			    edgecolors='b',  s=float(20)
			)

			plt.title( "Estimated GHI for sensor {}, {} minutes in advance.".format(sensor, lt ) )
			plt.xlabel( "Date" )
			plt.ylabel( "GHI" )
			plt.legend( ["Prediction", "Truth"], loc="upper right" )
			#ax = fig.add_subplot(122)
			#ghis = ground_truth["ghi"]
			#xs = pd.to_datetime(ground_truth['timestamp'].astype(str),format="%Y%m%d%H%M%S").tolist() 
			#slc = [a.strftime( "%Y%m%d" ) == "20200101" for a in xs]
			#xs = [ x for [i,x] in enumerate(xs) if slc[i]]
			#ghis = [ ghi for [i,ghi] in enumerate(ghis) if slc[i]]
			#plt.scatter(xs, ghis)

			plt.show()


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
	forecast( cp, logger )
