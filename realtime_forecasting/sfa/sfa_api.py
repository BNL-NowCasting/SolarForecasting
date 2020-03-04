import subprocess
import glob
import json

from config_handler import handle_config

cp = handle_config()

SITE_IDS = cp["sfa"]["site_ids"]

def auth():
	json_str = str(subprocess.check_output( "/home/tchapman/root/api_methods/auth" ))

	if len(json_str < 5 ):
		print( "Something went wrong with auth; received response '{}'".format(json_str) )
		exit(1)

	resp = json.loads( json_str[2:-3] )
	if not "access_token" in resp:
		print( "Did not receive access_token; response: '{}'".format(json_str) )
		exit(1)

	return resp["access_token"] 

def create_forecast( params ):
# def create_forecast(name, site_id, lead_time_to_start, run_length=0, interval_length=0.5, variable="ghi", interval_label="instant", interval_value_type="instantaneous", issue_time_of_day="00:00" ):
	if "run_length" not in params:
		params["run_length"] = 0
	if "interval_length" not in params:
		params["interval_length"] = 0.5
	if "interval_label" not in params:
		params["interval_label"] = "instant"
	# etc...

	keys = [
	    "interval_label", "interval_length", "interval_value_type", 
	    "issue_time_of_day", "lead_time_to_start", "name", "run_length", 
	    "site_id", "variable"
	]

	args_array = ["/home/tchapman/root/api_methods/create_forecast"] + [str(params[k]) for k in keys]
	f_id = subprocess.check_output( args_array )
	# TODO parse f_id out of response
	print( "created forecast with response: " + str(f_id) )
	return f_id

def upload_forecasts( config, logger, auth_token="" ):
	file_sets = config["pipeline"]["target_days"]
	site_name = config["pipeline"]["site_name"]
	
	if not auth_token:
		auth_token = auth()

	file_names = []
	for fs in file_sets:
		file_names.extend( glob.glob(fs) )
	# TODO don't upload all_forecast files
	
	site_id = SITE_IDS[site_name]
	print( "Site_id: " + str(site_id) )

	success = []
	for f in file_names:
		print( "Considering file " + str(f) )
		
		# [{day}, {forecast_type}, {ghi}, {sensor}, {lead_time}min.csv]
		pieces = f.split( "_" )
		pieces[0] = pieces[0].split("/")[-1]
		pieces[-1] = pieces[-1][:-7] # {lead_time}
		# {site_name} sensor {sensor} {forecast_type} {lead_time} min ahead GHI {day}*
		forecast_name = "{} sensor {} {} {} min ahead GHI {}*".format(
		    site_name, pieces[3], pieces[1], pieces[-1], pieces[0]
		)
		print( "forecast name: " + forecast_name )
		forecast_id = create_forecast({
		    "name": forecast_name, "site_id": site_id, 
		    "lead_time_to_start": pieces[-1]
		})

	#	p = subprocess.Popen( ["/home/tchapman/root/api_methods/upload_forecast", forecast_id, f, auth_token] )
	#	while p.poll() is None:
	#		time.sleep( 0.1 )
	#	success.append( p.return_code == 0)
	#	print( "success is " + str(success[-1]))
	return success

if __name__ == "__main__":
	from config_handler import handle_config
	from logger import Logger
	config = handle_config(metadata={"invoking_script":"sfa_api"}, header="pipeline")
	logger = Logger( "forecast_pipeline", cp )
	# logger.log( "Starting forecast upload with configuration(s): " + str(config["metadata"]["name"]) )

	# auth()
	upload_forecast( config, logger )
