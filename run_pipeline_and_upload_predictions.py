from tools.config_handler import handle_config
from tools.logger import Logger
import traceback

#KEY = "test_methods_1"
KEY = ""
# Automate the whole pipeline of taking images and the actual GHI,
# stitching and extracting features from the images,
# generating our forecast and a persistence forecast,
# and posting both to sfa
# and then creating a report on sfa about the relative performances

# run python3 run_pipeline_and_upload_predictions.py -h for usage details
config = handle_config(
    metadata={"invoking_script": "run_pipeline"},
    header="pipeline"
)
logger = Logger( "forecast_pipeline", config )

### PREPROCESS
# preprocess the target images
# extract features to csv files
from realtime_forecasting.preprocessing.preprocess import preprocess
print( "Start preprocess" )
preprocess( config, logger, KEY=KEY )

### FORECAST
# make forecasts based on these preprocessed images and actual GHI
from realtime_forecasting.forecasting.forecast import forecast
print( "Start forecast" )
#forecast( config, logger, KEY=KEY )

### POST
# create and populate an sfa forecast using our forecast
# create and populate an observation run using the actual GHI
# request a report
#from sfa_api import upload_forecast
#upload_forecast( config, logger )

print( "End pipeline" )
