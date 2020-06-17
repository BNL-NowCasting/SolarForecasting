import numpy as np
#import matplotlib
#matplotlib.use('agg')
#from matplotlib import pyplot as plt
import configparser
import os
import glob
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
#from datetime import datetime, timezone, timedelta
from ast import literal_eval as le
import pytz
import pandas as pd

try:
    config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)
except Exception:
    print("No config.conf found")

try:
    time_int = sys.argv[1]
except Exception:
    print("No time interval provided, using all in path")
    
try:
    GHI_path=le(cp["paths"]["GHI_path"])
    print("Using config.conf path ("+GHI_path+")")
except Exception:
    try:
        GHI_path = sys.argv[2]
        print("Using provided path ("+GHI_path+")")
    except Exception:
        print("No path provided, using current path ("+os.getcwd()+")")
        GHI_path = os.getcwd()
         
try:
    local_tz=pytz.timezone(cp["GHI_sensors"]["GHI_timezone"])
    print("Using config.conf timezone: %s" % str(local_tz))
except Exception:
    try:
        local_tz = sys.argv[3]
        print("Using provided timezone: %s" % str(local_tz))
    except Exception:     
        local_tz=pytz.timezone("utc")    
        print("No timezone provided, assuming UTC")

try:
    GHI_Coor = le(cp["GHI_sensors"]["GHI_Coor"])
    GHI_loc=[GHI_Coor[key] for key in sorted(GHI_Coor)]
    GHI_loc=np.array(GHI_loc)
except Exception as e:
    print("Error loading GHI locations: %s" % e)


flist = glob.glob(GHI_path+'*/*.npz')

for fn in flist:
    print("Processing GHI file: %s" % fn)
    
    try:
        with np.load(fn) as data:
            ty, y = data['timestamp'], data['ghi']

        y[np.isnan(y)] = 0  #replace nans with 0s, not a universal solution

        try:
            sensor = int(os.path.split(fn)[1][4:-4])
            loc = Location(GHI_loc[sensor][0], GHI_loc[sensor][1], local_tz)
            times = pd.DatetimeIndex(pd.to_datetime(ty, unit='s'))
            max_ghi = list(loc.get_clearsky(times)['ghi'])
            max_dni = list(loc.get_clearsky(times)['dni'])
            max_dhi = list(loc.get_clearsky(times)['dhi'])
        except Exception as e:
            print("Error calculating max GHI, omitting: %s" % e)

        out_data = np.column_stack((ty, y, max_ghi, max_dni, max_dhi))

        d, n = os.path.split(fn)
        np.savetxt(os.path.join(d,n[:-3]+"csv"), out_data, fmt="%i, %f, %f, %f, %f", delimiter=',', header="Timestamp,Actual_GHI,Calc_GHI, Calc_DNI, Calc_DHI")
    except Exception as e:
        print("Error: %s" % str(e))
        
print("Done.")