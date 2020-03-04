from datetime import datetime, timedelta
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import traceback
import xarray as xr

date = sys.argv[1]
year = int(date[:4])
month = int(date[4:6])
day = int(date[6:])


fig = plt.figure()
ax = plt.subplot(121)

### This whole block is setting up properly displaying dates on the x-axis
### The automatic formatting for dates get bad if you zoom in, so I overrode it
### I sort of tried to make it less fragile, but it'll likely break if we update
#plt.xticks(rotation=15)
#x_locator = dates.AutoDateLocator()
#ax.xaxis.set_major_locator( x_locator )
## print( dates.AutoDateFormatter( x_locator ).__dict__ )
#formatter = dates.AutoDateFormatter( x_locator )
#for k,v in formatter.scaled.items():
#        if v == "%H:%M:%S":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#        elif v == "%H:%M:%S.%f":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#        elif v == "%d %H:%M":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#        elif v == "%M:%S.%f":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#        elif v == "%m-%d %H":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#        elif v == "%H:%M:%S.%f":
#                formatter.scaled[k] = "%b %d %Y %H:%M:%S"
#ax.xaxis.set_major_formatter( formatter )
### end

truth_df = xr.open_dataset( '/home/tchapman/old_root/lidar_data/BNL_lidars_CBH/netcdfs/BNL_lidars_CBH_{}.nc'.format(date) ).to_dataframe()

print( truth_df.columns )
print( list(truth_df["CEIL_time"])[:5] )
truth_df = truth_df.iloc[::30]
times = truth_df["CEIL_time"][truth_df["CEIL_cbh"] >= 0] 
truth = truth_df["CEIL_cbh"][truth_df["CEIL_cbh"] >= 0]

ax.scatter( times, truth, c='red', s=10 )

pred_df = pd.read_csv( '/home/tchapman/root/data/bnl/height_estimation_.csv' )
pred_df = pred_df[pred_df.apply(lambda x : str(x["timestamp"][:4]) + str(x["timestamp"][5:7]) + str(x["timestamp"][8:10]) == date,axis=1)]
times = pd.to_datetime( pred_df["timestamp"], format="%Y-%m-%d %H:%M:%S" ) - datetime( year, month, day ) - timedelta(hours=5)
times = [t.seconds / 3600. for t in times]
pred = pred_df["avg"]

ax.scatter( times, pred, c='blue', s=20 )
plt.legend( ["truth", "predicted"], loc="upper right" )

plt.show()
