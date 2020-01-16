import pandas as pd
import numpy as np
import stat_tools as st
import matplotlib.pyplot as plt
import datetime as dt
import os, glob

Sensors = np.arange(1,26);
DirPath = '~/ldata/originalGHI/'
outPath = '~/ldata/GHI/'

str2seconds = lambda x: (dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')-dt.datetime(2018,1,1)).total_seconds()+3600*5

if not os.path.isdir(outPath):
    os.makedirs(outPath)
    os.chmod(outPath,0o755)

for sensor in Sensors:
    flist = glob.glob(DirPath + '2018-??_bps_' + str(sensor) + '_second.dat')
    for fn in flist:
        df = pd.read_csv(fn, sep = ',', header = 0, skiprows = [0, 2, 3], usecols = ['TIMESTAMP', 'SP2A_H'], converters={'TIMESTAMP':str2seconds})
        df = df.values;
        df[df[:,1]<=5, 1] = np.nan

        bins = np.arange(df[0,0], df[-1,0], 60); timestamp = 0.5 * (bins[1:] + bins[:-1])
        ghi = st.bin_average_reg(df[:,1], df[:,0], bins);

#         plt.figure(); plt.plot(timestamp/3600,ghi); plt.show()
        np.savez(outPath+'GHI_'+str(sensor), timestamp=timestamp,ghi=ghi);

