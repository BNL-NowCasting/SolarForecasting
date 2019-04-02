import pandas as pd
import numpy as np
import stat_tools as st
import matplotlib.pyplot as plt
import datetime as dt
import os, glob

Sensors = ['spower'];
DirPath = '~/ldata/originalGHI/'
outPath = '~/ldata/GHI/'

str2seconds = lambda x: (dt.datetime.strptime(x, '%m/%d/%Y %H:%M')-dt.datetime(2018,1,1)).total_seconds()+3600*4

if not os.path.isdir(outPath):
    os.makedirs(outPath)
    os.chmod(outPath,0o755)

for sensor in Sensors:
    flist = glob.glob(DirPath + '2018_' + sensor + '_5min.dat')
    for fn in flist:
        print(fn)
        df = pd.read_csv(fn, sep = ',', usecols = [0,1], converters={0:str2seconds})
        df = df.values;
        df[df[:,1]<=5, 1] = np.nan
        
        t1 = (dt.datetime(2018,8,1)-dt.datetime(2018,1,1)).total_seconds()
        df = df[df[:,0]>t1]
        
        bins = np.arange(t1, df[-1,0], 60); timestamp = 0.5 * (bins[1:] + bins[:-1])
        ghi = np.interp(timestamp,df[:,0],df[:,1])
        plt.figure(); plt.plot((timestamp%86400)/3600,ghi,'ro'); plt.show();        
#         ghi = st.bin_average_reg(df[:,1], df[:,0], bins);
# 
# #         plt.figure(); plt.plot(timestamp/3600,ghi); plt.show()
        np.savez(outPath+'GHI_'+str(sensor), timestamp=timestamp,ghi=ghi);

