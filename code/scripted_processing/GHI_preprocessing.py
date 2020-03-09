import pandas as pd
import numpy as np
import stat_tools as st
import matplotlib.pyplot as plt
import datetime as dt
import os, glob, multiprocessing
import configparser
from ast import literal_eval as le
import pytz
import subprocess

#Sensors = np.arange(1,26);
#DirPath = '/home/amcmahon/data/originalGHI/'
#outPath = '/home/amcmahon/data/GHI/'


def GHI_pre_MP(args):
    sensor, yrmonth = args
#for sensor in Sensors:
    #print(DirPath + yrmonth[:4] + '-' + yrmonth[4:6] + '/' + yrmonth[:4] + '-' + yrmonth[4:6] + '_bps_' + str(sensor) + '_second.dat')
    flist = glob.glob(DirPath + yrmonth[:4] + '-' + yrmonth[4:6] + '/' + yrmonth[:4] + '-' + yrmonth[4:6] + '_bps_' + str(sensor) + '_second.dat')
    #flist = glob.glob(DirPath + 'BPS_' + str(sensor) + '_Second.dat')
    for fn in flist:
        print("Processing GHI file: %s" % fn)
        df = pd.read_csv(fn, sep = ',', header = 0, skiprows = [0, 2, 3], usecols = ['TIMESTAMP', 'SP2A_H'], converters={'TIMESTAMP':str2seconds})
        df = df.values;
        df[df[:,1]<=5, 1] = np.nan

        print(df[0,0], df[-1,0])
        
        #bins = np.arange(df[0,0], df[-1,0], dt.timedelta(minutes=1))
        bins = np.arange(df[0,0], df[-1,0], 60)
        #timestamp = 0.5 * (bins[1:] + bins[:-1])
        timestamp = bins[1:] + (bins[:-1] - bins[1:])/2
        ghi = st.bin_average_reg(df[:,1], df[:,0], bins);

        #plt.figure(); plt.plot(timestamp/3600,ghi); plt.show()
        np.savez(outPath+yrmonth+'/'+'GHI_'+str(sensor), timestamp=timestamp,ghi=ghi);


def str2seconds(x):
    #Note: local_tz must be defined globally
    x = dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    x_local = local_tz.localize(x, is_dst=None)
    x_utc = x_local.astimezone(pytz.utc)
    return x_utc.timestamp()


if __name__ == "__main__":

    try:
        config_path = sys.argv[1]
    except Exception:
        config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)

    outPath=le(cp["paths"]["GHI_path"])
    DirPath=le(cp["paths"]["raw_GHI_path"])
    Sensors=le(cp["forecast"]["sensors"])
    days=le(cp["forecast"]["days"])
    yrmonths=set([d[:6] for d in days])
    
    try:
        local_tz=pytz.timezone(cp["GHI_sensors"]["GHI_timezone"])
        print("Using timezone: %s" % str(local_tz))
    except Exception:
        local_tz=pytz.timezone("utc")    
        print("Error processsing timezone config, assuming UTC")

    try:
        cores_to_use = int(cp["server"]["cores_to_use"])
    except Exception:
        cores_to_use = 20
    
    if not os.path.isdir(outPath):
        os.makedirs(outPath)
        os.chmod(outPath,0o755) 
    
    p = multiprocessing.Pool(cores_to_use)      
    for yrmonth in yrmonths:
        print("Preprocessing %s\n" % yrmonth)
        
        if not os.path.isdir(outPath+yrmonth+'/'):
            try:
                subprocess.call(['mkdir', outPath+yrmonth+'/']) 
                os.chmod(outPath+yrmonth+'/',0o755) 
            except:
                print('Cannot create directory,',outPath+yrmonth+'/')
                continue

    #str2seconds = lambda x: (dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')-dt.datetime(2018,1,1)).total_seconds()+3600*5
    #str2seconds = lambda x: (dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(tzinfo = dt.timezone(-dt.timedelta(hours=5)))-dt.datetime(2018,1,1).replace(tzinfo = dt.timezone.utc))

             
        p.map(GHI_pre_MP,[(sensor,yrmonth) for sensor in Sensors])  
        
    p.close()
    p.join()
