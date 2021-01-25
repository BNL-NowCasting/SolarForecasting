import pandas as pd
import numpy as np
import stat_tools as st
import datetime as dt
import os, glob, multiprocessing
import configparser
from ast import literal_eval
import pytz
import subprocess
#import utils

#############################################################################

def str2seconds(x):
    # Note: local_tz must be defined globally
    x = dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    x_local = local_tz.localize(x, is_dst=None)
    x_utc = x_local.astimezone(pytz.utc)
    return x_utc.timestamp()

#############################################################################

def GHI_pre_MP(args):

    sensor, yrmonth = args

    flist = glob.glob(DirPath + yrmonth[:4] + '-' + yrmonth[4:6] + '/' + yrmonth[:4] + '-' + yrmonth[4:6] + '_bps_' + str(sensor) + '_second.dat')

    for fn in flist:

        print("Processing GHI file: %s" % fn)
        df = pd.read_csv(fn, sep=',', header=0, skiprows=[0, 2, 3], usecols=['TIMESTAMP', 'SP2A_H'], converters={'TIMESTAMP': str2seconds})
        df = df.values;
        df[df[:,1] <= 5, 1] = np.nan

        bins = np.arange(df[0,0], df[-1,0], 60)
        timestamp = bins[1:] + (bins[:-1] - bins[1:])/2
        ghi = st.bin_average_reg(df[:,1], df[:,0], bins);

        out_file = outPath+yrmonth+'/'+'GHI_'+format(sensor,'02') + '.npz'
        print("Creating file " + out_file)
        np.savez(out_file, timestamp=timestamp, ghi=ghi);

#############################################################################

if __name__ == "__main__":

    print("/// GHI preprocessing ///")

    try:
        config_path = sys.argv[1]
    except Exception:
        config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)

    outPath = literal_eval(cp["paths"]["GHI_path"])
    DirPath = literal_eval(cp["paths"]["raw_GHI_path"])
    Sensors = literal_eval(cp["forecast"]["sensors"])
    days = literal_eval(cp["forecast"]["days"])
    yrmonths = set([d[:6] for d in days])
    
    try:
        local_tz=pytz.timezone(cp["GHI_sensors"]["GHI_timezone"])
        print("Using timezone: %s" % str(local_tz))
    except Exception:
        local_tz = pytz.timezone("utc")
        print("Error processsing timezone config, assuming UTC")

    try:
        cores_to_use = int(cp["server"]["cores_to_use"])
    except Exception:
        cores_to_use = 20
    
    if not os.path.isdir(outPath):
        print("Creating directory " + outPath)
        os.makedirs(outPath)
        #os.chmod(outPath, 0o755)

    if cores_to_use > 1:
        pool = multiprocessing.Pool(cores_to_use, maxtasksperchild=128)

    for yrmonth in yrmonths:
        print("Preprocessing %s\n" % yrmonth)

        dir = outPath + yrmonth + '/'

        if not os.path.isdir(dir):
            try:
                print("Creating directory " + dir)
                os.mkdir(dir)
                #os.chmod(dir, 0o755)
            except:
                print('Cannot create directory,', dir)
                continue

        args = [(sensor, yrmonth) for sensor in Sensors]

        if cores_to_use > 1:
            pool.map(GHI_pre_MP, args)
        else:
            for arg in args:
                GHI_pre_MP(arg)

    if cores_to_use > 1:
        pool.close()
        pool.join()
