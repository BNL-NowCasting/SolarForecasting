import pandas as pd
import numpy as np
import stat_tools as st
import matplotlib.pyplot as plt
import datetime as dt
import os, glob

GHI_path2 = '~/ldata/GHI2/'
GHI_path = '~/ldata/GHI/'
sensors = np.arange(1,11)

for sensor in sensors:
    with np.load(GHI_path2+'GHI_'+str(sensor)+'.npz') as data:
        ty, y = data['timestamp'], data['ghi']
        ty += 4*3600;        
        mk = ty>=(dt.datetime(2018,9,29)-dt.datetime(2018,1,1)).total_seconds()
        ty[mk] += 3600
    try:
        if len(ty)>1:
            np.savez(GHI_path+'GHI_'+str(sensor), timestamp=ty,ghi=y);       
    except:
        pass
            
