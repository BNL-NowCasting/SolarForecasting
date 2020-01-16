import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as md
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize, scale
import sklearn.metrics as metrics
import pickle
import stat_tools as st
import configparser
import os, subprocess
from datetime import datetime, timezone, timedelta
from ast import literal_eval as le
import pytz

try:
    try:
        config_path = sys.argv[1]
    except Exception:
        config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)

    inpath=le(cp["paths"]["feature_path"])
    GHI_path=le(cp["paths"]["GHI_path"])
    forecast_path=le(cp["paths"]["forecast_path"])

    lead_minutes=le(cp["forecast"]["lead_minutes"])

    #lead_minutes=[1,3,5,10,15,30,45]; 
    #sensors = np.arange(99,100)
    sensors = le(cp["forecast"]["sensors"])
except KeyError as e:
    print("Error loading config: %s" % e)
      
if not os.path.isdir(forecast_path):
    try:
        os.mkdir(forecast_path[:-1]) 
    except:
        print('Cannot create directory,', forecast_path[:-1])
        
if not os.path.isdir(forecast_path + "plots"):
    try:
        os.mkdir(forecast_path + "plots")
    except:
        print('Cannot create directory,', forecast_path + "plots")

        
plt.ioff()  #Turn off interactive plotting for running automatically

MAE, MSE = [], []
MAE2, MSE2 = [], []
for forward in lead_minutes:
    timestamp, DataX, DataY = [], [], []
    for sensor in sensors:
        x = np.genfromtxt(inpath+'GHI'+str(sensor)+'.csv',delimiter=',');
        x = x[x[:,0]==forward];
        with np.load(GHI_path+'GHI_'+str(sensor)+'.npz') as data:
            ty, y = data['timestamp'], data['ghi']
        x = x[x[:,1]<=ty[-1]]
        tx=x[:,1].copy(); 
        itx = ((tx-ty[0]+30)//60).astype(int)
    #     x[:,1] = 0.5*(y[itx]+y[itx+1])
        x[:,1] = (y[itx])
        DataX += [x[:,1:]]
    #     DataY += [0.5*(y[itx + forward]+y[itx+1+forward])]
        DataY += [(y[itx + forward])]
        timestamp += [tx];
        #sensor_list += [sensor]
        
    # DataX_all = DataX
    # DataY_all = DataY
    # timestamp_all = timestamp
    # print("\t",len(DataX),len(DataY),len(timestamp),len(DataX[0]))
    
    # for sensor in range(len(DataX_all)):
        # DataX = DataX_all[sensor]
        # DataY = DataY_all[sensor]
        # timestamp = timestamp_all[sensor]
        #try:
    DataX = np.vstack(DataX)
    DataY = np.hstack(DataY)
    timestamp = np.hstack(timestamp)
    #sensor = numpy.hstack(sensor_list)
    
    #print(timestamp, DataX, DataY)

    mk = (DataY > 0) & (DataX[:,0] > 0)
    DataX = DataX[mk]
    DataX[:,0]/=400;
    DataX[:,1:] = scale(DataX[:,1:]);  
    # DataX[:,1:] = normalize(DataX[:,1:],axis=0);  
    DataY = DataY[mk]
    timestamp = timestamp[mk]
    print("%i minute forecast, location %i" % (forward, sensor))
    print("\t",DataX.shape,DataY.shape)
    print('\tMean GHI:', np.nanmean(DataY))

    with open('optimal_model{:02d}.mod99'.format(forward),'rb') as fmod:
#     with open('optimal_model{:02d}.md'.format(forward),'rb') as fmod:
        SVR_linear = pickle.load(fmod)

    testY_hat = SVR_linear.predict(DataX)
#     testY_hat = SVR_linear.predict(DataX[:,1:])*DataX[:,0]*400
    testY_per = DataX[:,0]*400
    #print(testY_hat)
    # print(DataY)
    

    ts_offset = datetime(2018,1,1) #fix offset
    ts_offset.replace(tzinfo=timezone.utc)
    ts_fixed = timestamp+ts_offset.timestamp()
    txt_timestamp = np.asarray(ts_fixed, dtype='datetime64[s]')
    md_timestamp = md.epoch2num(ts_fixed)
    
    np.savetxt(forecast_path + "forecast_" + str(sensor) + "_" + str(forward) + "min.csv", np.column_stack((ts_fixed, DataX)), header="Timestamp,RawForecast",delimiter=",")
    np.savetxt(forecast_path + "ML_forecast_" + str(sensor) + "_" + str(forward) + "min.csv", np.column_stack((ts_fixed, DataY, testY_hat)), header="Timestamp,Actual_GHI,Forecast_GHI",delimiter=",")
    
    
    xfmt = md.DateFormatter('%H:%M', tz=pytz.timezone('US/Eastern'))
    plt.figure();
    ax=plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    #xlocator = md.MinuteLocator(byminute=[0], interval = 1)
    #ax.xaxis.set_major_locator(xlocator)
    plt.title("Actual vs. Forecast Irradiance",y=1.08, fontsize=16)
    plt.xlabel('Hour (EST)');
    plt.ylabel('Irradiance (W/m^2)'); 
    plt.plot(md_timestamp,DataY, label='Actual GHI');
    plt.plot(md_timestamp,testY_hat, label='Cloud-Tracking Forecast');
    plt.plot(md_timestamp,testY_per, label='Persistence Forecast');
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, borderaxespad=0., fontsize="small");
    plt.tight_layout()
    plt.savefig(forecast_path + "plots/GHI_ActualvsForecast_"+ str(sensor) + "_" + str(forward) + ".png")
    plt.close()
    #plt.show();
    #plt.figure(); plt.plot(testY_hat); plt.plot(DataY); plt.show();
    
    # bins=np.unique(timestamp)
    # Y = st.bin_average(DataY,timestamp,bins);
    # Yhat = st.bin_average(testY_hat,timestamp,bins);
    # plt.figure(); plt.plot(Yhat); plt.plot(Y); plt.show();
    MAE += [metrics.mean_absolute_error(DataY, testY_hat)]
    MSE += [metrics.mean_squared_error(DataY, testY_hat)]
    MAE2 += [metrics.mean_absolute_error(DataY, testY_per)]
    MSE2 += [metrics.mean_squared_error(DataY, testY_per)]
    # MAE = metrics.mean_absolute_error(Y, Yhat)
    # MSE = metrics.mean_squared_error(Y, Yhat)
            
    #     print("##################################")  
    #except Exception as e:
    #    raise
    #    print("Exception: %s" % str(e))

print('MAE and MAE2:', MAE, MAE2)

plt.figure();
plt.title("Cloud-Tracking vs. Persistent Model MAE",y=1.08, fontsize=16)
plt.plot(lead_minutes,MAE,label='Cloud-Tracking Forecast'); 
plt.plot(lead_minutes,MAE2,label='Persistent Model'); 
plt.xlabel('Forecast Lead Time (Minutes)');
plt.ylabel('Mean Absolute Error (W/m^2)'); 
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, borderaxespad=0., fontsize="small");
plt.tight_layout()
plt.savefig(forecast_path + "plots/MAE_CloudvsPersist.png")
plt.close()
#plt.show();
