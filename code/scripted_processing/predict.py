import numpy as np
import matplotlib
matplotlib.use('agg')
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
    days=le(cp["forecast"]["days"])

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

        
plt.ioff()  #Turn off interactive plotting for running automatically



for day in days:
    MAE, MSE = [], []
    MAE2, MSE2 = [], []

    print("Predicting for " + day)
    
    if not os.path.isdir(forecast_path+day[:8]):
        try:
            subprocess.call(['mkdir', forecast_path+day[:8]]) 
        except:
            print('Cannot create directory,',forecast_path+day[:8])
            continue
    if not os.path.isdir(forecast_path+day[:8] + "/plots"):
        try:
            os.mkdir(forecast_path+day[:8] + "/plots")
        except:
            print('Cannot create directory,', forecast_path+day[:8] + "/plots")
                
    for forward in lead_minutes:
        timestamp, DataX, DataY = {},{},{}
        MAE_period, MSE_period = [], []
        MAE2_period, MSE2_period = [], []
        for sensor in sensors:
            timestamp[sensor] = []
            DataX[sensor] = []
            DataY[sensor] = []
            
            try:
                x = np.genfromtxt(inpath+day[:8]+'/GHI'+str(sensor)+'.csv',delimiter=',');  # < ORIGINAL
                #x = np.genfromtxt(inpath+'/GHI'+str(sensor)+'.csv',delimiter=',');         # Temp change to allow running of old data in dhuang3
                
                x = x[x[:,0]==forward];                                 #Take all rows where forecast period == forward
                
                if sensor == 26:                                                # Temp added for 2018-09-22 test with location 99
                    with np.load(GHI_path+'GHI_'+str(99)+'.npz') as data:       #
                        ty, y = data['timestamp'], data['ghi']                  #
                else:                                                           #
                    with np.load(GHI_path+'GHI_'+str(sensor)+'.npz') as data:   # < ORIGINAL
                        ty, y = data['timestamp'], data['ghi']                  # < ORIGINAL
                        ty -= 3600 #Add an hour (testing only!)
                
                x = x[x[:,1]<=ty[-1]]                                   #Take all "feature" elements where timestamp is less than last GHI timestamp
                tx=x[:,1].copy();                                       #Create copy of feature timestamps
                itx = ((tx-ty[0]+30)//60).astype(int)                   #Create array of relative time based on first GHI timestamp, add 30 secs, floor to minutes, convert to int
                print("len(x): %i\tlen y: %i\n" % (len(tx), len(ty)))
                try:
                    print("tx: %i\ty: %i\titx: %i\n" % (tx[0],ty[0],itx[0]))
                except IndexError:
                    pass
                x[:,1] = (y[itx])                                       #Select x values corresponding to times in itx
                DataX[sensor] += [x[:,1:]]                              #Append timestamp and x values to DataX (does NOT copy forecast period "forward" column)
                DataY[sensor] += [(y[itx + forward])]                   #Get future actual GHI
                timestamp[sensor] += [tx];


                DataX[sensor] = np.vstack(DataX[sensor])                #stack time series for all GHI locations vertically
                DataY[sensor] = np.hstack(DataY[sensor])                #stack time series for persistence horizontally
                timestamp[sensor] = np.hstack(timestamp[sensor])        #stack timestamps horizontally

            #print(DataY[sensor], DataX[sensor][:,0])
            #try:
                mk = (DataY[sensor] > 0) & (DataX[sensor][:,0] > 0)     #create boolean list where persistence value and timestamp are both >0
                DataX[sensor] = DataX[sensor][mk]                       #take subset selected above
                DataX[sensor][:,0]/=400;                                #scale GHI by 400?  (note: data in *.npz is already scaled?)
                DataX[sensor][:,1:] = scale(DataX[sensor][:,1:]);       #normalize other x values
            except ValueError as e:
                print("Skipping sensor %i, %s" % (sensor, str(e)))
                continue                                                #This will get thrown if there's no GHI data and DataY is filled with NaNs
            # DataX[:,1:] = normalize(DataX[:,1:],axis=0);  
            DataY[sensor] = DataY[sensor][mk]                       #take subset to match x values
            timestamp[sensor] = timestamp[sensor][mk]               #take subset to match x values
            print("%i minute forecast, location %i" % (forward, sensor))
            print("\t",DataX[sensor].shape,DataY[sensor].shape)
            print('\tMean GHI:', np.nanmean(DataY[sensor]))

            with open('optimal_model{:02d}.mod99'.format(forward),'rb') as fmod:
                SVR_linear = pickle.load(fmod)                      #Load model

            testY_hat = SVR_linear.predict(DataX[sensor])           #Run model
            testY_per = DataX[sensor][:,0]*400                      #Create persistence model (and rescale)


            ts_offset = datetime(2018,1,1)                          #fix offset
            #ts_offset.replace(tzinfo=timezone.utc)
            #ts_fixed = (timestamp[sensor]+ts_offset.timestamp()-dt.timedelta(hours=5)
            ts_fixed = timestamp[sensor] + (ts_offset.timestamp()-(3600*5))
            #ts_fixed = (datetime.strptime(timestamp[sensor], '%Y-%m-%d %H:%M:%S')-datetime(2018,1,1)).total_seconds()+(3600*5)
            txt_timestamp = np.asarray(ts_fixed, dtype='datetime64[s]')
            md_timestamp = md.epoch2num(ts_fixed)
            #ts_str = ts_fixed.astype(object)
            #ts_str = [datetime.fromtimestamp(ts).strftime("%m/%d/%Y %H:%M:%S") for ts in ts_str]
            #print(ts_str)
            
            #np.savetxt(forecast_path + "forecast_" + str(sensor) + "_" + str(forward) + "min.csv", np.column_stack((ts_fixed, DataX[sensor])), header="Timestamp,DateTime String, RawForecast",delimiter=",")
            np.savetxt(forecast_path + day[:8] + "/ML_forecast_" + str(sensor) + "_" + str(forward) + "min.csv", np.column_stack((ts_fixed, DataY[sensor], testY_hat, testY_per)), header="Timestamp,Actual_GHI,Forecast_GHI,Persistence_GHI",delimiter=",")
            
            xfmt = md.DateFormatter('%H:%M', tz=timezone.utc ) #pytz.timezone('US/Eastern'))
            plt.figure();
            ax=plt.gca()
            ax.xaxis.set_major_formatter(xfmt)
            #xlocator = md.MinuteLocator(byminute=[0], interval = 1)
            #ax.xaxis.set_major_locator(xlocator)
            plt.title("Actual vs. Forecast Irradiance "+day,y=1.08, fontsize=16)
            plt.xlabel('Hour (UTC)');
            plt.ylabel('Irradiance (W/m^2)');
            plt.plot(md_timestamp,DataY[sensor], "-", linewidth=1, label='Actual GHI');
            plt.plot(md_timestamp,testY_hat, "-", linewidth=1, label='Cloud-Tracking Forecast');
            plt.plot(md_timestamp,testY_per, "-", linewidth=1, label='Persistence Forecast');
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, borderaxespad=0., fontsize="small");
            plt.tight_layout()
            plt.savefig(forecast_path + day[:8] + "/plots/GHI_ActualvsForecast_"+ str(sensor) + "_" + str(forward) + ".png")
            plt.close()
            #plt.show();
            #plt.figure(); plt.plot(testY_hat); plt.plot(DataY); plt.show();
            
            # bins=np.unique(timestamp)
            # Y = st.bin_average(DataY,timestamp,bins);
            # Yhat = st.bin_average(testY_hat,timestamp,bins);
            # plt.figure(); plt.plot(Yhat); plt.plot(Y); plt.show();
            MAE_period += [metrics.mean_absolute_error(DataY[sensor], testY_hat)]
            MSE_period += [metrics.mean_squared_error(DataY[sensor], testY_hat)]
            MAE2_period += [metrics.mean_absolute_error(DataY[sensor], testY_per)]
            MSE2_period += [metrics.mean_squared_error(DataY[sensor], testY_per)]
            # MAE = metrics.mean_absolute_error(Y, Yhat)
            # MSE = metrics.mean_squared_error(Y, Yhat)
                    
            #     print("##################################")  
            #except Exception as e:
            #    raise
            #    print("Exception: %s" % str(e))
            
        MAE += [sum(MAE_period)/len(MAE_period)]
        MSE += [sum(MSE_period)/len(MSE_period)]
        MAE2 += [sum(MAE2_period)/len(MAE2_period)]
        MSE2 += [sum(MSE2_period)/len(MSE2_period)]

    print('MAE and MAE2:', MAE, MAE2)

    plt.figure();
    plt.title("Cloud-Tracking vs. Persistent Model MAE "+day,y=1.08, fontsize=16)
    plt.plot(lead_minutes,MAE,label='Cloud-Tracking Forecast'); 
    plt.plot(lead_minutes,MAE2,label='Persistent Model'); 
    plt.xlabel('Forecast Lead Time (Minutes)');
    plt.ylabel('Mean Absolute Error (W/m^2)'); 
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, borderaxespad=0., fontsize="small");
    plt.tight_layout()
    plt.savefig(forecast_path + day[:8] + "/plots/MAE_CloudvsPersist.png")
    plt.close()
    #plt.show();
