import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize, scale
import sklearn.metrics as metrics
import pickle
import stat_tools as st
import datetime as dt

# inpath = '~/ldata/training2/'
inpath = '~/ldata/feature/'
GHI_path = '~/ldata/GHI/'

forward = 10

sensors = np.arange(99,100)
timestamp, DataX, DataY = [], [], []
for sensor in sensors:
    x = np.genfromtxt(inpath+'GHI'+str(sensor)+'.csv',delimiter=',');
    x = x[x[:,0]==forward];
    tx=x[:,1].copy(); 
    with np.load(GHI_path+'GHI_'+str(sensor)+'.npz') as data:
        ty, y = data['timestamp'], data['ghi']
    itx = ((tx-ty[0]+30)//60).astype(int)
#     x[:,1] = 0.5*(y[itx]+y[itx+1])
    x[:,1] = (y[itx])
    DataX += [x[:,1:]]
#     DataY += [0.5*(y[itx + forward]+y[itx+1+forward])]
    DataY += [(y[itx + forward])]
    timestamp += [tx];
DataX = np.vstack(DataX)
DataY = np.hstack(DataY)
timestamp = np.hstack(timestamp)

# mk = (DataY > 0) & (DataX[:,0] > 0)
t0=(dt.datetime(2018,9,22)-dt.datetime(2018,1,1)).total_seconds()
mk = (DataY > 0) & (DataX[:,0] > 0) & (timestamp>t0) & (timestamp<t0+86400) 
DataX = DataX[mk]
DataX[:,0]/=400;
DataX[:,1:] = scale(DataX[:,1:]);  
# DataX[:,1:] = normalize(DataX[:,1:],axis=0);  
DataY = DataY[mk]
timestamp = timestamp[mk]
# print(DataX.shape,DataY.shape)
#     print('Mean GHI:', np.nanmean(DataY))

with open('optimal_model{:02d}.m99'.format(forward),'rb') as fmod:
    SVR_linear = pickle.load(fmod)

testY_hat = SVR_linear.predict(DataX); testY_hat[testY_hat<5]=5
# testY_hat = SVR_linear.predict(DataX[:,1:])*DataX[:,0]*400
baseY = DataX[:,0]*400
# plt.figure(); plt.plot(timestamp,testY_hat); plt.plot(timestamp,DataY); plt.show();
plt.figure(); plt.plot(DataY,label='Measured');  plt.plot(testY_hat,label='Cloud-tracking-based forecast');  plt.plot(baseY,label='Persistent model'); 
# plt.xticks(np.arange(0,timestamp.size,250), ['{:2.1f}'.format(t/86400-243.3) for t in timestamp[::250]])
plt.xticks(np.arange(0,timestamp.size,50), ['{:2.1f}'.format((t%86400/3600)) for t in timestamp[::50]])
plt.xlabel('UTC time, September 22, 2018',fontsize=16); plt.ylabel('GHI, W/m^2',fontsize=16);
plt.title('Forecast lead time: '+str(forward)+' minutes'); plt.legend(); plt.show();

# bins=np.unique(timestamp)
# Y = st.bin_average(DataY,timestamp,bins);
# Yhat = st.bin_average(testY_hat,timestamp,bins);
# plt.figure(); plt.plot(Yhat); plt.plot(Y); plt.show();
MAE = [metrics.mean_absolute_error(DataY, testY_hat)]
MSE = [metrics.mean_squared_error(DataY, testY_hat)]
# MAE = metrics.mean_absolute_error(Y, Yhat)
# MSE = metrics.mean_squared_error(Y, Yhat)
        
#     print("##################################")  

#     print('Lead time:', forward, '  MAE and MSE:', MAE, MSE)

