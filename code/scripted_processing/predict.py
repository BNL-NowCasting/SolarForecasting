import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import normalize, scale
import sklearn.metrics as metrics
import pickle
import stat_tools as st

inpath = '~/ldata/training/'
# inpath = '~/ldata/feature/'
GHI_path = '~/ldata/GHI/'

lead_minutes=[1,3,5,10,15,30,45]; 

sensors = np.arange(99,100)
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
    DataX = np.vstack(DataX)
    DataY = np.hstack(DataY)
    timestamp = np.hstack(timestamp)

    mk = (DataY > 0) & (DataX[:,0] > 0)
    DataX = DataX[mk]
    DataX[:,0]/=400;
    DataX[:,1:] = scale(DataX[:,1:]);  
    # DataX[:,1:] = normalize(DataX[:,1:],axis=0);  
    DataY = DataY[mk]
    timestamp = timestamp[mk]
    # print(DataX.shape,DataY.shape)
#     print('Mean GHI:', np.nanmean(DataY))

    with open('optimal_model{:02d}.mod99'.format(forward),'rb') as fmod:
#     with open('optimal_model{:02d}.md'.format(forward),'rb') as fmod:
        SVR_linear = pickle.load(fmod)

    testY_hat = SVR_linear.predict(DataX)
#     testY_hat = SVR_linear.predict(DataX[:,1:])*DataX[:,0]*400
    testY_per = DataX[:,0]*400
    # print(testY_hat)
    # print(DataY)
    # plt.figure(); plt.plot(timestamp,testY_hat); plt.plot(timestamp,DataY); plt.show();
#     plt.figure(); plt.plot(testY_hat); plt.plot(DataY); plt.show();
    
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

print('MAE and MAE2:', MAE, MAE2)

plt.figure(); plt.plot(lead_minutes,MAE,label='Cloud-tracking-based forecast');  plt.plot(lead_minutes,MAE2,label='Persistent model'); 
plt.xlabel('Forecast lead time, minutes',fontsize=18); plt.ylabel('Mean absolute error, W/m^2',fontsize=18); 
plt.legend(loc='center_right'); plt.show();
