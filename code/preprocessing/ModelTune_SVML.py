import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import normalize,scale
import sklearn.metrics as metrics
from sklearn.model_selection import RepeatedKFold
from copy import deepcopy
import pickle
import datetime as dt

inpath = '~/ldata/training2/'
# inpath = '~/ldata/feature/'
GHI_path = '~/ldata/GHI/'

lead_minutes=[1,3,5,10,15,30,45]; 
# lead_minutes=[1]; 
# lead_minutes=[10,15,30,45]; 
sensors = np.arange(99,100)

for  forward in lead_minutes:

    timestamp, DataX, DataY = [], [], []
    for sensor in sensors:
        x = np.genfromtxt(inpath+'GHI'+str(sensor)+'.csv',delimiter=',');
        x = x[x[:,0]==forward];
        with np.load(GHI_path+'GHI_'+str(sensor)+'.npz') as data:
            ty, y = data['timestamp'], data['ghi']
        x = x[x[:,1]<=ty[-1]]
        tx=x[:,1].copy(); 
        itx = ((tx-ty[0]+30)//60).astype(int)
        x[:,1] = y[itx]
        DataX += [x[:,1:]]
#         DataY += [y[itx + forward]/x[:,1]]
        DataY += [y[itx + forward]]
        timestamp += [tx];

    DataX = np.vstack(DataX)
    DataY = np.hstack(DataY)
    timestamp = np.hstack(timestamp)

    mk = (DataY > 0) & (DataX[:,0] > 0) #& (timestamp>(dt.datetime(2018,9,5)-dt.datetime(2018,1,1)).total_seconds()) & (timestamp<(dt.datetime(2018,9,5)-dt.datetime(2018,1,1)).total_seconds()+86400) 
    DataX = DataX[mk]
    DataX[:,0]/=400;
    DataX[:,1:] = scale(DataX[:,1:]);  
#     DataX = scale(DataX);  
#     DataX[:,1:] = normalize(DataX[:,1:],axis=0);      
#     DataX = normalize(DataX,axis=0);  
    DataY = DataY[mk]
    print(DataX.shape,DataY.shape)
    print(np.mean(DataY))
#     plt.figure(); plt.plot(DataY/800); plt.plot(DataX[:,15]/2)
#     plt.show();

    # DataX=DataX[::5];
    # DataY=DataY[::5];

#     C = [1e2,1e3,1e1]
#     epsilon= [ 1e-2, 1e-1, 1e-3]
    C = [3e2,3e3,1e3]
#     C = [1e0,1e1]
    epsilon= [ 1e-1, 3e-1, 1e0]

    MAE_max = 1e8
    MSE_max = 1e8
    C_max1,C_max2,epsilon_ma1,epsilon_max2 = [0]*4
    for cc in C:
        
        MAE_1 = []
        MSE_1 = []
        
        for ep in epsilon:
            
            print('C:', cc, '  epsilon =', ep)
            
            MAE_list = []
            MSE_list = []
            
            skf = RepeatedKFold(n_splits=2, n_repeats = 2, random_state=1)        
            
            for train_index, test_index in skf.split(DataX, DataY):
                
                trainX, testX = DataX[train_index], DataX[test_index]
                trainY, testY = DataY[train_index], DataY[test_index]
                
#                 SVR_linear = SVR(C=cc, epsilon=ep, kernel='linear')
                SVR_linear = SVR(C=cc, epsilon=ep, kernel='rbf')
                SVR_linear.fit(trainX, trainY)
                testY_hat = SVR_linear.predict(testX)

                MAE = metrics.mean_absolute_error(testY, testY_hat)
                MSE = metrics.mean_squared_error(testY, testY_hat)

                MAE_list.append(MAE)
                MSE_list.append(MSE)

            MAE_all = sum(MAE_list)/len(MAE_list) 
            MSE_all = sum(MSE_list)/len(MSE_list) 
            print('MAE and MSE:', MAE_all, MSE_all,'\n')


            if (MAE_all < MAE_max):
                MAE_max = MAE_all
                C_max1 = cc
                epsilon_max1 = ep
                opt_model=deepcopy(SVR_linear);
                    
            if (MSE_all < MSE_max):
                MSE_max = MSE_all
                C_max2 = cc
                epsilon_max2 = ep

    with open('optimal_model{:02d}.md99'.format(forward),'wb') as fmod:
        pickle.dump(opt_model,fmod)

    print("#######")       
    print('C_max1:', C_max1)
    print('Eps_max1:', epsilon_max1)
        
    print('C_max2:', C_max2)
    print('Eps_max2:', epsilon_max2)
    print("#######")      
    #        MAE_1.append(sum(MAE_list)/len(MAE_list))
    #        MSE_1.append(sum(MSE_list)/len(MSE_list))
        
    print("##################################")  

    print('Min MAE and MSE:', MAE_max, MSE_max)
#    MAE_2.append(MAE_1)
#    MSE_2.append(MSE_1)
    
