import numpy as np
import glob
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
from scipy.optimize import minimize,leastsq

f1 = glob.glob('D:/tmp/sun_photo/*04b.jpg')[0];
f2 = glob.glob('D:/tmp/sun_photo/*04a.jpg')[0];

frame1=plt.imread(f1)[10:-10,10:-10,:].astype('float32'); 
frame2=plt.imread(f2)[10:-10,10:-10,:].astype('float32');  
#fig,ax=plt.subplots(2,1,sharex=True, sharey=True); ax[0].imshow(frame1[:,200:-200,0]); 
#ax[1].imshow(frame2[:,200:-200,0]); 

flag=np.all((frame1[:,:,0]>5,frame1[:,:,0]<245,frame2[:,:,0]>5,frame2[:,:,0]<245),axis=0);
plt.figure(); cnt,x,y,fig=plt.hist2d(frame2[flag,0].ravel(),frame1[flag,0].ravel(),bins=255,norm=LogNorm());
#cnt,x,y=np.histogram2d(frame2[flag,0].ravel(),frame1[flag,0].ravel(),bins=255);
    
foo=y[np.argmax(cnt,1)];  
zind=np.nonzero(np.diff(foo)<=-9); zind=zind[0]; zind+=1; foo[zind]=0.5*(foo[zind+1]+foo[zind-1]);
if sum(foo>238)>=1:
    cut=np.min(np.nonzero(foo>238)); 
else:
    cut=foo.shape[0];
plt.plot(x[:cut],foo[:cut]); plt.title('Scatter plot of image RED intensity for 1/60 and 1/30 exposures');
M=x[:cut]/255; MP=foo[:cut]/255;  

#def cost(x):
#    return np.sum((1+x[0]*M+x[1]*M**2+x[2]*M**3- \
#           a*(1+x[0]*MP+x[1]*MP**2+x[2]*MP**3))**2);
#res = minimize(cost, [0.1,0.01,0.001], method='nelder-mead', \
#                options={'xtol': 1e-8, 'disp': True});
#ss=res.x;

def resid(x,a=1,M=1,MP=1):
    p=np.poly1d(list(x[::-1])+[1-sum(x)]);
    return p(M)- a*p(MP);           
#for a in [1,1.5,1.7,2,2.2,2.5,3.0,4.0,5.0,8.0]:  
for a in [0.5]: 
    a=1.0/a;
    ss=leastsq(resid,[0.1]*6,args=(a,M,MP))[0];
    print(a,ss,np.linalg.norm(resid(ss,a,M,MP)));              
    plt.figure(); plt.plot(M,np.poly1d(list(ss[::-1])+[1-sum(ss)])(M));
    plt.xlabel("RED intensity"); plt.ylabel("Normalized Irradiance"); plt.title('Camera Response Function');