import numpy as np
import glob
from matplotlib import pyplot as plt
import stat_tools as st
from datetime import datetime,timedelta
import ephem
from scipy.ndimage import rotate
plt.rcParams['image.cmap']='jet'

deg2rad=np.pi/180
camera='HD815_1'
# camera='HD490'
# camera='HD17'

####set up paths, constants, and initial parameters
inpath='d:/data/HD20_calib/'
inpath='d:/data/images/'+camera+'/'

lat,lon=40.88,-72.87


bnds=[[None,None]]*9;
if camera=='HD815_1': 
    bnds[0]=[2821,2821]; bnds[1]=[1430,1446]; bnds[2]=[1421,1441]  ####815_1
elif camera=='HD815_2': 
    bnds[0]=[2821,2821]; bnds[1]=[1423,1429]; bnds[2]=[1450,1460]  ####815_2
elif camera=='HD490':
    bnds[0]=[2843,2843]; bnds[1]=[1471,1481]; bnds[2]=[1475,1485]  ####490


nx0=ny0=2810
nr0=(nx0+ny0)/4.0        ##### radius of the valid image   
rotation=0 
# cy,cx=1481,1481  ####490
cy,cx=1428,1457  ###815_2
c1=c2=2

ts=np.load(camera+'.npy').item()
    
gatech = ephem.Observer(); 
gatech.lat, gatech.lon = '40.88', '-72.87'
moon=ephem.Moon() 
    
def cost_sun_match(params):
    cost=0;
    nx0,cy,cx,rotation,beta,azm = params
    ny0=nx0; 
    
    for tstr in ts:
        gatech.date = datetime.strptime(tstr,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
        moon.compute(gatech) 
        sz=np.pi/2-moon.alt; saz=(rotation+moon.az-np.pi)%(2*np.pi);     
        
        k=np.array((-np.sin(azm),-np.cos(azm),0))        
        a=np.array((np.sin(sz)*np.cos(saz),np.sin(sz)*np.sin(saz),np.cos(sz)))
        b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a) + k*a*(1-np.cos(beta))*k
        sz=np.arctan(np.sqrt(b[0]**2+b[1]**2)/b[2])
        saz=np.arctan2(b[1],b[0])%(2*np.pi) 
        
        rref=np.sin(sz/2)*np.sqrt(2)/2
        xref,yref=cx+nx0*rref*np.sin(saz)+0.5,cy+ny0*rref*np.cos(saz)+0.5
        yobs,xobs=ts[tstr];
        cost += (xref-xobs)**2+(yref-yobs)**2
#         cost += np.abs(xref-xobs)+np.abs(yref-yobs)
#         cost += ((xref-xobs)**2+(yref-yobs)**2)/np.cos(sz)**2
#     print(nx0,dx,dy,rotation/deg2rad,beta,azm,cost)
    return cost

def cost_sun_match2(params):
    cost=0;
    nx0,cy,cx,rotation,beta,azm,c1,c2,c3 = params
    ny0=nx0
#     cy=1423; cx=1454
    for tstr in ts:
        gatech.date = datetime.strptime(tstr,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
        moon.compute(gatech) 
        sz=np.pi/2-moon.alt; saz=(rotation+moon.az-np.pi)%(2*np.pi);     
        
        k=np.array((-np.sin(azm),-np.cos(azm),0))        
        a=np.array((np.sin(sz)*np.cos(saz),np.sin(sz)*np.sin(saz),np.cos(sz)))
        b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a) + k*a*(1-np.cos(beta))*k
        sz=np.arctan(np.sqrt(b[0]**2+b[1]**2)/b[2])
        saz=np.arctan2(b[1],b[0])%(2*np.pi) 
        
#         rref=np.sin(sz/c1)*np.sqrt(c2)/2
        rref=c1*sz+c2*sz**3+c3*sz**5
        xref,yref=cx+nx0*rref*np.sin(saz)+0.5,cy+ny0*rref*np.cos(saz)+0.5
        yobs,xobs=ts[tstr];
        cost += (xref-xobs)**2+(yref-yobs)**2
#         cost += ((xref-xobs)**2+(yref-yobs)**2)/np.cos(sz)**2
#     print(nx0,dx,dy,rotation/deg2rad,beta,azm,cost)
    return cost

from scipy.optimize import fmin,fmin_l_bfgs_b
if __name__ == "__main__":
#     guess= [2826,1459,1459,0*deg2rad,0,0]    
#     xopt = fmin(cost_sun_match, guess, maxiter=2000,xtol=1e-6) 
    guess= [2810,1439,1439,0*deg2rad,0,0,2.2,2.36,0]
#     xopt = fmin(cost_sun_match2, guess, maxiter=4000,xtol=1e-6) 
    xopt = fmin_l_bfgs_b(cost_sun_match2, guess, bounds=bnds,approx_grad=1)[0] 
    print(xopt)
    res=xopt.copy(); cost_min=1e10
    for i in range(2):
#         guess=np.array(xopt)+np.array([10,8,8,0.03,0.03,0.1])*(np.random.random(6)-0.5)        
#         xopt = fmin(cost_sun_match, guess, maxiter=2000,xtol=1e-6)
        guess=np.array(xopt)+np.array([10,8,8,0.03,0.03,0.1,0.05,0,0])*(np.random.random(9)-0.5)
#         xopt = fmin(cost_sun_match2, guess,maxiter=2000,xtol=1e-6)
        xopt = fmin_l_bfgs_b(cost_sun_match2, guess, bounds=bnds,approx_grad=1)[0] 
        cost=cost_sun_match2(xopt)
        if cost<=cost_min: 
            cost_min=cost
            res= xopt 
        print(xopt)     
    print("Final Cost:", cost_sun_match2(res))
    print("Final solution: [", ",".join("{:.4f}".format(s) for s in res),']');

#     nx0,cy,cx,rotation,beta,azm=res;   ny0=nx0; 
    nx0,cy,cx,rotation,beta,azm,c1,c2,c3=res;  ny0=nx0
    xref,yref=np.zeros((2,len(ts)));
    for its,tstr in enumerate(ts):
        gatech.date = datetime.strptime(tstr,'%Y%m%d%H%M%S').strftime('%Y/%m/%d %H:%M:%S')
        moon.compute(gatech) 
        sz=np.pi/2-moon.alt; saz=(rotation+moon.az-np.pi)%(2*np.pi);        
        
        k=np.array((-np.sin(azm),-np.cos(azm),0))        
        a=np.array((np.sin(sz)*np.cos(saz),np.sin(sz)*np.sin(saz),np.cos(sz)))
        b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a) + k*a*(1-np.cos(beta))*k

        sz=np.arctan(np.sqrt(b[0]**2+b[1]**2)/b[2])
        saz=np.arctan2(b[1],b[0])%(2*np.pi)        
        
#         rref=np.sin(sz/c1)*np.sqrt(c2)/2
        rref=c1*sz+c2*sz**3+c3*sz**5
        xref[its],yref[its]=cx+nx0*rref*np.sin(saz),cy+ny0*rref*np.cos(saz)
    yobs,xobs=zip(*[ts[tstr] for tstr in ts]);
    plt.figure();
    plt.scatter(xref,yref,color=['blue'])  
    plt.scatter(xobs,yobs,color=['green'])       
#     plt.show() 