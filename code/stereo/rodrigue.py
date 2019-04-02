####Rodrigue rotation 

import numpy as np

azm=0.0; beta=0.1
k=np.array((-np.sin(azm),-np.cos(azm),0))

a=np.array((1,1,0))
# a=(np.sqrt(2)/2,np.sqrt(2)/2,0)

b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a) + k*a*(1-np.cos(beta))*k

theta=np.arctan(np.sqrt(b[0]**2+b[1]**2)/b[2])
phi=np.arctan(b[1]/b[0])
print(theta,phi)