import numpy as np
import os, glob
from matplotlib import pyplot as plt
import stat_tools as st
from PIL import Image

deg2rad=np.pi/180

camera='HD20'
day='20180310'



coordinate = {'HD815_1': [40.87203321,  -72.87348295],
              'HD815_2': [40.87189059,  -72.873687],
              'HD490'  : [40.865968816, -72.884647222], 
              'HD17'   : [40.8575056,   -72.8547344], 
              'HD19'   : [40.8580088,   -72.8575717], 
              'HD20'   : [40.85785,     -72.8597], 
              'HD01'   : [40.947353,    -72.899617],
              'HD02'   : [40.948044,    -72.898372],
              'HD03'   : [40.897122,    -72.879053],
              'HD04'   : [40.8975,      -72.877497],
              'HD05'   : [40.915708,    -72.892406],
              'HD06'   : [40.917275,    -72.891592]                       
              }
params = {'HD815_1':[2821.0000, 1442.8231, 1421.0000,  0.1700, -0.0135, -2.4368, 0.3465, -0.0026, -0.0038],
          'HD815_2':[2821.0000, 1424.0000, 1449.0000,  0.0310, -0.0114, -0.9816, 0.3462, -0.0038, -0.0030],
          'HD490'  :[2843.0000, 1472.9511, 1482.6685,  0.1616,  0.0210, -0.5859, 0.3465, -0.0043, -0.0030], 
          'HD17'   :[2830.0007, 1473.2675, 1459.7203, -0.0986, -0.0106, -1.2440, 0.3441, -0.0015, -0.0042], 
          'HD19'   :[2826.5389, 1461.0000, 1476.6598, -0.0097,  0.0030,  2.9563, 0.3415,  0.0004, -0.0044], 
          'HD20'   :[2812.7874, 1475.1453, 1415.0000,  0.1410, -0.0126,  0.4769, 0.3441,  0.0004, -0.0046], 
          'HD05'   :[2813.3741, 1435.1706, 1453.7087, -0.0119, -0.0857, -1.8675, 0.3499, -0.0033, -0.0027],
          'HD06'   :[2809.2813, 1446.4900, 1438.0777, -0.0237, -0.0120, -1.3384, 0.3479, -0.0024, -0.0037],
          'HD01'   :[2813.7462, 1472.2066, 1446.3682,  0.3196, -0.0200, -1.9636, 0.3444, -0.0008, -0.0042],
          'HD03'   :[2807.8902, 1436.1619, 1439.3879, -0.3942,  0.0527,  2.4658, 0.3334,  0.0129, -0.0085]}   
####set up paths, constantsand initial parameters
inpath = '~/data/images/'
inpath = inpath + camera + '/' 
outpath = '~/data/undistort_output/'

lat = coordinate[camera][0]
lon = coordinate[camera][1]

min_scatter_angle = 8
dark_threshold = 25      #### threshold of dark DN value (i.e., shadow band)
var_threshold = 4        #### threshold for cloud spatial variation
rbr_clear = -0.15     ### upper bound of clear sky red/blue index
rbr_cloud = -0.05     ### lower bound of cloud red/blue index
ndilate=19
####dimension of the valid portion of the original image, i.e., the disk with elevation_angle>0
####they need to be tuned for each camera           

nx,ny=2001,2001          #### size of the undistorted image 
max_theta=70*deg2rad     ##### maximum zenith angle used for processing
max_tan = np.tan(max_theta)

dest=outpath+camera
if not os.path.isdir(dest):
    os.makedirs(dest)
    os.chmod(dest,0o755)
dest=outpath+camera+'/'+day+'/'
if not os.path.isdir(dest):
    os.makedirs(dest)
    os.chmod(dest,0o755)

xbin,ybin=np.linspace(-max_tan,max_tan,nx), np.linspace(-max_tan,max_tan,ny)  
xgrid,ygrid=np.meshgrid(xbin,ybin)####(xgrid, ygrid) are the grids of the undistorted space
valid = xgrid**2+ygrid**2 <= max_tan**2   
invalid = xgrid**2+ygrid**2 > (max_tan-1e-2)**2 

nx0=ny0=params[camera][0]
nr0=(nx0+ny0)/4
xstart=int(params[camera][2]-nx0/2+0.5); ystart=int(params[camera][1]-ny0/2+0.5)
nx0=int(nx0+0.5); ny0=int(ny0+0.5)
#####compute the zenith and azimuth angles for each pixel
x0,y0=np.meshgrid(np.linspace(-nx0//2,nx0//2,nx0),np.linspace(-ny0//2,ny0//2,ny0)); 
r0=np.sqrt(x0**2+y0**2)/nr0;
#     theta0=2*np.arcsin(r0/np.sqrt(2))

roots=np.zeros(51)
rr=np.arange(51)/100.0
c1,c2,c3=params[camera][6:9]
for i,ref in enumerate(rr):
    roots[i]=np.real(np.roots([c3,0,c2,0,c1,-ref])[-1])
theta0=np.interp(r0/2,rr,roots)
              
phi0 = np.arctan2(x0,y0) - params[camera][3]  ####phi (i.e., azimuth) is reckoned with -pi corresponding to north, increasing clockwise, NOTE: pysolar use sub-standard definition
phi0=phi0%(2*np.pi)

beta,azm=params[camera][4:6]
theta=theta0; phi=phi0
#####correction for the mis-pointing error
k=np.array((np.sin(azm),np.cos(azm),0))
a=np.array([np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)]); 
a = np.transpose(a,[1,2,0])
b=np.cos(beta)*a + np.sin(beta)*np.cross(k,a,axisb=2) \
  + np.reshape(np.outer(np.dot(a,k),k),(ny0,nx0,3))*(1-np.cos(beta))
theta=np.arctan(np.sqrt(b[:,:,0]**2+b[:,:,1]**2)/b[:,:,2])
phi=np.arctan2(b[:,:,1],b[:,:,0])%(2*np.pi)

theta_filter = (theta>max_theta) | (theta<=0); theta[theta_filter]=np.nan;

#####coordinate system for the undistorted space
r=np.tan(theta); 
x,y=r*np.sin(phi), r*np.cos(phi)        
filepath = inpath + day + '/'
flist = sorted(glob.glob(filepath + '*jpg'))

for f in sorted(flist):  ###8200
    print(f)   
#         ######read the image to array
    im0=plt.imread(f).astype(np.float32);        
    im0=im0[ystart:ystart+ny0,xstart:xstart+nx0,:]
    im0[theta_filter,:]=np.nan           
   
    im=np.zeros((ny,nx,3))
    for i in range(3):
        im[:,:,i]=st.bin_average2_reg(im0[:,:,i],x,y,xbin,ybin,mask=valid);    
        im[:,:,i]=st.fill_by_mean2(im[:,:,i],7, mask=(np.isnan(im[:,:,i])) & valid )  

    ims = Image.fromarray(im.astype(np.uint8))
    ims.save(outpath+camera+'/'+day+'/'+os.path.basename(f)[:-3]+'jpg', "JPEG")

