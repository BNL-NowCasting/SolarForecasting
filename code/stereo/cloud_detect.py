import numpy as np
import glob
from matplotlib import pyplot as plt
import imgmatch as ig 
import scipy.ndimage as ndimage 
import tsi
import stat_tools as st

#f1=glob.glob('.\\tsi2\*2013061420*jpg'); f2=glob.glob('.\\tsi1\*141020.bmp');
f1=glob.glob('.\\tsi1\*140600.bmp'); f2=glob.glob('.\\tsi1\*140620.bmp'); f3=glob.glob('.\\tsi1\*140640.bmp');
frame1=plt.imread(f1[0]); frame1=np.float32(frame1); 
frame2=plt.imread(f2[0]); frame2=np.float32(frame2); 
frame3=plt.imread(f3[0]); frame3=np.float32(frame3); 

r1=frame1[:,:,0]; r2=frame2[:,:,0]; r3=frame3[:,:,0];   
rb1=r1-frame1[:,:,2]; rb2=r2-frame2[:,:,2]; rb3=r3-frame3[:,:,2];
nan_mask=(r1<30);
nan_mask=ndimage.morphology.binary_dilation(nan_mask,iterations=3)
rb1[nan_mask]=np.nan; r1[nan_mask]=np.nan;
nan_mask=(r2<30);
nan_mask=ndimage.morphology.binary_dilation(nan_mask,iterations=3)
rb2[nan_mask]=np.nan; r2[nan_mask]=np.nan;
nan_mask=(r3<30);
nan_mask=ndimage.morphology.binary_dilation(nan_mask,iterations=3)
rb3[nan_mask]=np.nan; r3[nan_mask]=np.nan;

ot1=r1-tsi.moving_avg(r1,40,40); ot2=r2-tsi.moving_avg(r2,40,40); ot3=r3-tsi.moving_avg(r3,40,40);

#r1[np.isnan(r1)]=np.nanmean(r1); r2[np.isnan(r2)]=np.nanmean(r2); 
cm1=np.all((rb1>-43,ot1>2.0),axis=0); cm1[rb1>-32]=1;
cm2=np.all((rb2>-43,ot2>2.0),axis=0); cm2[rb2>-32]=1;
cm3=np.all((rb3>-43,ot3>2.0),axis=0); cm3[rb3>-32]=1;

c=np.zeros((40,40));
for ix in range(40):
    for iy in range(40):
        c[ix,iy]=ig.cost_fun((ix-20,iy-20),r1,r2); 
#t0=time.time();        
c=ndimage.filters.gaussian_filter(c, 2, mode='nearest') 
#print(time.time()-t0)       
#neighborhood = ndimage.morphology.generate_binary_structure(2,2)
#apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
#local_min = ndimage.filters.minimum_filter(c, footprint=neighborhood); 
is_min=(c==ndimage.filters.minimum_filter(c, 20));
iy_min,ix_min = np.nonzero(is_min); 
r1_s1=st.shift_2d(r1,20-ix_min[0],20-iy_min[0]);
rdf=abs(r2-r1_s1); 
foo=np.all((rdf>20,cm2),axis=0);
r1_s2=st.shift_2d(r1,20-ix_min[1],20-iy_min[1]);
rdf2=abs(r2-r1_s2); 
foo=np.argmin((rdf,rdf2),axis=0); foo[abs(rdf-rdf2)<5]=-2; foo[np.isnan(rdf)]=-1; foo[np.isnan(rdf2)]=-1; foo[cm2<0.5]=-1;
#foo[np.all((foo==0,abs(st.shift_2d(foo,20-ix_min[0],20-iy_min[0],constant=-2))>0.5),axis=0)]=-1;

r3_s1=st.shift_2d(r3,ix_min[0]-20,iy_min[0]-20);
r3_s2=st.shift_2d(r3,ix_min[1]-20,iy_min[1]-20);

plt.figure(); ax1=plt.subplot(2,2,1); im1=ax1.imshow(r2-r1_s2); plt.colorbar(im1);
ax2=plt.subplot(2,2,2,sharex=ax1,sharey=ax1); im2=ax2.imshow(foo); plt.colorbar(im2);
ax3=plt.subplot(2,2,3,sharex=ax1,sharey=ax1); im3=ax3.imshow(r2); plt.colorbar(im3);
ax4=plt.subplot(2,2,4,sharex=ax1,sharey=ax1); im4=ax4.imshow(r2-r3_s2); plt.colorbar(im4);       
       

