import glob
import matplotlib.pyplot as plt
import pickle
import camera
import imageio
import stat_tools as st

stitch_path='~/ldata/stitch/'
day='20181001141'    #####scattered cloud
# day='20180922150'    ####high clouds

flist = sorted(glob.glob(stitch_path+day[:8]+'/'+day+'*sth'))

imgs=[]
for f in flist:
    with open(f,'rb') as input:
        img=pickle.load(input);
        ny,nx = img.cm.shape
        print(img.time,img.lon,img.lat,img.sz,img.saz,img.pixel_size,img.v,img.height);
        xlen=nx*img.pixel_size; ylen=ny*img.pixel_size;

#         imgs += [img.rgb]
        imgs += [st.block_average2(img.rgb,3)]
#         plt.figure(); plt.imshow(img.rgb,extent=[0,xlen,ylen,0]);
#         plt.xlabel('East distance, km'); plt.ylabel('South distance, km')
#         plt.tight_layout();
#         plt.show();
# #             fig.savefig(outpath+ymdhms); plt.close(); 

# imageio.mimsave('high_cloud.gif',imgs,'GIF',duration=0.5);
imageio.mimsave('low_cloud.gif',imgs,'GIF',duration=0.5);

