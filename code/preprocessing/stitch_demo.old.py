import numpy as np
import os,sys, glob
from matplotlib import pyplot as plt
import camera as cam
import stat_tools as st
from scipy.ndimage import morphology,filters, sobel  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque
import multiprocessing,subprocess,pickle
import time, geo
from functools import reduce
from operator import concat

SAVE_FIG=True
deg2km=6367*np.pi/180


all_cams=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
height_group={'HD1B':['HD1C', 'HD2B'], 'HD1C':['HD1B', 'HD2C'], 'HD2B':['HD2C', 'HD3A'], 'HD2C':['HD2B', 'HD3B'],\
       'HD3A':['HD3B','HD4A'], 'HD3B':['HD3A', 'HD4B'], 'HD4A':['HD4B','HD5A'], 'HD4B':['HD4A', 'HD5A', 'HD3B'],\
       'HD5A':['HD5B', 'HD4A', 'HD4B'], 'HD5B':['HD5A', 'HD4B']}
stitch_pair={'HD1B':'HD1C', 'HD1C':'HD1B','HD2B':'HD2C','HD2C':'HD2B','HD3A':'HD3B','HD3B':'HD3A', 'HD4A':'HD4B','HD4B':'HD4A', 'HD5A':'HD5B','HD5B':'HD5A'}
# camIDs=[['HD1B','HD1C'],['HD2B','HD2C'],['HD3A','HD3B'],['HD4A','HD4B'],['HD5A','HD5B']];
camIDs=['HD1B','HD2B','HD3A','HD4A','HD5A'];
cid_flat=camIDs+[stitch_pair[camID] for camID in camIDs]

days=['20180823124','20180829165'];
# days=['20180825161']; ####multilayer cloud
days=['20180829165']
days=['20180829162']    #####scattered cloud
days=['20181001141']    #####scattered cloud
days=['20180922150']    ####high clouds

inpath='~/data/images/' 
outpath='~/ldata/results_parallel/'
tmpfs='/dev/shm/'


cameras={};
for camID in all_cams:
    cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000) 
    
x_HD5A_HD1B=(cameras['HD1B'].lon-cameras['HD5A'].lon)*deg2km*np.cos(cameras['HD1B'].lat*np.pi/180)
y_HD5A_HD1B=(cameras['HD5A'].lat-cameras['HD1B'].lat)*deg2km  
        
def stitch(cams, dates):        
    for day in dates:                       
        flist = sorted(glob.glob(inpath+camIDs[0]+'/'+day[:8]+'/'+camIDs[0]+'_'+day+'*jpg'))        
            
        if len(flist)<=1:
            continue
        for f in flist[1:]:
            ymdhms=f[-18:-4]
            
            counter=0;
            selected=[]; imgs=[]  
            for counter in range(20):
                pkls=sorted(glob.glob(tmpfs+day[:8]+'/HD*_'+ymdhms+'.hkl'));
                for pkl in pkls:
                    camID=pkl[-23:-19]
                    if camID not in cid_flat or camID in selected or stitch_pair[camID] in selected:
                        continue;
                    with open(pkl,'rb') as input:
                        imgs+=[pickle.load(input)];
                        selected += [camID]                  
                if len(selected)>=len(camIDs)-2:
                    break;
                time.sleep(5);           
            print(selected)
            
            h=[]
            for i, img in enumerate(imgs):                
                h += [img.height]
            h=np.array(h);

            h=np.nanmedian(h,axis=0);
            print(h)
            max_tan=np.tan(imgs[0].max_theta*np.pi/180) 
            for ilayer in range(img.layers):
                if np.isnan(h[ilayer]): continue
                pixel_size=2*h[ilayer]/1000*max_tan/imgs[0].nx
                xlen,ylen=2*h[ilayer]/1000*max_tan+x_HD5A_HD1B, 2*h[ilayer]/1000*max_tan+y_HD5A_HD1B
#                 print(pixel_size,xlen,ylen)
                stitched=np.zeros((10+int(ylen//pixel_size),10+int(xlen//pixel_size),3),dtype=np.float32)
                cnt=np.zeros(stitched.shape[:2],dtype=np.float32);
                for i, img in enumerate(imgs):
                    start_x=(img.lon-cameras['HD5A'].lon)*deg2km*np.cos(img.lat*np.pi/180)/pixel_size; start_x=int(start_x)
                    start_y=(cameras['HD5A'].lat-img.lat)*deg2km/pixel_size; start_y=int(start_y)
                    
                    tmp=np.flip(img.rgb,axis=1); #tmp[img.cm!=ilayer+1,:]=0;                    
                    mk=tmp[...,0]>0
#                     print(img.camID,ilayer,h[ilayer],start_x,start_y,mk.shape,stitched.shape)
                    stitched[start_y:start_y+img.ny,start_x:start_x+img.nx][mk]+=tmp[mk]
                    cnt[start_y:start_y+img.ny,start_x:start_x+img.nx]+=mk
                                    
                for i in range(3):
                    stitched[...,i]/=cnt
                plt.figure(); plt.imshow(stitched.astype(np.uint8),extent=[0,xlen,ylen,0]);
                plt.xlabel('East distance, km'); plt.ylabel('South distance, km')
                plt.tight_layout();
                plt.show();
#             fig.savefig(outpath+ymdhms); plt.close(); 

def height(args):
    imager,neighbors,day=args 
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+imager.camID+'/'+ymd+'/'+imager.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return
     
    for f in flist:
        basename=f[-23:-4]
#         print('Procesing', basename)
        fpickle = glob.glob(tmpfs+f[-18:-10]+'/'+basename+'*pkl')
        img=None
        if len(fpickle)<=0:
            img=cam.preprocess(imager,f,tmpfs);
        else:
            with open(fpickle[0],'rb') as input:
                img=pickle.load(input);
        if img is None or img.red is None or img.layers<=0:
            continue
            
        h = [np.nan]*img.layers
        for inghb,neighbor in enumerate(neighbors):
            bname=basename.replace(imager.camID,neighbor.camID);
            fp_nb = glob.glob(tmpfs+f[-18:-10]+'/'+bname+'*pkl')  
            img1=None;
            if len(fp_nb)<=0:
                fnb=f.replace(imager.camID,neighbor.camID) 
                img1=cam.preprocess(neighbor,fnb,tmpfs);  ###img object contains four data fields: rgb, red, rbr, and cm 
            else:
                with open(fp_nb[0],'rb') as input:
                    img1=pickle.load(input);
            if img1 is None or img1.red is None or img.layers<=0:
                continue                           
            
            distance = 6367e3*geo.distance_sphere(img.lat,img.lon,img1.lat,img1.lon)
            for ih in range(img.layers):
                if np.isfinite(h[ih]):
                    continue
                if (ih>=1) and (distance<500):
                    break;
                res=cam.cloud_height(img,img1,layer=ih+1, distance=distance)
                if np.isfinite(res) and res<20*distance and res>0.5*distance:
                    h[ih]=int(res);
#                     print('Cloud height computed for', f[-23:]);
#                     print('Cloud layer',ih+1,':',res,' computed with cameras ',img.camID,img1.camID,'(distance:',distance,'m)')
                    if not SAVE_FIG:
                        fig,ax=plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True);
                        ax[0,0].imshow(img.rgb); ax[0,1].imshow(img1.rgb); 
                        ax[0,0].set_title(img.camID); ax[0,1].set_title(img1.camID)
                        ax[1,0].imshow(img.cm); ax[1,1].imshow(img1.cm); 
                        ax[1,0].set_title(str(6367e3*geo.distance_sphere(img.lat,img.lon,img1.lat,img1.lon)))  
                        plt.tight_layout(); 
                        plt.show();                     
            
            if np.isfinite(h[-1]):                
                break                               
            
        img.height=h;
#         img.red=None;
        img.dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.hkl');


if __name__ == "__main__":  
   
    
    p0=multiprocessing.Process(target=stitch, args=(camIDs, days,))
    p0.start(); 

    p = multiprocessing.Pool(len(camIDs)) 
 
    for day in days:
        if not os.path.isdir(tmpfs+day[:8]):
            try:
                subprocess.call(['mkdir', tmpfs+day[:8]]) 
            except:
                print('Cannot create directory,',tmpfs+day[:8])
                continue  
#         args=[[[camID for camID in camg], day] for camg in camIDs]      
        args=[[cameras[camID], [cameras[cmr] for cmr in height_group[camID]], day] for camID in cid_flat]                   
        p.map(height,args)  
            
    p0.join(); 
            



