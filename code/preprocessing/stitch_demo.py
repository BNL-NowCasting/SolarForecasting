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
REPROCESS=False  ####reprocess already processed file?
MAX_INTERVAL = 179 ####max allowed interval between two frames for cloud motion estimation
deg2km=6367*np.pi/180


all_cams=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
height_group={'HD1B':['HD1C', 'HD2B'], 'HD1C':['HD1B', 'HD2C'], 'HD2B':['HD2C', 'HD3A'], 'HD2C':['HD2B', 'HD3B'],\
       'HD3A':['HD3B','HD4A'], 'HD3B':['HD3A', 'HD4B'], 'HD4A':['HD4B','HD5A'], 'HD4B':['HD4A', 'HD5A', 'HD3B'],\
       'HD5A':['HD5B', 'HD4A', 'HD4B'], 'HD5B':['HD5A', 'HD4B']}
stitch_pair={'HD1B':'HD1C', 'HD1C':'HD1B','HD2B':'HD2C','HD2C':'HD2B','HD3A':'HD3B','HD3B':'HD3A', 'HD4A':'HD4B','HD4B':'HD4A', 'HD5A':'HD5B','HD5B':'HD5A'}
# camIDs=[['HD1B','HD1C'],['HD2B','HD2C'],['HD3A','HD3B'],['HD4A','HD4B'],['HD5A','HD5B']];
camIDs=['HD1B','HD2B','HD3A','HD4A','HD5A'];
cid_flat=camIDs+[stitch_pair[camID] for camID in camIDs]

# days=['20180823124','20180829165'];
# # days=['20180825161']; ####multilayer cloud
# days=['20180829165']
# days=['20180829162']    #####scattered cloud
days=['20181001141']    #####scattered cloud
# days=['20180922150']    ####high clouds

inpath='~/data/images/' 
# tmpfs='/dev/shm/'
tmpfs='~/ldata/tmp/'
stitch_path='~/ldata/stitch/'


cameras={};
for camID in all_cams:
    cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000) 

lon0,lat0=cameras['HD5A'].lon,cameras['HD5B'].lat
x_cams=(cameras['HD1B'].lon-lon0)*deg2km*np.cos(cameras['HD1B'].lat*np.pi/180)
y_cams=(lat0-cameras['HD1B'].lat)*deg2km  

      
def stitch(cams, dates):        
    for day in dates:                       
        flist = sorted(glob.glob(inpath+camIDs[0]+'/'+day[:8]+'/'+camIDs[0]+'_'+day+'*jpg'))        
            
        if len(flist)<=1:
            continue
        for f in flist[1:]:
            ymdhms=f[-18:-4]
            print('Processing',ymdhms)
            if os.path.isfile(stitch_path+ymdhms+'.sth') and (~REPROCESS):  ######already processed, skip
                continue
            
            counter=0;
            selected=[]; imgs=[]  
            for counter in range(20):
                pkls=sorted(glob.glob(tmpfs+day[:8]+'/HD*_'+ymdhms+'.hkl'));
                for pkl in pkls:
                    camID=pkl[-23:-19]
                    if camID not in cid_flat or camID in selected or stitch_pair[camID] in selected:
                        continue;
                    with open(pkl,'rb') as input:
                        try:
                            img=pickle.load(input)
                        except EOFError:
                            img=None
                        if img is not None:
                            imgs+=[img];
                            selected += [camID]                                                                      
                if len(selected)>=len(camIDs)-2:
                    break;
                time.sleep(5); 
            if len(imgs)<=0:
                continue
#             print(selected)
            
            h=[]; v=[]
            for i, img in enumerate(imgs):
                if np.isfinite(img.height):                
                    h += [img.height]
                if len(img.v)>=1:    
                    v += [img.v]
            if len(h)<=0 or len(v)<=0:  ####clear sky
                h=[15]; 
                v=[[0,0]];
            else:
                h=np.array(h)/1e3; v=np.array(v)
                h=np.nanmedian(h,axis=0); v=np.nanmedian(v,axis=0);
            
            max_tan=np.tan(imgs[0].max_theta*np.pi/180) 
            for ilayer,height in enumerate(h):
                if np.isnan(h[ilayer]): continue
                stch=cam.stitch(ymdhms); 
                stch.sz,stch.saz=imgs[0].sz,imgs[0].saz
                stch.height=height; stch.v=v
                
                pixel_size=2*h[ilayer]*max_tan/imgs[0].nx; 
                stch.pixel_size=pixel_size;
                xlen,ylen=2*h[ilayer]*max_tan+x_cams, 2*h[ilayer]*max_tan+y_cams
                nstch_y,nstch_x=int(ylen//pixel_size),int(xlen//pixel_size)
                stch.lon=lon0-h[ilayer]*max_tan/deg2km/np.cos(cameras['HD3A'].lat*np.pi/180); 
                stch.lat=lat0+h[ilayer]*max_tan/deg2km;
#                 print(pixel_size,xlen,ylen)
                rgb=np.zeros((nstch_y,nstch_x,3),dtype=np.float32)
                cnt=np.zeros((nstch_y,nstch_x),dtype=np.uint8);
                cm=np.zeros((nstch_y,nstch_x),dtype=np.float32)              
                for i, img in enumerate(imgs):
                    start_x=(img.lon-lon0)*deg2km*np.cos(img.lat*np.pi/180)/pixel_size; start_x=int(start_x)
                    start_y=(lat0-img.lat)*deg2km/pixel_size; start_y=int(start_y)
                    
                    tmp=np.flip(img.rgb,axis=1); #tmp[img.cm!=ilayer+1,:]=0;                    
                    mk=tmp[...,0]>0
#                     print(img.camID,ilayer,h[ilayer],start_x,start_y,mk.shape,stitched.shape)
                    rgb[start_y:start_y+img.ny,start_x:start_x+img.nx][mk]+=tmp[mk]
                    cnt[start_y:start_y+img.ny,start_x:start_x+img.nx]+=mk
                    
                    if (img.cm is not None):
                        tmp=np.flip(img.cm,axis=1); #tmp[img.cm!=ilayer+1,:]=0;                    
                        cm[start_y:start_y+img.ny,start_x:start_x+img.nx][mk]+=tmp[mk]                
                                    
                for i in range(3):
                    rgb[...,i]/=cnt
                cm/=cnt
#                 fig,ax=plt.subplots(1,2); ax[0].imshow(cnt); ax[1].imshow(rgb.astype(np.uint8)); plt.show() 
                stch.rgb=rgb.astype(np.uint8); stch.cm=(cm+0.5).astype(np.uint8)
#                 stch.dump_stitch(stitch_path+'/'+day[:8]+'/'+ymdhms+'.sth');
                
                plt.figure(); plt.imshow(stch.rgb,extent=[0,xlen,ylen,0]);
                plt.xlabel('East distance, km'); plt.ylabel('South distance, km')
                plt.tight_layout();
                plt.show();
# #             fig.savefig(outpath+ymdhms); plt.close(); 


def height(args):
    imager,neighbors,day=args 
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+imager.camID+'/'+ymd+'/'+imager.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return
     
    for f in flist:
        basename=f[-23:-4]
        if os.path.isfile(tmpfs+f[-18:-10]+'/'+basename+'.hkl') and (~REPROCESS):  ######already processed, skip
            continue          
        
#         print('Procesing', basename)
        fpickle = glob.glob(tmpfs+f[-18:-10]+'/'+basename+'*pkl')
        img=None
        if len(fpickle)<=0:
            img=cam.preprocess(imager,f,tmpfs);
        else:
            with open(fpickle[0],'rb') as input:
                try:
                    img=pickle.load(input);
                except EOFError:
                    img=None                    
        if img is None or img.red is None:
            continue
        if img.layers<=0:
            img.dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.hkl');
            continue;

        if img.layers>=1:
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
                        try:
                            img1=pickle.load(input);
                        except EOFError:
                            img1=None
                                
                if img1 is None or img1.red is None:
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
#             img.height+=[h];
            img.height=h;
        
        img.dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.hkl');


if __name__ == "__main__":  
    for day in days:
        if not os.path.isdir(stitch_path+day[:8]):
            try:
                subprocess.call(['mkdir', stitch_path+day[:8]]) 
            except:
                print('Cannot create directory,',stitch_path+day[:8])
                continue  
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
            



