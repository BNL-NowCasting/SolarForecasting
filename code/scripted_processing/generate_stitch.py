from __future__ import division
import numpy as np
import os,sys, glob
import matplotlib
matplotlib.use('agg')
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
import configparser
from ast import literal_eval as le
import warnings
warnings.filterwarnings("ignore")
import pytz

SAVE_FIG=True
REPROCESS=False; #False  ####reprocess already processed file?
MAX_INTERVAL = 179 ####max allowed interval between two frames for cloud motion estimation
deg2km=6367*np.pi/180

            
def stitch_MP(args):
    ymdhms, cam_list = args
    imgs=[]
    
    for camID, pkl in cam_list.items():
        #print("camID: %s\n" % camID)
        with open(pkl,'rb') as input:
            try:
                img=pickle.load(input)
            except EOFError:
                img=None
            if img is not None:
                imgs+=[img];
        if len(imgs)<=0:
            continue
            print("\t\t%s: No images found for %s" % (ymdhms, camID))
#            print(selected)

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
        #h[ilayer]=0.675; height=0.675
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
        stch.dump_stitch(stitch_path+ymdhms[:8]+'/'+ymdhms+'.sth');
        
        plt.ioff()  #Turn off interactive plotting for running automatically
        plt.figure(); plt.imshow(stch.rgb,extent=[0,xlen,ylen,0]);
        plt.xlabel('East distance, km'); plt.ylabel('South distance, km')
        plt.tight_layout();
#                plt.show();
        plt.savefig(stitch_path+ymdhms[:8]+'/'+ymdhms+'.png'); plt.close(); 
    
    return



def stitch(cams, dates, cores_to_use):
    for day in dates:

        print('Stitching ',day)
        if os.path.isfile(stitch_path+day+'*.sth') and (~REPROCESS):  ######already processed, skip
            print('(Skipped)')
            continue

        selected=[]
        pkl_dict = {}
        for camID in all_cams:
            if camID not in cid_flat or stitch_pair[camID] in selected:
                continue;
            selected += camID
            pkls=sorted(glob.glob(tmpfs+day[:8]+'/'+camID+'_'+day+'*.hkl'));
            print("\t%s: %i hkl files found" % (camID, len(pkls)))
            for pkl in pkls:
                try:
                    pkl_dict[pkl[-18:-4]].update({camID:pkl})
                except KeyError:
                    pkl_dict[pkl[-18:-4]] = {camID:pkl}

        print("\n\tUsing %i cores to process %i image times" % (cores_to_use, len(pkl_dict)))
        p = multiprocessing.Pool(cores_to_use,maxtasksperchild=128)      
        args=[[ymdhms, cam_list] for ymdhms, cam_list in pkl_dict.items()]      
        #print("len(args)=%i" % len(args))        
        #for i, _ in enumerate(p.imap_unordered(stitch_MP, args), 0):
        #    sys.stderr.write('\r\t {0:%}'.format(i/len(args)))         
        p.map(stitch_MP,args,chunksize=16)
        p.close()
        p.terminate()   #Added because some workers were hanging, should probably be resolved more elegantly
        p.join()
        

if __name__ == "__main__":

    try:
        try:
            config_path = sys.argv[1]
        except Exception:
            config_path = "./config.conf"
        cp = configparser.ConfigParser()
        cp.read(config_path)


        all_cams=le(cp["cameras"]["all_cams"])
        height_group=le(cp["cameras"]["height_group"])
        stitch_pair=le(cp["cameras"]["stitch_pair"])
        camIDs=le(cp["cameras"]["camIDs"])
        cid_flat=camIDs+[stitch_pair[camID] for camID in camIDs]

        days=le(cp["forecast"]["days"])

        inpath=le(cp["paths"]["inpath"])
        # tmpfs='/dev/shm/'
        tmpfs=le(cp["paths"]["tmpfs"])
        stitch_path=le(cp["paths"]["stitch_path"])
        static_mask_path=le(cp["paths"]["static_mask_path"])
        
        try:
            cam_tz=pytz.timezone(cp["cameras"]["cam_timezone"])
            print("Using camera timezone: %s" % str(cam_tz))
        except Exception:
            cam_tz=pytz.timezone("utc")    
            print("Error processsing cameara timezone config, assuming UTC")
        
        try:
            cores_to_use = int(cp["server"]["cores_to_use"])
        except Exception:
            cores_to_use = 20

    except KeyError as e:
        print("Error loading config: %s" % e)
        
    plt.ioff()  #Turn off interactive plotting for running automatically
    
    cameras={};
    for camID in all_cams:
        cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000,cam_tz=cam_tz,static_mask_path=static_mask_path) 

    lon0,lat0=cameras['HD5A'].lon,cameras['HD5B'].lat
    x_cams=(cameras['HD1B'].lon-lon0)*deg2km*np.cos(cameras['HD1B'].lat*np.pi/180)
    y_cams=(lat0-cameras['HD1B'].lat)*deg2km  
    
    
    print("DAYS: %s" % days)
    for day in days:
        if not os.path.isdir(stitch_path+day[:8]):
            try:
                subprocess.call(['mkdir', stitch_path+day[:8]]) 
            except:
                print('Cannot create directory,',stitch_path+day[:8])
                continue 
                
    stitch(camIDs, days, cores_to_use)
            



