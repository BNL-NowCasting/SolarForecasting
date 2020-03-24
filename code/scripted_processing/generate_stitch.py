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
        #for i, _ in enumerate(p.imap_unordered(stitch_MP, args), 0):
        #    sys.stderr.write('\r\t {0:%}'.format(i/len(args)))         
        p.imap(stitch_MP,args,chunksize=16)  
        p.close()
        p.terminate()   #Added because some workers were hanging, should probably be resolved more elegantly
        p.join()
            
def stitch_MP(args):
    ymdhms, cam_list = args
    imgs=[]
    
    for camID, pkl in cam_list.items():
        with open(pkl,'rb') as input:
            try:
                img=pickle.load(input)
            except EOFError:
                img=None
            if img is not None:
                imgs+=[img];
        if len(imgs)<=0:
            continue
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


def height(args_list, cores):

    p = multiprocessing.Pool(cores,maxtasksperchild=128)
    
    for args in args_list:
        imager,neighbors,day=args 
        ymd=day[:8]
        print('\t'+inpath+imager.camID+'/'+ymd+'/')
        flist = sorted(glob.glob(inpath+imager.camID+'/'+ymd+'/'+imager.camID+'_'+day+'*jpg'))
        
        if (REPROCESS==False):
            flist_done = sorted(glob.glob(tmpfs+'/'+ymd+'/'+imager.camID+'_'+day+'*hkl'))
            if len(flist_done) > 0:
                for f in flist_done:
                    #try:
                    #print(inpath+imager.camID+'/'+ymd+'/'+f[-23:-4]+'.jpg')
                    flist.remove(inpath+imager.camID+'/'+ymd+'/'+f[-23:-4]+'.jpg')
                    #except Exception:
                    #    print("Mismatched processing.  Consider re-running with REPROCESS = true.")
                print("\tSkipped %i files that were already processed for %s" % (len(flist_done),imager.camID))
                
                
        
        print("\tFound %i image files for %s" % (len(flist),imager.camID))
        if len(flist)<=0:
            continue    
        mp_args=[[args, f] for f in flist]    
        #for i, _ in enumerate(p.imap_unordered(height_MP, mp_args), 0):
        #    sys.stderr.write('\r\t {0:%}'.format(i/len(mp_args)))        
        p.imap(height_MP,mp_args,chunksize=16)
        
    p.close()
    p.join()
    
# def sunAngleValid(fn):
    # t_local=datetime.strptime(fn[-18:-4],'%Y%m%d%H%M%S');
    # t_std = t_local.replace(tzinfo=timezone(-timedelta(hours=5)))       
    # gatech = ephem.Observer(); 
    # gatech.date = t_std.strftime('%Y/%m/%d %H:%M:%S')
    # gatech.lat, gatech.lon = str(self.lat),str(self.lon)
    # sun=ephem.Sun()  ; sun.compute(gatech);
    # sz = np.pi/2-sun.alt; 
    # self.sz = sz
    # if (sz>75*deg2rad):
        # return False
    # else:
        # return True
    
    
def height_MP(mp_args):
    args,f = mp_args
    imager,neighbors,day = args

    #for f in flist:
    basename=f[-23:-4]
    #if os.path.isfile(tmpfs+f[-18:-10]+'/'+basename+'.hkl') and (~REPROCESS):  ######already processed, skip
    #    return          
    
    print('Processing', basename)
    print("Full name: %s" % tmpfs+f[-18:-10]+'/'+basename+'*pkl')
    fpickle = glob.glob(tmpfs+f[-18:-10]+'/'+basename+'*pkl')
    img=None
    #print("len(fpickle)=%i" % len(fpickle))
    if len(fpickle)<=0:
        print("\tPreprocessing\n")
        img=cam.preprocess(imager,f,tmpfs);
    else:
        with open(fpickle[0],'rb') as input:
            try:
                img=pickle.load(input);
            except EOFError:
                img=None                    
    if img is None or img.red is None:
        return
    if img.layers<=0:
        img.dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.hkl');
        return;

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
                    print('Cloud height computed for', f[-23:]);
                    print('Cloud layer',ih+1,':',res,' computed with cameras ',img.camID,img1.camID,'(distance:',distance,'m)')

                    #if not SAVE_FIG:
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

        # days=['20180823124','20180829165'];
        # # days=['20180825161']; ####multilayer cloud
        # days=['20180829165']
        # days=['20180829162']    #####scattered cloud
        # days=['20181001141']    #####scattered cloud
        # days=['20180922150']    ####high clouds

        #days=['201809021[3-9]','201809031[3-5]','201809051[5-7]','201809111[7-8]','201809151[4-6]','201809161[6-8]','201809171[4-6]'\
        #      ,'201809191[6-9]','201809192[0-1]','201809201[5-6]','201809221[4-9]','201810011[3-7]','201810021[7-9]'\
        #      ,'2018100220','2018100320','201810031[3-9]','201810041[3-6]','201810051[3-5]']

        # days=['201809151[4-6]','201809171[4-6]','2018100220','2018100320']

        # days=['201809{:02d}15'.format(iday) for iday in range(1,9)]
        # days=['201809151[3,7,8,9]','2018091520','2018092213','2018092220'] 
        #days=['201809{:02d}1[3-9]'.format(iday) for iday in range(1,31)]
        #days=['20181001141']    #####scattered cloud
        #days=['2019102816']  #2019-10-28 4PM
        days=le(cp["forecast"]["days"])

        inpath=le(cp["paths"]["inpath"])
        # tmpfs='/dev/shm/'
        tmpfs=le(cp["paths"]["tmpfs"])
        stitch_path=le(cp["paths"]["stitch_path"])
        
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
        cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000,cam_tz=cam_tz) 

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
                
    #p = multiprocessing.Pool(len(cid_flat)) 
    for day in days:
       if not os.path.isdir(tmpfs+day[:8]):
           try:
               subprocess.call(['mkdir', tmpfs+day[:8]]) 
           except:
               print('Cannot create directory,',tmpfs+day[:8])
               continue  
#         args=[[[camID for camID in camg], day] for camg in camIDs]    
       print("Running Preprocessing/Height for: %s" % day)
       args=[[cameras[camID], [cameras[cmr] for cmr in height_group[camID]], day] for camID in cid_flat]
       height(args, cores_to_use)  
    #p.close()
    #p.join()
    
    stitch(camIDs, days, cores_to_use)
            



