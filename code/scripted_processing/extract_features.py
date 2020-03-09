import numpy as np
import os, sys, glob, subprocess
from matplotlib import pyplot as plt
import pickle
import stat_tools as st
import multiprocessing
import datetime as dt
import configparser
from ast import literal_eval as le
import pytz

SAVE_FIG=True
deg2km=6367*np.pi/180
deg2rad=np.pi/180
WIN_SIZE = 50
INTERVAL = 0.5 ####0.5 min or 30 sec

def extract_MP(args):
    iGHI,iy0,ix0,ny,nx,img,timestamp = args
#for iGHI in range(len(GHI_loc)):
    #iy0, ix0 = iys[iGHI], ixs[iGHI]
    #print("\tExtracting t=%s for %i: %i, %i" % (timestamp,iGHI,iy0,ix0))
    slc=np.s_[max(0,iy0-WIN_SIZE):min(ny-1,iy0+WIN_SIZE),max(0,ix0-WIN_SIZE):min(nx-1,ix0+WIN_SIZE)]
    if img.cm[slc].size >= 1:
        rgb0 = img.rgb.astype(np.float32); 
        print(rgb0[rgb0>0])
        rgb0[rgb0<=0] = np.nan
        rgb = np.reshape(rgb0[slc], (-1,3));
        R_mean1, G_mean1, B_mean1 = np.nanmean(rgb,axis=0);
        if np.isnan(R_mean1):
            print("\tAt timestamp %s, sensor %i, np.isnan(R_mean1)==true" % (timestamp,iGHI))
            return
        R_min1, G_min1, B_min1 = np.nanmin(rgb,axis=0);
        R_max1, G_max1, B_max1 = np.nanmax(rgb,axis=0);
        RBR1 = (R_mean1 - B_mean1) / (R_mean1 + B_mean1)
        cf1 = np.sum(img.cm[slc]) / np.sum(rgb[:,0]>0);
        
        out_args = []
        for ilt, lead_time in enumerate(lead_steps):
            iy = int(0.5 + iy0 + img.v[0][0]*lead_time); 
            ix = int(0.5 + ix0 - img.v[0][1]*lead_time);   #####need to revert vx since the image is flipped
            slc=np.s_[max(0,iy-WIN_SIZE):min(ny-1,iy+WIN_SIZE),max(0,ix-WIN_SIZE):min(nx-1,ix+WIN_SIZE)]
            if img.cm[slc].size >= 1:
                rgb = np.reshape(rgb0[slc], (-1,3));
                R_mean2,G_mean2,B_mean2 = np.nanmean(rgb,axis=0);
                if np.isnan(R_mean2):
                    #return
                    continue
                R_min2,G_min2,B_min2 = np.nanmin(rgb,axis=0);
                R_max2,G_max2,B_max2 = np.nanmax(rgb,axis=0);
                RBR2 = (R_mean2 - B_mean2) / (R_mean2 + B_mean2)
                cf2 = np.sum(img.cm[slc]) / np.sum(rgb[:,0]>0);
                tmp = np.asarray([lead_minutes[ilt],timestamp,img.height,img.sz,\
                        cf1, R_mean1,G_mean1,B_mean1,R_min1,G_min1,B_min1,R_max1,G_max1,B_max1,RBR1,\
                        cf2, R_mean2,G_mean2,B_mean2,R_min2,G_min2,B_min2,R_max2,G_max2,B_max2,RBR2],dtype=np.float32) 
                tmp=np.reshape(tmp,(1,-1));
                out_args += [(iGHI,tmp,', '.join(['%g']+['%f']+['%g']*(tmp.size-2)))]
        return out_args

#fig,ax=plt.subplots(1,2,sharex=True,sharey=True);
#ax[0].imshow(img.rgb); ax[0].scatter(ix,iy,s=3,marker='o');
#ax[1].imshow(img.cm); 
#plt.tight_layout(); 
#plt.savefig(outpath+'GHI'+str(iGHI+1)+'.png')
#plt.show();

def localToUTCtimestamp(t, local_tz):
    t_local = local_tz.localize(t, is_dst=None)
    t_utc = t_local.astimezone(pytz.utc)
    return t_utc.timestamp()


if __name__ == "__main__":

    try:
        try:
            config_path = sys.argv[1]
        except Exception:
            config_path = "./config.conf"
        cp = configparser.ConfigParser()
        cp.read(config_path)

        inpath=le(cp["paths"]["inpath"])
        outpath=le(cp["paths"]["feature_path"])
        tmpfs=le(cp["paths"]["tmpfs"])
        stitch_path=le(cp["paths"]["stitch_path"])

        GHI_Coor = le(cp["GHI_sensors"]["GHI_Coor"])
        GHI_loc=[GHI_Coor[key] for key in sorted(GHI_Coor)]
        GHI_loc=np.array(GHI_loc) #[:1];

        lead_minutes=le(cp["forecast"]["lead_minutes"])
        lead_steps=[lt/INTERVAL for lt in lead_minutes]
        days=le(cp["forecast"]["days"])
     
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

    #print(GHI_loc, len(GHI_loc))

                     

    for day in days:
        if not os.path.isdir(outpath+day[:8]):
            try:
                subprocess.call(['mkdir', outpath+day[:8]]) 
            except:
                print('Cannot create directory,',outpath+day[:8])
                continue
    
        fhs=[]
        for iGHI in range(len(GHI_loc)):        
            fhs += [open(outpath+day[:8]+'/GHI'+str(iGHI+1)+'.csv','wb')]
            
        print("Extracting features for %s, GHI sensors:\n\t%s" % (day, ("\n\t").join(str(fhs).split(","))))
        
        p = multiprocessing.Pool(cores_to_use)  
        flist = sorted(glob.glob(stitch_path+day[:8]+'/'+day+'*sth'))
        #flist = sorted(glob.glob(stitch_path+day[:8]+'/'+day[:8]+'1511*sth'))
        for f in flist:
            with open(f,'rb') as input:
                #timestamp = (dt.datetime.strptime(f[-18:-4], '%Y%m%d%H%M%S')-dt.datetime(2018,1,1)).total_seconds()
                timestamp  = localToUTCtimestamp(dt.datetime.strptime(f[-18:-4], '%Y%m%d%H%M%S'), cam_tz)
                
                print(timestamp, f)
                img=pickle.load(input);
                if not np.isfinite(img.height):
                    continue
                ny,nx = img.cm.shape
        #         print(img.time,img.lon,img.lat,img.sz,img.saz,img.pixel_size,img.v,img.height);
                y, x = (img.lat-GHI_loc[:,0])*deg2km, (GHI_loc[:,1]-img.lon)*deg2km*np.cos(GHI_loc[0,0]*deg2rad);
                iys = (0.5 + (y + img.height*np.tan(img.sz)*np.cos(img.saz))/img.pixel_size).astype(np.int32)
                ixs = (0.5 + (x - img.height*np.tan(img.sz)*np.sin(img.saz))/img.pixel_size).astype(np.int32)
                
                features = p.map(extract_MP,[[iGHI,iys[iGHI],ixs[iGHI],ny,nx,img,timestamp] for iGHI in range(len(GHI_loc))])
                for timesteps in features:
                    if timesteps is not None:
                        for args in timesteps:
                            np.savetxt(fhs[args[0]], *args[1:])
                
        p.close()
        p.join()      

        for fh in fhs:
            fh.close()       
