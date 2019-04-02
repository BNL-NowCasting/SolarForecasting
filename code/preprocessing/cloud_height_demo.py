import numpy as np
import os,sys, glob
from matplotlib import pyplot as plt
import camera as cam
import stat_tools as st
from scipy.ndimage import morphology,filters, sobel  ####more efficient than skimage
from skimage.morphology import remove_small_objects
from collections import deque
import multiprocessing,pickle
import time, geo
import subprocess

SAVE_FIG=True
VIS=False

camIDs=['HD5A', 'HD5B', 'HD4A','HD4B', 'HD3A', 'HD3B','HD2B', 'HD2C', 'HD1B',  'HD1C'];
groups={'HD1B':['HD1C', 'HD2B'], 'HD1C':['HD1B', 'HD2C'], 'HD2B':['HD2C', 'HD1B','HD1C','HD3A'], 'HD2C':['HD2B', 'HD1B','HD1C','HD3A'],\
       'HD3A':['HD3B','HD4A'], 'HD3B':['HD3A', 'HD4B'], 'HD4A':['HD4B','HD5A'], 'HD4B':['HD4A', 'HD5A', 'HD3B'],\
       'HD5A':['HD5B', 'HD4A', 'HD4B'], 'HD5B':['HD5A', 'HD4B']}
ixy={'HD5A':[0,0],'HD5B':[1,0],'HD4A':[2,0],'HD4B':[3,0],'HD3A':[4,0],'HD3B':[0,2], \
     'HD2B':[1,2],'HD2C':[2,2],'HD1B':[3,2],'HD1C':[4,2]}
# camIDs=['HD3A','HD3B', 'HD4A','HD4B','HD2B'];
cams=['HD2B'];
# camIDs=['HD1B','HD1C', 'HD2B'];
cams=['HD3B','HD2B'];

# cams=['HD1B','HD2B','HD3A','HD4A','HD5A'];

if len(sys.argv)>=2:
    days=[sys.argv[1]]
else:
#     days=['2018082312[0,2,4]','20180825161','20180829165','2018082112[0,1]','20180830181','20180824171','20180821132'] 
    days=['20180823124','20180829165'];
    # days=['20180825161']; ####multilayer cloud
    days=['20180829165']    #####scattered cloud
#     days=['2018092921']
#     days=['201810011[4-8]']
#     days=['2018100118']

inpath='~/data/images/' 
outpath='~/data/results_parallel/'
tmpfs='/dev/shm/'

def visualize(camIDs, dates):
    for day in dates:
        if not os.path.isdir(tmpfs+day[:8]):
            os.makedirs(tmpfs+day[:8])
        flist = sorted(glob.glob(inpath+camIDs[0]+'/'+day[:8]+'/'+camIDs[0]+'_'+day+'*.jpg')) 
        if len(flist)<=1:
            continue
        for f in flist[1:]:
            ymdhms=f[-18:-4]
            print(ymdhms)
            counter=0;
            for counter in range(8):
                pkls=sorted(glob.glob(tmpfs+day[:8]+'/HD*_'+ymdhms+'.hkl'));
                if len(pkls)>=max(1,len(camIDs)-5):
                    break;
                time.sleep(5);
                
            fig,ax=plt.subplots(4,5,figsize=(12,10),sharex=True,sharey=True);
            for pkl in pkls:
                camID=pkl[-23:-19]
                if camID not in camIDs:
                    continue;
                ix, iy=ixy[camID]
                ax[iy,ix].set_title(camID);
                ax[iy,ix].set_xticks([]); ax[iy,ix].set_yticks([])
                ax[iy+1,ix].set_xticks([]); ax[iy+1,ix].set_yticks([])                
                img=None
                with open(pkl,'rb') as input:
                    img=pickle.load(input);
                if img is None:
                    continue
                ax[iy+1,ix].set_title(str(img.height));
#                 ax[iy+1,ix].set_title(str(img.v)+str(img.height));
                ax[iy,ix].imshow(img.rgb);
                ax[iy+1,ix].imshow(img.cm,vmax=2);

            plt.tight_layout();
            plt.show();
#             fig.savefig(outpath+ymdhms); plt.close(); 

def height(args):
    imager,neighbors,day=args 
    hs=[];
    hms=[]
    
    ymd=day[:8]
    flist = sorted(glob.glob(inpath+imager.camID+'/'+ymd+'/'+imager.camID+'_'+day+'*jpg'))
    if len(flist)<=0:
        return
     
    for f in flist:
        basename=f[-23:-4]
        print('Procesing', basename)
        hms+=[f[-10:-4]]
        fpickle = glob.glob(tmpfs+f[-18:-10]+'/'+basename+'*pkl')
        img=None
        if len(fpickle)<=0:
            img=cam.preprocess(imager,f,tmpfs);
        else:
            with open(fpickle[0],'rb') as input:
                img=pickle.load(input);
        if img is None or img.red is None or img.layers<=0:
            hs+=[[np.nan]*img.layers]
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
        hs+=[h]
        img.height=h;
#         img.red=None;
        img.dump_img(tmpfs+f[-18:-10]+'/'+f[-23:-4]+'.hkl');
        
#     return [hms,hs];

if __name__ == "__main__":  
    cameras={};
    for camID in camIDs:
        cameras[camID] = cam.camera(camID,max_theta=70,nx=1000,ny=1000)

    if VIS:
        p0=multiprocessing.Process(target=visualize, args=(camIDs, days,))
        p0.start(); 

    p = multiprocessing.Pool(len(cams)) 
 
    for day in days:
        if not os.path.isdir(tmpfs+day[:8]):
            try:
                subprocess.call(['mkdir', tmpfs+day[:8]]) 
            except:
                print('Cannot create directory,',tmpfs+day[:8])
                continue         
        args=[[cameras[camID], [cameras[cmr] for cmr in groups[camID]], day] for camID in cams]                   
        hs=p.map(height,args)
#         with open(day[:8]+'.pkl','wb') as fhs:
#             pickle.dump(hs,fhs, pickle.HIGHEST_PROTOCOL)
#         print(hs)
#         plt.figure(); plt.plot(np.array(hs[0][0]),np.array(hs[0][1])); plt.plot(np.array(hs[1][0]),np.array(hs[1][1])); plt.show();
    if VIS:            
        p0.join(); 
            



