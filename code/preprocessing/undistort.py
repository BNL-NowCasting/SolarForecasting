import numpy as np
import os, glob
from matplotlib import pyplot as plt
import camera as cam
from PIL import Image

camID='HD2B'

inpath='~/data/images/' 
outpath='~/data/undistorted/'    

camera=cam.camera(camID,max_theta=70)

def preprocess(camera,f):    
    img=cam.image(camera,f);  ###img object contains four data fields: rgb, red, rbr, and cm 
    img.undistort(camera,rgb=True);  ###undistortion
    if img.red is None:
        return
    ims = Image.fromarray(img.rgb); ims.save(outpath+camID+'/'+os.path.basename(f)[:-4]+'.png', "PNG"); 

doy='2018082516[1-2]'
#####get the list of files to be processed
flist = sorted(glob.glob(inpath+camID+'/'+doy[:8]+'/'+'*_'+doy+'*jpg'))


for i, f in enumerate(flist):
    preprocess(camera,f)

