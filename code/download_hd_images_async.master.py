import os
import glob
from datetime import datetime
import urllib.request as req
import time
import pysolar.solar as ps
import socket
import multiprocessing
####HD815_2 became HD815_W after May, 2018. It was relocated.
#ips = {"HD815_1": "x.x.x.2", "HD815_W": "x.x.x.29", "HD17": "x.x.x.117", 
#       "HD19": "x.x.x.119", "HD20": "x.x.x.120", "HD490": "x.x.x.27", 
#       "HD05":"x.x.x.38:2048","HD06":"x.x.x.218:2048", "HD01":"x.x.x.202:2048",
#       "HD02":"x.x.x.142:2048","HD03":"x.x.x.78:2048", "HD04":"x.x.x.182:2048" }   
ips = {"HD2C": "x.x.x.2", "HD2B": "x.x.x.29", "HD1B": "x.x.x.117", 
       "HD1A": "x.x.x.119", "HD1C": "x.x.x.120", "HD2A": "x.x.x.27", 
       "HD4A":"x.x.x.38:2048","HD4B":"x.x.x.218:2048", "HD5A":"x.x.x.202:2048",
       "HD5B":"x.x.x.142:2048","HD3A":"x.x.x.78:2048", "HD3B":"x.x.x.182:2048" }   
imgurl = "/cgi-bin/viewer/video.jpg?quality=5&streamid=0"
cachepath = "~/data/cache/"
latest = "~/data/latest/"
outpath = "~/data/images/"
organize_interval = 86400 ####organize files once per day
interval = 30  ####save one image every 30 sec in the day time
interval_night = 300  ####save one image every 300 sec during night
lat, lon=40.87,-72.87

socket.setdefaulttimeout(2);

from threading import Event, Thread

def call_repeatedly(intv, func, *args):
    stopped = Event()
    def loop():
        while not stopped.wait(intv): # the first call is in `intv` secs
            func(*args)
    Thread(target=loop).start()    
    return stopped.set

def organize_files(cams):
    for camera in cams:
        fns=glob.glob(cachepath+camera+'/*jpg')
        print(camera,len(fns))
        for fn in fns:
            doy=fn[-18:-10]
            dest=outpath+camera+'/'+doy
            if not os.path.isdir(dest):
                os.makedirs(dest)
                os.chmod(dest,0o755)
            os.rename(fn,dest+'/'+camera+'_'+fn[-18:])

def makeRequest(cam):
    starttime = datetime.utcnow()
    timestamp=starttime.strftime("%Y%m%d%H%M%S")
    proxy = req.ProxyHandler({})
    opener = req.build_opener(proxy)
    req.install_opener(opener)
    try:
        fn=cachepath+cam+"/"+cam+"_"+timestamp+".jpg"
        req.urlretrieve("http://"+ips[cam]+imgurl, fn)
        os.chmod(fn,0o755); ####set the permission
    except Exception as e: 
        print(e)
        return
    try:
        fn_latest=latest+cam+'_latest.jpg'
        os.system('cp '+fn+' '+fn_latest);
        os.chmod(fn_latest,0o755); ####set the permission
    except Exception as e: 
        print(e)
        return

if __name__ == "__main__":  
    organize_event = call_repeatedly(organize_interval, organize_files, ips)

    p = multiprocessing.Pool(len(ips))
    while (True):
        day_flag = ps.get_altitude(lat,lon,datetime.now())>5
        intv=interval if day_flag else interval_night
        saveimage_event = call_repeatedly(intv, p.map_async, makeRequest, ips)
        
        if day_flag:
            while ps.get_altitude(lat,lon,datetime.now())>5:  
                time.sleep(180)
        else:
            while ps.get_altitude(lat,lon,datetime.now())<=5:  
                time.sleep(600)
        saveimage_event()
            
