import glob
import os
import datetime as dt


#####windows version
#inpath='E:/data/HDSI2/';
#outpath=inpath;
#start_day='20150922'; end_day='20150925';
#cday=dt.datetime.strptime(start_day,"%Y%m%d");
#eday=dt.datetime.strptime(end_day,"%Y%m%d");
#while cday<=eday:
#    doy=cday.strftime('%Y%m%d');
#    
#    print(doy);    
#    if not os.path.isdir(outpath+doy):
#        os.mkdir(outpath+doy);        
#    yday=cday.strftime('%Y-%m-%d');
#    for f1 in glob.glob(inpath+yday+'*jpg'):   
#        fn=os.path.basename(f1); fn=fn.replace(yday,doy); fn=fn.replace('%3A',''); fn=fn.replace(' ','.'); 
##        print(fn)
#        os.rename(f1,outpath+doy+'/'+fn);
#    if not os.listdir(outpath+doy):
#        os.rmdir(outpath+doy);
#    cday=cday+dt.timedelta(days=1);   
    
###Linux/Unix version  
import subprocess
inpath='/data/SGP_03_sync/HDTSI/';
outpath='~/data/HDTSI/';
start_day='20150921'; end_day='20150930';
cday=dt.datetime.strptime(start_day,"%Y%m%d");
eday=dt.datetime.strptime(end_day,"%Y%m%d");
while cday<=eday:
    doy=cday.strftime('%Y%m%d');    
    print(doy);    
    if not os.path.isdir(outpath+doy):
        os.mkdir(outpath+doy);        
    yday=cday.strftime('%Y-%m-%d');
    for f1 in glob.glob(inpath+yday+'*jpg'):   
        fn=os.path.basename(f1);
        if int(fn[-11:-13])<13.5:
            continue;
        print(fn,'   ',fn[-11:-13]);
        fn=fn.replace(yday,doy); fn=fn.replace(':',''); fn=fn.replace(' ','.'); 
#        subprocess.call(['cp',f1, outpath+doy+'/'+fn]);
#    if not os.listdir(outpath+doy):
#        os.rmdir(outpath+doy);
    cday=cday+dt.timedelta(days=1); 
   

