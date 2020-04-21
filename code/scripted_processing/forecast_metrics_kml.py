import pandas as pd

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import dates as md

import configparser
import os, subprocess
from datetime import datetime, timezone, timedelta
from ast import literal_eval as le
import pytz


 
def localToUTCtimestamp(t, local_tz):
    t_local = local_tz.localize(t, is_dst=None)
    t_utc = t_local.astimezone(pytz.utc)
    return t_utc.timestamp()
    
def UTCtimestampTolocal(ts, local_tz):
    t_utc = datetime.fromtimestamp(ts,tz=pytz.timezone("UTC"))
    t_local = t_utc.astimezone(local_tz)
    return t_local
    
def create_kml(minlong, minlat, maxlong, maxlat, overlay, kml_file, name, legend, day):
    content = '''<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
    <name>Forecast Metrics for %s</name>
    <GroundOverlay>
       <name>%s</name>
       <color>7fffffff</color>
       <drawOrder>1</drawOrder>
       <Icon>
          <href>%s</href>
       </Icon>
       <LatLonBox>
          <north>%f</north>
          <south>%f</south>
          <east>%f</east>
          <west>%f</west>
          <rotation>0</rotation>
       </LatLonBox>
    </GroundOverlay>
    <ScreenOverlay>
      <name>Legend</name>
      <Icon>
        <href>%s</href>
      </Icon>
      <overlayXY x="1" y="0" xunits="fraction" yunits="fraction"/>
      <screenXY x="1" y="0.1" xunits="fraction" yunits="fraction"/>
      <rotation>0</rotation>
      <size x="-1" y="-1" xunits="pixels" yunits="pixels"/>
    </ScreenOverlay>
    </Document>
    </kml>''' % (day, name, overlay, maxlat, minlat, maxlong, minlong, legend)
    
    with open(kml_file, 'w') as k_file:
        k_file.write(content)


try:
    try:
        config_path = sys.argv[1]
    except Exception:
        config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)

    feature_path=le(cp["paths"]["feature_path"])

    lead_minutes=le(cp["forecast"]["lead_minutes"])
    days=le(cp["forecast"]["days"])

    #lead_minutes=[1,3,5,10,15,30,45]; 
    #sensors = np.arange(99,100)
    
    try:
        sensors = le(cp["forecast"]["sensors"])
    except Exception:
        GHI_Coor = le(cp["GHI_sensors"]["GHI_Coor"])    #if sensor list isn't provided, forecast for all GHI points
            
    gridspaces=int(cp["forecast"]["gridspaces"])
   
except KeyError as e:
    print("Error loading config: %s" % e)

        
plt.ioff()  #Turn off interactive plotting for running automatically



for day in days:

    print("Calculating metrics for " + day)
      
    f_stats = np.genfromtxt(feature_path+day[:8]+'/forecast_stats.csv',delimiter=',',dtype=np.int16)
    
    scale_max = np.amax(f_stats)
    
    for idx, forecast_int in enumerate(lead_minutes):
        int_stats = f_stats[:,idx]
        if not len(int_stats) == len(GHI_Coor):
            print("Stats array different length than configured GHI coords list: %i, %i" % (len(int_stats), len(GHI_Coor)))
            
        plot_stats = int_stats.reshape(gridspaces,gridspaces)
        
        #Create plot
        plt.imshow(plot_stats, cmap='jet', vmin=0, vmax=scale_max)
        cb = plt.colorbar()
        cb.set_label('Forecast Availability (Points/Day)', rotation=-90, color='k', labelpad=20)
        plt.tight_layout(); 
        ax = plt.gca()
        ax.invert_yaxis()
        plt.savefig(feature_path+day[:8]+'/forecast_stats_'+str(forecast_int)+'.png')
        
        #Create legend
        ax.remove()
        plt.savefig(feature_path+day[:8]+'/stats_legend_'+str(forecast_int)+'.png', transparent=False, bbox_inches='tight')
        
        plt.close()
        
        #Create overlay
        plt.imshow(plot_stats, cmap='jet', vmin=0, vmax=scale_max)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        overlay_file = feature_path+day[:8]+'/stats_overlay_'+str(forecast_int)+'.png'
        plt.savefig(overlay_file, bbox_inches='tight',transparent=True, pad_inches=0)
        plt.close()

        longs = [i[1] for i in GHI_Coor.values()]
        lats = [i[0] for i in GHI_Coor.values()]

        kml_file = feature_path+day[:8]+'/stats_overlay_'+str(forecast_int)+'.kml'
        create_kml(min(longs), min(lats), max(longs), max(lats), overlay='stats_overlay_'+str(forecast_int)+'.png', kml_file=kml_file, name=str(forecast_int)+' Min Forecast Availability', legend='stats_legend_'+str(forecast_int)+'.png', day=day)
    