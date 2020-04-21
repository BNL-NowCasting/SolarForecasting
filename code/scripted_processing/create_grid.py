# This script converts a KML polygon into a grid of Lat/Long points based on its min/max extents
# Generates coords.csv and coords.dict (which is in the format used by config.conf
#
# KML filename should be passed as command line argument (i.e. "python3 create_grid.py ForecastArea.kml")
#   or as a list of points in config.conf under ['forecast']['area']
#
# Also requires ['forecast']['gridspaces'] to be defined in config.conf, defines the divisions/side of the area (i.e. gridspaces=10 creates a grid with 100 points)

import configparser
import sys
from ast import literal_eval as le
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def get_polygon(file):
    tree = ET.parse(file) 
    root = tree.getroot()
    ns = root.tag[:-3]
    coords = tree.findall('.//'+ns+'coordinates')

    longs = []
    lats = []
    for i in coords:
        csv_coords = i.text.strip().split(' ')
        for pt in csv_coords:
            csv_pt = pt.split(',')
            longs += [float(csv_pt[0])]
            lats += [float(csv_pt[1])]
    print("Loaded coords from KML:\n\tLongs: %s\n\tLats: %s\n" % (longs, lats))
    return longs, lats

if __name__ == "__main__":

    config_path = "./config.conf"
    cp = configparser.ConfigParser()
    cp.read(config_path)
    gridspaces=cp["forecast"]["gridspaces"]
    
    if len(sys.argv) > 1:
        kml_file = sys.argv[1]
        longs, lats = get_polygon(kml_file)
    else:
        area=le(cp["forecast"]["area"])
        longs = [i[0] for i in area]
        lats = [i[1] for i in area]
    
    print("Creating grid...")
    
    x0 = min(longs)
    x1 = max(longs)
    
    y0 = min(lats)
    y1 = max(lats)
        
    x = np.linspace(x0,x1,gridspaces)
    y = np.linspace(y0,y1,gridspaces)
    xx,yy = np.meshgrid(x,y)
    pts = list(map(list, zip(yy.ravel(), xx.ravel())))  #note, x and y order swapped for compatibility with forecasting dict
    coords = {i:d for (i, d) in enumerate(pts)}
    
    with open("coords.dict",'w') as f:
        #dict
        f.write("#dict for config.conf\nGHI_Coor =\t%s\n\n" % str(coords).replace("],","],\n\t\t"))
    
    with open("coords.csv",'w') as f:
        #CSV
        f.write("Point,Lat,Long\n")
        for i, d in coords.items():
            print(i, d)
            f.write("%i,%f,%f\n" % (i, d[0], d[1]))

    #plt.scatter(xx,yy)
    #plt.show()