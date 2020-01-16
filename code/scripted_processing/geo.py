import numpy as np
#import warnings

def distance_sphere(lat1, long1, lat2, long2, flag=True):
 
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    if flag:
        degrees_to_radians = np.pi/180.0;
    else:
        degrees_to_radians = 1;    
         
    # theta = 90 - latitude
    theta1 = (90.0 - lat1)*degrees_to_radians
    theta2 = (90.0 - lat2)*degrees_to_radians
         
    # phi = longitude
    phi1 = long1*degrees_to_radians
    phi2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, phi, theta) and (1, phi', theta')
    # cosine( arc length ) = 
    #    sin theta sin theta' cos(phi-phi') + cos theta cos theta'
    # distance = rho * arc length
     
    cos = (np.sin(theta1)*np.sin(theta2)*np.cos(phi1 - phi2) + 
           np.cos(theta1)*np.cos(theta2))
    arc = np.arccos( cos )
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc
    
def latlon_from_vz(vz,vaz,lat0,lon0,ix=1022):
    ##compute the lat and lon given the view 
    ##zenith and view azimuth angle from the DSCOVR platform
    cos_vz=np.cos(vz); sin_vz=np.sin(vz);
    cos_vaz=np.cos(vaz);
    
    A=np.sin(lat0)**2-cos_vz**2;
    B=2*cos_vaz*sin_vz*cos_vz;
    C=np.sin(lat0)**2-(cos_vaz*sin_vz)**2;
    tan_lat=(-B+np.sqrt(B**2-4*A*C))/(2*A);
    
#    A=(cos_vaz*sin_vz)**2+cos_vz**2;
#    B=2*cos_vaz*sin_vz*np.sin(lat0);
#    C=np.sin(lat0)**2-cos_vz**2;
#    flag=abs(A)<3e-2;
#    cos_lat=np.zeros_like(vz);
#    cos_lat[flag]=-C[flag]/B[flag];
#    cos_lat[~flag]=(-B[~flag]+np.sqrt(B[~flag]**2-4*A[~flag]*C[~flag]))/(2*A[~flag]); 
#    cos_lat=(-B+np.sqrt(abs(B**2-4*A*C)))/(2*A);
#    sin_lat=(np.sin(lat0)+cos_vaz*sin_vz*cos_lat)/cos_vz; 
#    lat=np.arctan(sin_lat/cos_lat);
    lat=np.arctan(tan_lat); cos_lat=np.cos(lat);
    cos_lon=(cos_vz/cos_lat-np.sin(lat0)*tan_lat)/np.cos(lat0);   
    lon=np.arccos(cos_lon); lon[:,:ix]=-lon[:,:ix];
    lon+=lon0; lon[lon<-np.pi]+=2*np.pi;

    return lat,lon
    
def latlon_from_xy(x,y,lat0,lon0,R,E2=6.694e-3):   
    d2r=np.pi/180.0;
    lat0*=d2r; lon0*=d2r;
    latout=np.nan+np.zeros_like(x);  lonout=np.nan+np.zeros_like(x);     
    r=np.sqrt(x**2+y**2);        
    mask=r<R;
    x=x[mask]; y=y[mask]; r=r[mask];   
    
    c=np.arcsin(r/R); 
    lat=np.arcsin(np.cos(c)*np.sin(lat0)-y*np.cos(lat0)/R);
#    print(np.sin(lat))
    if E2>0:
        foo=np.sin(lat);
        RR=R/(1+0.5*E2*foo**2+(13.0e-3)*np.abs(foo)**(1/3)*(1-np.abs(foo)));  ##ellipsoid
#        RR=R/(1+0.5*E2*foo**2);
    #    RR=R;  ##spheroid
        c=np.arcsin(r/RR); 
        lat=np.arcsin(np.cos(c)*np.sin(lat0)-y*np.cos(lat0)/RR);
        
        RR=R/(1+0.5*E2*np.sin(lat)**2);  ##ellipsoid
    else:
        RR=R;
    c=np.arcsin(r/RR); sin_c=r/RR; ctan_c=np.cos(c)/sin_c;
    lon=lon0+np.arctan2(x,(r*ctan_c*np.cos(lat0)+y*np.sin(lat0)));
    if isinstance(lon, (list, tuple, np.ndarray)):
        lon[lon<-np.pi]+=(2*np.pi); lon[lon>np.pi]-=(2*np.pi);
    elif lon<-np.pi:
        lon+=(2*np.pi);
    elif lon>np.pi:
        lon-=(2*np.pi);
    latout[mask]=lat; lonout[mask]=lon;             
    return latout/d2r,lonout/d2r

def grid_lonlat(dx=17000,dy=17000,lon0=0,x0=0,y0=0):      
    import pyproj
    from pyresample import utils

    proj_dict = '+proj=sinu +lon_0='+str(lon0)+'+x_0='+str(x0)+'y_0='+str(y0);
    prj = pyproj.Proj(proj_dict);

    x_len, tmp = prj(180,0); x_len-=dx/2; 
    tmp, y_len = prj(0,90); y_len-=dy/2;        

    area_id = 'ease_sh'; name = area_id; proj_id = area_id;
    x_size = np.floor((2*x_len)/dx); y_size = np.floor((2*y_len)/dy);
    area_extent=(-x_len,-y_len,-x_len+x_size*dx,-y_len+y_size*dy);
    area_def = utils.get_area_def(area_id, name, proj_id, proj_dict, x_size, y_size, area_extent);
    lonp,latp=area_def.get_lonlats(); 
    lonp=lonp.astype('float32'); latp=latp.astype('float32');
    for i in range(lonp.shape[0]):
        icenter=int(np.ceil(lonp.shape[1]/2)); tmp=lonp[i,:icenter]; 
        ixb=icenter-np.argmax(tmp[::-1]>10); 
        if ixb>=icenter: ixb=0;
        lonp[i,:ixb]=np.nan;
        latp[i,:ixb]=np.nan;
        icenter=int(np.floor(lonp.shape[1]/2)); tmp=lonp[i,icenter:]; 
        ixb=icenter+np.argmax(tmp<-10); 
        if ixb<=icenter: ixb=lonp.shape[1];
        lonp[i,ixb:]=np.nan;
        latp[i,ixb:]=np.nan; 
        
    return lonp,latp,area_def
    
    
def latlon_from_xy_EPIC (x, y, lat0, lon0, R, E2=6.694e-3, FalseN=0, FalseE=0):

    latout=np.nan+np.zeros_like(x);  lonout=np.nan+np.zeros_like(x);    
    mask=x**2+y**2<(1-E2)*R**2;
    x=x[mask]; y=-y[mask];
#    % Useful constants
    d2r=np.pi/180; lat0*=d2r; lon0*=d2r;
    sin_lat0 = np.sin(lat0);
    cos_lat0 = np.cos(lat0);
    nu0 = R/np.sqrt(1-2*E2*sin_lat0**2 );
    const1 = nu0*E2*cos_lat0*sin_lat0;
#% Seed with center of projection
    lat = lat0+np.zeros_like(x);  
    lon = lon0+np.zeros_like(x);  
#% Start the iteration    
    flag=lat>-1e30; iter=0;
    while flag.any() and iter<40:
#   % Pre-computations
        sin_lat = np.sin(lat[flag]);
        cos_lat = np.cos(lat[flag]);
        sin_lon_minus_lon0 = np.sin(lon[flag]-lon0);
        cos_lon_minus_lon0 = np.cos(lon[flag]-lon0);
#% Radii
        nu = R / np.sqrt(1 - E2*sin_lat**2);
        rho = R * (1 - E2) / (np.sqrt(1 - E2 * sin_lat**2)**3);
#% Test value using forward equations
        xtest = FalseE + nu*cos_lat*sin_lon_minus_lon0;
        ytest = FalseN - nu*sin_lat0*cos_lat*cos_lon_minus_lon0 + const1 + nu*(1-E2)*sin_lat*cos_lat0;
#% Solve the partials
        Xlat = -rho*sin_lat*sin_lon_minus_lon0;  #% dX/dlat
        Xlon = nu*cos_lat*cos_lon_minus_lon0;   #% dX/dlon
        Ylat = rho*(cos_lat*cos_lat0 + sin_lat*sin_lat0*cos_lon_minus_lon0); #% dY/dlat
        Ylon = nu*sin_lat0*cos_lat*sin_lon_minus_lon0; #%  dY/dlon
#% Determinant of the Jacobian
        deter = Xlat*Ylon-Xlon*Ylat;
#% X/Y error this iteration
        ytestnum = y[flag] - ytest;
        xtestnum = x[flag] - xtest;
        testnum = np.sqrt(ytestnum**2+xtestnum**2); #% Radial errorr
#% Adjust the geographicals
#        print(deter)
        lat[flag] = lat[flag] + (Ylon*xtestnum - Xlon*ytestnum)/deter;
        lon[flag] = lon[flag] + (-Ylat*xtestnum + Xlat*ytestnum)/deter;
        foo=flag[flag];  foo[testnum<=0.001]=False; flag[flag]=foo;
        iter+=1;
    lon[flag]=np.nan; lat[flag]=np.nan;    
    lon[lon<-np.pi]+=(2*np.pi); lon[lon>np.pi]-=(2*np.pi);   
    lat[np.abs(lat)>90]=np.nan; lon[np.abs(lon)>180]=np.nan;
    latout[mask]=lat; lonout[mask]=lon;  
    return latout/d2r,lonout/d2r        
    
def xy_from_latlon_EPIC (lat, lon, lat0, lon0, R=1, E2=6.694e-3, FalseN=0, FalseE=0):    
#% A is semi-major axis of the ellipsoid, E2 is eccentricity squared
#% lat0, lon0 define the origin (radians), lat
#rad, lon point to be converted (radians)
#% xgrid, ygrid are Easting and Northing, FalseE, Fal

#% Useful constants
    d2r = np.pi/180;  #% degrees to radians
    lat=lat*d2r; lon=lon*d2r; lat0=lat0*d2r; lon0=lon0*d2r;
    sin_lat0 = np.sin(lat0);
    cos_lat0 = np.cos(lat0);
    nu0 = R/np.sqrt(1-E2*sin_lat0**2);
    const1 = nu0*E2*cos_lat0*sin_lat0;
#% Pre-computations
    sin_lat = np.sin(lat);
    cos_lat = np.cos(lat);
    sin_lon_minus_lon0 = np.sin(lon-lon0);
    cos_lon_minus_lon0 = np.cos(lon-lon0);
#% Radii
    nu = R / np.sqrt(1 - E2*sin_lat**2);  #% Prime vertical
#    rho = R * (1 - E2) / (np.sqrt(1 - E2 * sin_lat**2)**3); #% Meridian
#% Forward equations
    x = FalseE + nu*cos_lat*sin_lon_minus_lon0;
    y = FalseN + nu*sin_lat0*cos_lat*cos_lon_minus_lon0 - const1 -  nu*(1-E2)*sin_lat*cos_lat0;    
    
    return x,y
    
def view_angle_from_latlon_EPIC (lat, lon, lat0, lon0, E2=6.694e-3):    
#% Useful constants
    d2r = np.pi/180;  #% degrees to radians
    lat=lat*d2r; lon=lon*d2r; lat0=lat0*d2r; lon0=lon0*d2r;
    sin_lat0 = np.sin(lat0);
    cos_lat0 = np.cos(lat0);
    nu0 = 1//np.sqrt(1-E2*sin_lat0**2);
    const1 = nu0*E2*cos_lat0*sin_lat0;
#% Pre-computations
    sin_lat = np.sin(lat);
    cos_lat = np.cos(lat);
    sin_lon_minus_lon0 = np.sin(lon-lon0);
    cos_lon_minus_lon0 = np.cos(lon-lon0);
#% Radii
    nu = 1 / np.sqrt(1 - E2*sin_lat**2);  #% Prime vertical
#    rho = R * (1 - E2) / (np.sqrt(1 - E2 * sin_lat**2)**3); #% Meridian
#% Forward equations
    x =  nu*cos_lat*sin_lon_minus_lon0;
    y =  nu*sin_lat0*cos_lat*cos_lon_minus_lon0 - const1 -  nu*(1-E2)*sin_lat*cos_lat0;
    
    rou = np.sqrt(x**2+y**2)
    vz = np.arcsin(rou)/d2r;
    vaz = np.arctan2(x,-y)/d2r;
    vz[rou>=1]=np.nan; vaz[rou>=1]=np.nan;
    
    return vz,vaz    

def argmax_ext(array, exponent):
    array[np.isnan(array)] = -1;
    idx = np.unravel_index(np.argmax(array),array.shape)
    c = array[idx]
    t1 = np.max(array[:,min(array.shape[1]-1,max(0,idx[1]+5))])
    t2 = np.max(array[min(array.shape[0]-1,max(0,idx[0]+5)),:])
    distx = c - t1; disty = c - t2 
    return idx[0],idx[1],round(c,3),round(disty,3),round(distx,3)   
    
#    col = np.arange(array.shape[0])[:, np.newaxis]
#    row = np.arange(array.shape[1])[np.newaxis, :]
#    array[np.isnan(array)] = 0
#    arr2 = array ** exponent
#    arrsum = arr2.sum()
#    if arrsum == 0:
#        # We have to return SOMETHING, so let's go for (0, 0)
#        return (0,0)
#    arrprody = np.sum(arr2 * col) / arrsum
#    arrprodx = np.sum(arr2 * row) / arrsum
#    return (arrprody, arrprodx)
