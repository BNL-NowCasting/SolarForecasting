from datetime import datetime,timedelta
import pysolar.solar as ps
import ephem

gatech = ephem.Observer(); gatech.lat, gatech.lon = '40.88', '-72.87'
sun,moon=ephem.Sun(), ephem.Moon()
sun.compute(gatech)
moon.compute(gatech)
print(ps.get_altitude(40.88,-72.87,datetime.now()))
print("%s %s" % (sun.alt, sun.az))
print("%s %s" % (moon.alt, moon.az))