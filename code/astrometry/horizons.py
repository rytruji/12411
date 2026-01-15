#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.jplhorizons import Horizons

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

def get_ephemeris(ids, utc):
    '''
    Docstring for get_ephemeris

    Given a list of object ids and equinox, queries JPL Horizons to produce corresponding ephemerides using header time. Returns:
        (1) ephem, dictionary containing id: (SkyCoord, ephemeris) pairs for each target id.
    
    :param ids: list, integers or strings corresponding to object ids.
    :param utc: Time, used as observation equinox
    '''
    ephem = {}
    for name in ids:
        obj = Horizons(name, location=810, epochs=[utc.tdb.jd])
        tab = obj.ephemerides()
        ra_deg  = float(tab['RA'][0])   # degrees
        dec_deg = float(tab['DEC'][0])  # degrees
        ephem[name] = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame='icrs')
    return ephem

def distance_km(id, loc, time):
    '''
    Docstring for distance_km
    
    Based on Horizons data, returns distance of object at the given time, in km. Returns:
        (1) dist_km, float of distance to observer in km

    :param id: str, id of object
    :param loc: str, location of observer
    :param time: Time, time at which to get distance
    '''
    jd = time.jd

    obj = Horizons(id=id, location=loc, epochs=jd)
    eph = obj.ephemerides()

    dist_au = eph['delta'][0]
    dist_km = (dist_au * u.au).to(u.km)
    return dist_km