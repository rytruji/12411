#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from astroquery.jplhorizons import Horizons
from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1

from photutils.psf import SourceGrouper

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

def get_ephemeris(id, loc, utc):
    '''
    Docstring for get_ephemeris

    Given a list of object ids and equinox, queries JPL Horizons to produce corresponding ephemerides using header time. Returns:
        (1) ephem, dictionary containing id: (SkyCoord, ephemeris) pairs for each target id.
    
    :param id: integer or string corresponding to object id.
    :param utc: Time, used as observation equinox
    '''
    obj = Horizons(id, location=loc, epochs=[utc.tdb.jd])
    tab = obj.ephemerides()
    ra_deg  = float(tab['RA'][0])   # degrees
    dec_deg = float(tab['DEC'][0])  # degrees
    return SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame='icrs')


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


def gaia_conical(coordinate, radius=u.Quantity(1, u.deg), transform=None):

    r = Gaia.query_object_async(coordinate=coordinate, radius=radius, columns=["ra","dec","phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag"])

    if transform:
        r["new_band_mag"] = transform(r["phot_g_mean_mag"], r["phot_bp_mean_mag"], r["phot_rp_mean_mag"])

    return r


def match_to_catalog(obs, catalog, radius=u.Quantity(0.2, u.deg)):
    '''
    Docstring for compare_to_catalog

    Queries Gaia DR3 for reference stars in a radius around the center of the given observation.
    Returns:
        (1) QTable, columns "field_x", "field_y", "mag", for all catalog stars in the observation without a neighbor in a 20-pixel radius.
    
    :param obs: Observation, instance for which to query gaia. (MUST CONTAIN XYLS)
    :param transform: G_filter_transform, function with which to transform Gaia "G" filter magnitudes to the specified band.
    :param radius: (optional, default 15 arcseconds) Angle, radius of gaia query around observation center
    '''

    if not obs.corr:
        raise AttributeError(f"Observation {obs.name} has no corr table.")

    obs.corr["index_ra"].unit = u.deg
    obs.corr["index_dec"].unit = u.deg

    im_sc = SkyCoord(obs.corr["index_ra"], obs.corr["index_dec"])
    cat_sc = SkyCoord(catalog["ra"], catalog["dec"])
    idx, d2d, _ = im_sc.match_to_catalog_sky(cat_sc)

    max_sep = 0.5 * u.arcsec

    sep_constraint = d2d < max_sep
    dao_matches = obs.corr[sep_constraint]
    gaia_matches = catalog[idx[sep_constraint]]

    matches_mag = Table()
    matches_mag["x"] = dao_matches["field_x"]
    matches_mag["y"] = dao_matches["field_y"]
    matches_mag["mag"] = gaia_matches["new_band_mag"]
    matches_mag["deltara"] = ((dao_matches["index_ra"] - gaia_matches["ra"])*u.deg).to(u.arcsec)
    matches_mag["deltadec"] = ((dao_matches["index_dec"] - gaia_matches["dec"])*u.deg).to(u.arcsec)

    grouper = SourceGrouper(min_separation=20)
    groups = grouper(dao_matches["field_x"], dao_matches["field_y"])

    group_ids, counts = np.unique(groups, return_counts=True)
    bad_groups = group_ids[counts > 1]
    grouped = np.isin(groups, bad_groups)

    return matches_mag[~grouped]


class G_filter_transform():
    def SDSSr(G_mag, G_bp_mag, G_rp_mag):
        x = G_bp_mag - G_rp_mag
        G_minus_r = -0.12879 + 0.24662*x - 0.027464*(x**2) - 0.049465*(x**3)
        return G_mag - G_minus_r