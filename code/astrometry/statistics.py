#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 24/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np
from scipy import special

from astropy.modeling import fitting, models
import astropy.units as u

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

def outlier_rejection(values):
    '''
    Docstring for outlier_rejection

    Determines which points in a the given set are outliers using the Chauvenet criterion. Ignores values that are None.

    Returns:
        (1) Numpy Array, chauvenet mask for outliers, where True is an outlier. None is automatically an outlier.
    
    :param values: Array-like, values to perform outlier rejection with.
    '''
    values = np.array(values)

    std = np.nanstd(values)
    mean = np.nanmean(values)

    t_scores = np.abs(values - mean) / std

    p_scores = special.erfc(t_scores)

    chauvenet_scores = p_scores * len(values)

    return (chauvenet_scores < 0.5) | (np.isnan(values))


def eq_residuals_from_line(positions, degree=1):

    line = models.Polynomial1D(degree=degree)

    fitter = fitting.LinearLSQFitter()

    all_ra = [coord.ra.deg for coord in positions]
    all_dec = [coord.dec.deg for coord in positions]

    ra_fit_line = fitter(line, x=all_dec, y=all_ra)
    dec_fit_line = fitter(line, x=all_ra, y=all_dec)

    ra_residuals = [(coord.ra.deg*u.deg - ra_fit_line(coord.dec.deg)*u.deg).to_value(u.arcsec) for coord in positions]
    dec_residuals = [(coord.dec.deg*u.deg - dec_fit_line(coord.ra.deg)*u.deg).to_value(u.arcsec) for coord in positions]

    return (ra_residuals, dec_residuals)