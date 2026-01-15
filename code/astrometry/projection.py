#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 15/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
import astropy.units as u

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################


class Projection():
    def __init__(self, corr, ra0, dec0, degree=1):
            
        # set image center
        self.ra0, self.dec0 = ra0, dec0

        # construct field (x, y) and index (ra, dec) positions using corr
        x_n = np.array(self.corr["field_x"])
        y_n = np.array(self.corr["field_y"])

        ra_n  = np.array(self.corr["index_ra"])
        dec_n = np.array(self.corr["index_dec"])

        # project all ra, dec around image center
        xi_n, eta_n = gnomonic_projection(ra_n, dec_n, ra0, dec0)

        # make and fit polynomials
        poly_xi  = models.Polynomial2D(degree=degree)
        poly_eta = models.Polynomial2D(degree=degree)

        poly_x = models.Polynomial2D(degree=degree)
        poly_y = models.Polynomial2D(degree=degree)

        fitter = fitting.LinearLSQFitter()

        self.fit_xi  = fitter(poly_xi, x_n, y_n, xi_n)
        self.fit_eta = fitter(poly_eta, x_n, y_n, eta_n)

        self.fit_x = fitter(poly_x, xi_n, eta_n, x_n)
        self.fit_y = fitter(poly_y, xi_n, eta_n, y_n)


    def px_to_eq(self, x, y):
        # x -= crpix1
        # y -= crpix2
        xi_T  = self.fit_xi(x, y)
        eta_T = self.fit_eta(x, y)

        ra_fit_deg, dec_fit_deg = gnomonic_inverse(xi_T, eta_T, self.ra0, self.dec0)

        sky_fit = SkyCoord(ra_fit_deg * u.deg, dec_fit_deg * u.deg, frame="icrs")

        return sky_fit
    

    def eq_to_px(self, coord):
        xi_T, eta_T = gnomonic_projection(coord.ra.deg, coord.dec.deg, self.ra0, self.dec0)

        x_fit = self.fit_x(xi_T, eta_T) # + crpix1
        y_fit = self.fit_y(xi_T, eta_T) # + crpix2

        return x_fit, y_fit
    
    
#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################


def gnomonic_projection(ra_deg, dec_deg, ra0_deg, dec0_deg):
    '''
    Docstring for gnomonic_projection
    
    Applies a gnomonic projection to given ra, dec values based on the given center ra0, dec0. 
    
    Returns:
        (1) xi, array of projected xi values
        (2) eta, array of projected eta values

    :param ra_deg: array, list of RA values in degrees 
    :param dec_deg: array, list of Dec values in degrees 
    :param ra0_deg: float, center RA in degrees
    :param dec0_deg: float, center Dec in degrees
    '''
    # convert to radians
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0  = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    # calculate denominator
    denom = (np.sin(dec) * np.sin(dec0) +
        np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0))

    # project into xi, eta
    xi  = np.cos(dec) * np.sin(ra - ra0) / denom
    eta = (np.cos(dec0) * np.sin(dec) -
        np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / denom

    return xi, eta



def gnomonic_inverse(xi, eta, ra0_deg, dec0_deg):
    '''
    Docstring for gnomonic_inverse
    
    Inverts a gnomonic projection from projected xi, eta to ra, dec values based on the given center ra0, dec0. 
    
    Returns:
        (1) ra, array of RA values in degrees
        (2) dec, array of Dec values in degrees

    :param xi: array, list of xi values
    :param eta: array, list of eta values 
    :param ra0_deg: float, center RA in degrees
    :param dec0_deg: float, center Dec in degrees
    '''

    # convert center into radians
    ra0  = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    # get denominator
    den = np.sqrt(1.0 + xi**2 + eta**2)

    # calculate dec
    sin_dec = (np.sin(dec0) + eta * np.cos(dec0)) / den

    dec = np.arcsin(sin_dec)

    # calculate ra
    den = np.cos(dec0) - eta * np.sin(dec0)

    delta_ra = np.arctan2(xi, den)
    ra = ra0 + delta_ra

    # get only remainder
    ra = ra % (2*np.pi)

    # convert back to degrees, return
    return np.rad2deg(ra), np.rad2deg(dec)


def fit_validator(projection, corr):
    x_n = np.array(corr["field_x"])
    y_n = np.array(corr["field_y"])

    ra_n  = np.array(corr["index_ra"])
    dec_n = np.array(corr["index_dec"])

    eq_from_index = SkyCoord(ra_n * u.deg, dec_n * u.deg, frame="icrs")
    eq_from_field = SkyCoord([projection.px_to_eq(x, y) for x, y in zip(x_n, y_n)])

    eq_offsets = [index.spherical_offsets_to(field) for (index, field) in zip(eq_from_index, eq_from_field)]
    dra_arcsec  = np.array([off[0].to_value(u.arcsec) for off in eq_offsets], dtype=float)
    ddec_arcsec = np.array([off[1].to_value(u.arcsec) for off in eq_offsets], dtype=float)

    xy_from_index = np.array([projection.eq_to_px(coord) for coord in eq_from_index], dtype=float)
    xy_from_field = np.column_stack([x_n, y_n]).astype(float) 

    xy_offsets = xy_from_index - xy_from_field
    dx_pix = xy_offsets[:, 0]
    dy_pix = xy_offsets[:, 1]

    return dra_arcsec, ddec_arcsec, dx_pix, dy_pix