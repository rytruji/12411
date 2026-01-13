'''
Created 09/01/2026 by truji@mit.edu

a collection of functions for detecting background stars and performing plate fits
'''

######################################################################################
#------------------------------------------------------------------------------------#
######################################################################################

from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.io import fits as f
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.table import Table
import astropy.units as u

from astroquery.jplhorizons import Horizons

from photutils.centroids import centroid_2dg, centroid_com

from copy import deepcopy

import os
import glob

import numpy as np

from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

######################################################################################
#------------------------------------------------------------------------------------#
######################################################################################

class observation():
    def __init__(self, dir, sigma=10, fwhm=10, verbose=False):
        self.dir = dir
        self.hdul = f.open(self.dir)
        self.data = self.hdul[0].data
        self.header = self.hdul[0].header

        self.sigma=sigma
        self.fwhm=fwhm
        self.verbose=verbose

        self.success = True

    def get_data(self):
        '''
        Docstring for get_data
        
        Returns:
            (1) deepcopy of data
        '''
        return deepcopy(self.data)
    
    def set_wcs(self, wcs):
        '''
        Sets self.wcs to the given WCS.
        '''
        wcs = f.open(wcs)
        self.wcs = WCS(wcs[0])

    def set_corr(self, corr):
        '''
        Sets self.corr to the given .corr.
        '''
        self.corr = Table.read(corr)
    
    def extract_from_aperture(self, coords, rad):
        '''
        Docstring for extract_from_aperture

        Creates a CircularAperture object at coords. Returns:
            (1) newdata, a 1-d array of deepcopied contained values
            (2) np.sum(newdata), integer total signal within the aperture
        
        :param data: FITS image data of observation
        :param coords: (x,y) integer pixel pair for center of CircularAperture
        :param rad: integer, radius of CircularAperture
        '''
        ap = CircularAperture(coords, rad)
        mask = ap.to_mask()
        new_data = mask.get_values(self.data)
        signal = np.sum(new_data)
        return new_data, signal

    def get_dao(self, mask_bounds=None):
        '''
        Docstring for get_dao_mask

        Applies the daofind function to data using the given sigma and fwhm. Sources within the mask will not be returned. Border excluded. Returns:
            (1) sources, QTable output from daofind (or None)
        
        :param data: FITS image data of observation
        :param mask_bounds: (optional, default None) List of 2-d limits for rectangular mask in form [(x_min,x_max), (y_min,y_max)]
        :param sigma: (optional, default 3) float, multiplier for observation sigma to be passed to daofind
        :param fwhm: (optional, default 3) float, fwhm to be passed to daofind
        :param verbose: (optional, default True) bool, Prints progress and statistics.
        '''

        # calculate sigma_clipped_stats to get rough background stats with significant pixels clipped
        if self.verbose: print("\n\nCalculating obervation statistics...")
        mean, median, std = sigma_clipped_stats(self.data, sigma=3.0)
        if self.verbose: print(f"Observation statistics:\nMean = {mean}\nMedian = {median}\nStd = {std}\n\nRunning daofind...")


        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=self.sigma*std, exclude_border=True)

        # create mask. Shape is mask[y_min:y_max, x_min:x_max]
        if mask_bounds:
            mask = np.zeros(self.data.shape, dtype=bool)
            # get bounds from param
            [(x_min, x_max), (y_min, y_max)] = mask_bounds
            mask[y_min:y_max, x_min:x_max] = True
        else: mask = None

        sources = daofind(self.data, mask=mask)
        if self.verbose and len(sources) == 0: print("daofind found no sources. Maybe decrease fwhm or sigma?")
        elif self.verbose: print(f"daofind returned {len(sources)} sources")
        # sort by source magnitude since most of this is probably noise or very small
        sources.sort("mag")
        # return top 50 sources to not overwhelm astrometry.net
        return sources[:50]

    def get_sources_xyls(self, sources, name=None, outdir=None):
        '''
        Docstring for write_sources_xyls

        Writes a FITS table with extension .xyls with columns (X, Y, FLUX, FWHM). 
        This table can be passed to Astrometry.net. By default, overwrites duplicates. Returns:
            None
        
        :param data: FITS data of observation
        :param sources: QTable, containing sources detected in observation. Must contain columns "xcentroid", "ycentroid", "flux".
        :param name: (optional, default "sources") str, name of file.
        :param outdir: (optional, default None) str, directory to write to. If None, writes to current directory.
        '''
        # write position and flux columns
        x_1 = (sources["xcentroid"].astype(float)).astype(np.float32)
        y_1 = (sources["ycentroid"].astype(float)).astype(np.float32)
        flux = sources["flux"].astype(np.float32)

        cols = [
            f.Column(name="X",     format="E", array=x_1),
            f.Column(name="Y",     format="E", array=y_1),
            f.Column(name="FLUX",  format="E", array=flux),
        ]
        hdu = f.BinTableHDU.from_columns(cols)
        hdu.name = "STARS"

        # make FITS header for Table. Must contain image width and height for Astrometry.net
        hdu.header["IMAGEW"], hdu.header["IMAGEH"] = self.data.shape

        self.xyls = hdu
        return self.xyls

    def get_ephemeris(self, ids):
        '''
        Docstring for get_ephemeris

        Given a list of object ids, queries JPL Horizons to produce corresponding ephemerides using header time. Returns:
            (1) ephem, dictionary containing id: (SkyCoord, ephemeris) pairs for each target id.
        
        :param ids: list, integers or strings corresponding to object ids.
        :param utc: (optional, default None) Time; if given, overrides header and is used as obstime instead.
        '''
        utc = Time(self.header.get("DATE-OBS"), scale='utc')
        ephem = {}
        for name in ids:
            obj = Horizons(name, location=810, epochs=[utc.tdb.jd])
            tab = obj.ephemerides()
            ra_deg  = float(tab['RA'][0])   # degrees
            dec_deg = float(tab['DEC'][0])  # degrees
            ephem[name] = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame='icrs')
        return ephem
    
    def fit_poly(self, degree=1):
        
        # construct x, y, ra, dec positions for all corr stars
        x_n = np.array(self.corr["field_x"])
        y_n = np.array(self.corr["field_y"])

        ra_n  = np.array(self.corr["index_ra"])
        dec_n = np.array(self.corr["index_dec"])

        # get image center
        ra0 = self.wcs.wcs.crval[0]
        dec0 = self.wcs.wcs.crval[1]

        # project all ra, dec around image center
        xi_n, eta_n = gnomonic_projection(ra_n, dec_n, ra0, dec0)

        # make and fit polynomials
        poly_xi  = models.Polynomial2D(degree=degree)
        poly_eta = models.Polynomial2D(degree=degree)

        poly_x = models.Polynomial2D(degree=degree)
        poly_y = models.Polynomial2D(degree=degree)

        fitter = fitting.LinearLSQFitter()

        fit_xi  = fitter(poly_xi, x_n, y_n, xi_n)
        fit_eta = fitter(poly_eta, x_n, y_n, eta_n)

        fit_x = fitter(poly_x, xi_n, eta_n, x_n)
        fit_y = fitter(poly_y, xi_n, eta_n, y_n)

        def converter(x, y):
            xi_T  = fit_xi(x, y)
            eta_T = fit_eta(x, y)

            ra_fit_deg, dec_fit_deg = gnomonic_inverse(xi_T, eta_T, ra0, dec0)

            sky_fit = SkyCoord(ra_fit_deg * u.deg, dec_fit_deg * u.deg, frame="icrs")

            return sky_fit
        
        def anticonverter(coord):
            xi_T, eta_T = gnomonic_projection(coord.ra.to(u.deg), coord.dec.to(u.deg), ra0, dec0)

            x_fit = fit_x(xi_T, eta_T)
            y_fit = fit_y(xi_T, eta_T)

            return (x_fit, y_fit)

        self.converter = converter
        self.anticonverter = anticonverter

    def Gaussean2D_centroid(self, ap):
        mask = ap.to_mask(method="center")
        values = mask.cutout(self.data, fill_value=np.nan)

        ny, nx = values.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        amp0 = np.nanmax(values)
        x0 = nx / 2.0
        y0 = ny / 2.0
        sigma0 = max(1.0, ap.r / 2.0)

        # beginning assumption: centered in aperture, sigma1 is half aperture radius
        g_init = models.Gaussian2D(
            amplitude=amp0,
            x_mean=x0,
            y_mean=y0,
            x_stddev=sigma0,
            y_stddev=sigma0,
            theta=0.0,
        )

        fitter = fitting.LevMarLSQFitter()
        g_fit = fitter(g_init, xx, yy, values)

        xc = float(g_fit.x_mean.value)
        yc = float(g_fit.y_mean.value)

        y0, x0 = mask.bbox.iymin, mask.bbox.ixmin

        # x_img = x0 + xc
        # y_img = y0 + yc

        # return (x_img, y_img, params)
        return (xc, yc)

    def c2dg(self, ap):
        mask = ap.to_mask(method="center")
        values = mask.cutout(self.data, fill_value=np.nan)

        y0, x0 = mask.bbox.iymin, mask.bbox.ixmin
        ny, nx = values.shape

        (xc, yc) = centroid_2dg(values)

        return (xc, yc)
    
    def centroid_test_plot(self, ap, dir, name="test_plot"):
        print("plotting now!!!!!!!!!!!")
        fig = plt.figure(figsize=(9, 5.0))
        ax = fig.add_subplot()
        
        mask = ap.to_mask(method="center")
        values = mask.cutout(self.data, fill_value=np.nan)

        im = ax.imshow(
            values,
            cmap="gray_r",
            norm=LogNorm(vmin=1, vmax=np.nanmax(self.data))
        )

        x_me, y_me = self.Gaussean2D_centroid(ap)

        x_phot, y_phot = self.c2dg(ap)
        print(x_me, y_me, "\n", x_phot, y_phot)

        dao = DAOStarFinder(fwhm=10, threshold=10)
        sources = dao(values)

        ax.scatter((x_me), (y_me), marker="+", c="r")
        ax.scatter((x_phot), (y_phot), marker="+", c="k")
        ax.scatter(sources["xcentroid"], sources["ycentroid"], marker="x", c="b")

        plotdir = os.path.join(dir, "plots").replace("\\","/")
        os.makedirs(plotdir, exist_ok=True)

        plt.savefig(os.path.join(plotdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")


    
    def fits_plot(self, dir, name="fits_plot", pred_aps=None, fit_aps=None):

        fig = plt.figure(figsize=(9, 5.0))
        ax = fig.add_subplot(projection=self.wcs)

        im = ax.imshow(
            self.data,
            cmap="gray_r",
            norm=LogNorm(vmin=1, vmax=np.nanmax(self.data))
        )
    
        ra  = ax.coords[0]
        dec = ax.coords[1]

        ra.set_major_formatter('hh:mm:ss')
        dec.set_major_formatter('dd:mm:ss')

        ra.display_minor_ticks(True)
        dec.display_minor_ticks(True)

        ra.set_minor_frequency(4)
        dec.set_minor_frequency(4)

        ra.set_ticklabel(exclude_overlapping=False, simplify=False)
        dec.set_ticklabel(exclude_overlapping=False, simplify=False)

        ra.set_axislabel("RA (HH MM SS)")
        dec.set_axislabel("Dec (DD MM SS)")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Counts")

        handles = []

        if fit_aps:
            apertures_f = CircularAperture(fit_aps, 15)
            apertures_f.plot(ax=ax, color='r')
            handles.append(Line2D([0], [0], color='r', label='Fit Position'))

        if pred_aps:
            apertures_p = CircularAperture(pred_aps, 30)
            apertures_p.plot(ax=ax, color='k', ls='--')
            handles.append(Line2D([0], [0], color='k', ls='--', label='Predicted Position'))

        ax.legend(handles=handles)

        plotdir = os.path.join(dir, "plots").replace("\\","/")
        os.makedirs(plotdir, exist_ok=True)

        plt.savefig(os.path.join(plotdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")





class astrometry():
    def __init__(self, dir, lindir, image_names=None):

        basedir = os.path.abspath(".")
        self.datadir = dir
        self.lindir = lindir

        self.observations = {}

        self.extend(image_names)
            
    def extend(self, image_names):

        # test for directory, call extend on contents if so
        if not image_names:
            # collect all files in datadir and recurvisely call extend
            new_files = glob.glob(os.path.join(self.datadir, "*.fits").replace("\\","/"))
            filenames = [os.path.basename(f) for f in new_files]
            self.extend(filenames)

        # must be a filename, create obs instance using its path
        elif type(image_names) is str:
            full_path = os.path.join(self.datadir, image_names)
            self.observations[image_names] = observation(full_path)
        
        # is either invalid or a list. observation() will handle invalid
        else:
            for name in image_names:
                full_path = os.path.join(self.datadir, name)
                self.observations[name] = observation(full_path)

    def get_xyls(self, optimize=False):
        for name, obs in tqdm(self.observations.items(), total=len(self.observations), desc="Identifying Sources"):
            sources = obs.get_dao()
            xyls = obs.get_sources_xyls(sources)
            
            out_xyls = os.path.join(self.datadir, f"xyls/{name}.xyls").replace("\\","/")
            os.makedirs(os.path.dirname(out_xyls), exist_ok=True)
            xyls.writeto(out_xyls, overwrite=True)

    def get_solutions(self, xyls=False):
        if xyls:
            self.get_xyls()
            platedir = os.path.join(self.lindir, "xyls/*.xyls").replace("\\","/")
        else:
            platedir = os.path.join(self.lindir, "*.fits").replace("\\","/")

        solveddir = os.path.join(self.lindir, "solved").replace("\\","/")
        os.makedirs(solveddir, exist_ok=True)

        os.system(f'wsl ~ -e sh -c "solve-field {platedir} --overwrite --dir {solveddir} --no-plots --scale-units arcsecperpix"')

        for name, obs in self.observations.items():
            try:
                obs.set_wcs(os.path.join(self.datadir, "solved", name + ".wcs").replace("\\","/"))
                obs.set_corr(os.path.join(self.datadir, "solved", name + ".corr").replace("\\","/"))
            except:
                print("Oh no!")
                obs.success = False

    def make_converters(self, degree=1):
        for obs in self.observations.values():
            if obs.success == False:
                continue
            obs.fit_poly(degree)

    def track_objects(self, ids, rad=15, make_plots=False):
        positions = {id: [] for id in ids}
        for name, obs in tqdm(self.observations.items(), total=len(self.observations), desc="Tracking"):
            if not obs.success: continue
            ephem = obs.get_ephemeris(ids)

            pred_pos = []
            fit_pos = []

            for id, coord in ephem.items():
                x, y = obs.anticonverter(coord)
                ap = CircularAperture((x,y), rad)
                # (x_img, y_img, error) = obs.Gaussean2D_centroid(ap)
                (x_img, y_img) = obs.c2dg(ap)

                positions[id].append((x_img, y_img))

                pred_pos.append((x,y))
                fit_pos.append((x_img,y_img))

            if make_plots:
                obs.fits_plot(self.datadir, f"track_results_{name}", pred_aps=pred_pos, fit_aps=fit_pos)
                print("plot requesting")
                obs.centroid_test_plot(ap, self.datadir, f"cent_fit_{name}")
                print("plot made?")

        return positions





def gnomonic_projection(ra_deg, dec_deg, ra0_deg, dec0_deg):
    '''
    Docstring for gnomonic_projection
    
    Applies a gnomonic projection to given ra, dec values based on the given center ra0, dec0. Returns
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
    
    Inverts a gnomonic projection from projected xi, eta to ra, dec values based on the given center ra0, dec0. Returns
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
    num = xi
    den = np.cos(dec0) - eta * np.sin(dec0)

    delta_ra = np.arctan(num, den)
    ra = ra0 + delta_ra

    # get only remainder
    ra = ra % (2*np.pi)

    # convert back to degrees, return
    return np.rad2deg(ra), np.rad2deg(dec)