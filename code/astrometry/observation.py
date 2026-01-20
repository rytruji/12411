#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from .projection import Projection

from astropy.stats import sigma_clipped_stats
from astropy.io import fits as f
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.table import Table

import warnings
from astropy.wcs import FITSFixedWarning
from astropy.units import UnitsWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=UnitsWarning)

import os

import numpy as np

from calibration import calibrate

from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

class Observation():
    def __init__(self, dir, name=None, sigma=10, fwhm=10, verbose=False):
        self.dir = dir
        with f.open(self.dir) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header

        self.name = name

        self.sigma=sigma
        self.fwhm=fwhm
        self.verbose=verbose

        self.success = True
    

    def set_data(self, data):
        '''
        Docstring for set_data
        
        Sets self.data to the given data. Saves old data to self.old_data. Returns:
            (1) None
        '''
        self._clipped_cache = None
        self.data = data
    

    def set_wcs(self, wcs):
        '''
        Sets self.wcs to the given WCS. Returns:
            (1) None
        '''
        with f.open(wcs) as hdul:
            self.wcs = WCS(hdul[0].header)


    def set_corr(self, corr):
        '''
        Sets self.corr to the given corr table. Returns:
            (1) None
        '''
        self.corr = Table.read(corr)


    def set_mask(self, x_min, x_max, y_min, y_max):
        self.set_data(self.data[y_min:y_max, x_min:x_max])


    def save_to(self, outdir):
        out = f.PrimaryHDU(self.data)
        os.makedirs(os.path.dirname(outdir), exist_ok=True)

        out.writeto(outdir, overwrite=True)


    def calibrate(self, biasdata, flatdata, darkdata=None):
        '''
        Docstring for calibrate

        Calubrates the given data using master calibration files. Dark is optional, will use simplified calibration

        result = (light - bias) / flat

        if no master dark is given. Returns:
            (1) None
        '''
        calibrated_data = calibrate(self.data, biasdata, flatdata, darkdata)
        self.set_data(calibrated_data)
    

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
    

    def _bg_stats(self):
        if getattr(self, "_clipped_cache", None) is None:
            self._clipped_cache = sigma_clipped_stats(self.data, sigma=3.0)
        return self._clipped_cache


    def get_dao(self, mask_bounds=None, make_plot=False):
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
        mean, median, std = self._bg_stats()
        if self.verbose: print(f"Observation statistics:\nMean = {mean}\nMedian = {median}\nStd = {std}\n\nRunning daofind...")

        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=self.sigma*std, exclude_border=True)

        # create mask. Shape is mask[y_min:y_max, x_min:x_max]
        if mask_bounds:
            mask = np.zeros(self.data.shape, dtype=bool)
            # get bounds from param
            [(x_min, x_max), (y_min, y_max)] = mask_bounds
            mask[y_min:y_max, x_min:x_max] = True
        else: mask = None

        self.sources = daofind(self.data, mask=mask)

        if self.verbose and len(self.sources) == 0: print("daofind found no sources. Maybe decrease fwhm or sigma?")
        elif self.verbose: print(f"daofind returned {len(self.sources)} sources")
        # sort by source magnitude since most of this is probably noise or very small
        self.sources.sort("mag")
        return self.sources

    def get_segmentation(self):
        bkg_estimator = MedianBackground()
        bkg = Background2D(self.data, 50, filter_size=(3, 3), bkg_estimator=bkg_estimator)
        back2d_data = self.data - bkg.background

        threshold = self.sigma * bkg.background_rms

        kernel = make_2dgaussian_kernel(self.fwhm, (2 * self.fwhm) - 1)
        convolved_data = convolve(back2d_data, kernel)

        segment_map = detect_sources(convolved_data, threshold, npixels=10)

        segm_deblend = deblend_sources(convolved_data, segment_map,
                               npixels=10, nlevels=32, contrast=0.001,
                               progress_bar=False)
        
        cat = SourceCatalog(self.data, segm_deblend, convolved_data=convolved_data)

        return cat, segm_deblend




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

        ny, nx = self.data.shape
        hdu.header["IMAGEW"] = nx
        hdu.header["IMAGEH"] = ny

        self.xyls = hdu
        return self.xyls
    
    
    def get_projection(self, degree):
        ra0, dec0 = self.wcs.wcs.crval
        self.projection = Projection(self.corr, ra0, dec0, degree=degree)
        self.eq_to_px = self.projection.eq_to_px
        self.px_to_eq = self.projection.px_to_eq


    def Moffat2D_centroid(self, ap):
        mask = ap.to_mask(method="center")
        values = mask.cutout(self.data, fill_value=np.nan)

        ny, nx = values.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        amp0 = np.nanmax(values)
        x0 = nx / 2.0
        y0 = ny / 2.0
        gamma0 = max(1.0, ap.r / 2.0)

        m_init = models.Moffat2D(
            amplitude=amp0,
            x_0 = x0,
            y_0 = y0,
            gamma = gamma0
        )

        fitter = fitting.LevMarLSQFitter()
        m_fit = fitter(m_init, xx, yy, values)

        xc = float(m_fit.x_0.value)
        yc = float(m_fit.y_0.value)

        y0, x0 = mask.bbox.iymin, mask.bbox.ixmin

        xc += x0
        yc += y0

        params = {name: getattr(m_fit, name).value for name in m_fit.param_names}

        return (xc, yc)


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

        xc += x0
        yc += y0

        params = {name: getattr(g_fit, name).value for name in g_fit.param_names}

        return (xc, yc)


    def dao_centroid(self, ap):
        mask = ap.to_mask(method="center")
        values = mask.cutout(self.data, fill_value=np.nan)

        mean, median, std = sigma_clipped_stats(values, sigma=3.0)

        dao = DAOStarFinder(fwhm=self.fwhm, threshold=self.sigma*std)
        sources = dao(values)

        y0, x0 = mask.bbox.iymin, mask.bbox.ixmin

        sources.sort("mag")
        return sources["xcentroid"][0] + x0, sources["ycentroid"][0] + y0