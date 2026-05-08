import numpy as np

from astrometry.astrometry import Astrometry
from astrometry.query import G_filter_transform

from astropy.modeling.fitting import warnings
import astropy.units as u
warnings.filterwarnings("ignore")

# create Astrometry coordinator object containing all data files
astro = Astrometry(r'C:\Users\wao\Desktop\NEO_pipeline\test_series\20260314\P12mo8c', 
                   r"/mnt/c\Users\wao\Desktop\NEO_pipeline\test_series\20260314\P12mo8c")

# crop and bin, if needed
# astro.center_mask_radius(1000)
astro.bin_observations(4)

# calibrate all data files using the given masters
astro.calibrate_observations(r'C:\Users\wao\Desktop\NEO_pipeline\test_series\calib\mbias_20250808_p2.fit',
                             r'C:\Users\wao\Desktop\NEO_pipeline\test_series\calib\mflat_20260407_p2.fit',
                             binning=4)

# get astrometric solutions for all plates
# debug for solve-field calls written to ./debug/ unless parameter 'silent' is True

astro.get_solutions(xyls=True, sigma=5, fwhm=5, make_plots=True, use_existing=True, degree=1)

# (optional) validate plate solutions by plotting source residuals against catalog
astro.validate_fits(make_plots=True)

# track object. if no coordinate/ephemeris is given/requested, will attempt to autotrack. Otherwise, will ask for start/end pixel locations.
# if manual tracking is needed, provide pixel locations for the .fits files given in ./for_linear_tracking/, not in the original .fits files.
positions_g, positions_m = astro.track_objects(make_plots=True, threshold=7, fwhm=5, stationary_error=1*u.arcsec, prediction_error=3*u.arcsec, depth=4, rejection=True)

mags = astro.get_magnitudes(positions_m, G_filter_transform.SDSSr, make_plots=True)

# astro.to_mpc(positions_g, packed_desig="K08M05B", filter="r", note2="C", obs_code=954, mags=mags, write_out=True)