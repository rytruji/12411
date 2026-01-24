import numpy as np

from astrometry.astrometry import Astrometry, residuals_from_line
import astropy.units as u
from astrometry.plotting import position_plot
from astrometry.query import G_filter_transform
from astrometry.statistics import outlier_rejection

from astropy.modeling.fitting import warnings
warnings.filterwarnings("ignore")

from astropy.time import Time
import astropy.units as u

TCS_midtime = lambda header: Time(header.get("DATE-OBS") + "T" + header.get("EXP-STRT")) + (float(header.get("EXPTIME")) / 2) * u.second

# create Astrometry coordinator object containing all data files
astro = Astrometry("C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/2026AX2", 
                   "/mnt/c/Users/truji/Desktop/MIT_F25/12_411/data/20260115/2026AX2",
                   midtime_format=TCS_midtime
                   )

# calibrate all data files using the given masters
astro.calibrate_observations('C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/masters/20260115_masterBIAS_SDSSr.fits',
                             'C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/masters/20260115_masterFLAT_SDSSr.fits',)

# crop into usable area, if needed
# astro.center_mask_radius(500)

# get astrometric solutions for all plates
# debug for solve-field calls written to ./debug/ unless parameter 'silent' is True
astro.get_solutions(xyls=True, sigma=5, fwhm=5, make_plots=True, use_existing=True, degree=1)

# validate plate solutions by plotting source residuals against catalog
astro.validate_fits(make_plots=True)

# track object. if no coordinate/ephemeris is given/requested, will attempt to autotrack. Otherwise, will ask for start/end pixel locations.
# if manual tracking is needed, provide pixel locations for the .fits files given in ./for_linear_tracking/, not in the original .fits files.
positions_g, positions_m = astro.track_objects(make_plots=False, threshold=3, fwhm=3, stationary_error=1*u.arcsec, prediction_error=5*u.arcsec, depth=9, rejection=True)

position_plot(positions_g, residuals=residuals_from_line(positions_g, degree=3), outdir=astro.plotdir, name="Gaussean_Position_Plot")
position_plot(positions_m, residuals=residuals_from_line(positions_m, degree=3), outdir=astro.plotdir, name="Moffat_Position_Plot")

mags = astro.get_magnitudes(positions_m, G_filter_transform.SDSSr, make_plots=True)

astro.to_mpc(positions_g, packed_desig="K26A02X", filter="r", obs_code=954, mags=mags, write_out=True)