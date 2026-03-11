import numpy as np
print("Test 1")

try:
    from astrometry.astrometry import Astrometry
    print("Import succeeded")
except Exception as e:
    print(f"Import failed: {e}")

print("Test 2")

from astropy.modeling.fitting import warnings
import astropy.units as u
warnings.filterwarnings("ignore")

print("Test")

# create Astrometry coordinator object containing all data files
astro = Astrometry("C:/Users/truji/Desktop/MIT_F25/12_411/data/20260112/2012PQ28", 
                   "/mnt/c/Users/truji/Desktop/MIT_F25/12_411/data/20260112/2012PQ28")

# calibrate all data files using the given masters
astro.calibrate_observations('C:/Users/truji/Desktop/MIT_F25/12_411/data/20260112/masters/20260112_masterBIAS_1x1.fits',
                             'C:/Users/truji/Desktop/MIT_F25/12_411/data/20260112/masters/20260112_masterFLAT_SDSSr_1x1.fits',)

# crop into usable area, if needed
# astro.center_mask_radius(1000)

# get astrometric solutions for all plates
# debug for solve-field calls written to ./debug/ unless parameter 'silent' is True
astro.get_solutions(xyls=True, sigma=10, fwhm=10, make_plots=True, use_existing=False, degree=1)

# (optional) validate plate solutions by plotting source residuals against catalog
# astro.validate_fits(make_plots=True)

# track object. if no coordinate/ephemeris is given/requested, will attempt to autotrack. Otherwise, will ask for start/end pixel locations.
# if manual tracking is needed, provide pixel locations for the .fits files given in ./for_linear_tracking/, not in the original .fits files.
positions_g, positions_m = astro.track_objects(make_plots=False, threshold=5, fwhm=5, stationary_error=1*u.arcsec, prediction_error=5*u.arcsec, depth=5, rejection=True)

# mags = astro.get_magnitudes(positions_m, G_filter_transform.SDSSr, make_plots=False)

# astro.to_mpc(positions_g, packed_desig="K08M05B", filter="r", note2="C", obs_code=954, mags=mags, write_out=True)