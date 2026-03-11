from astrometry.astrometry import Astrometry
from astrometry.query import G_filter_transform
from astrometry.header import Header

# from astropy.modeling.fitting import warnings
# warnings.filterwarnings("ignore")

import astropy.units as u

# create Astrometry coordinator object containing all data files
astro = Astrometry("C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/2026AX2", 
                   "/mnt/c/Users/truji/Desktop/MIT_F25/12_411/data/20260115/2026AX2",
                   midtime_format=Header.TCS_midtime # ensure this matches the telescope in use!
                   )

# calibrate all data files using the given masters
astro.calibrate_observations('C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/masters/20260115_masterBIAS_SDSSr.fits',
                             'C:/Users/truji/Desktop/MIT_F25/12_411/data/20260115/masters/20260115_masterFLAT_SDSSr.fits')

# get astrometric solutions for all plates
# debug for solve-field calls written to ./debug/ unless parameter 'silent' is True
astro.get_solutions(xyls=True, 
                    sigma=5, 
                    fwhm=5, 
                    make_plots=True, 
                    use_existing=True, 
                    degree=1)

# (optional) validate plate solutions by plotting source residuals against catalog
# astro.validate_fits(make_plots=True)

# track object. if no coordinate/ephemeris is given/requested, will attempt to autotrack. Otherwise, will ask for start/end pixel locations.
# if manual tracking is needed, provide pixel locations for the .fits files given in ./for_linear_tracking/, not in the original .fits files.
positions_g, positions_m = astro.track_objects(make_plots=False, 
                                               method="Segmentation", # choose from Segmentation or Differential
                                               threshold=3, fwhm=3, # for image segmentation
                                               stationary_error=1*u.arcsec, # eliminates unmoving sources based on this radius
                                               prediction_error=5*u.arcsec, # adds to track with an error tolerance of this radius
                                               depth=9, # max length of chain to predict with (uses sequential pairs)
                                               rejection=True) # n-iteration rejection via the Chauvenet criterion off a 2nd order fit

# uses differential photometry against the Gaia-DR3 catalog to determine estimate magnitudes for the object at every point in the track (positions).
mags = astro.get_magnitudes(positions_m, G_filter_transform.SDSSr, make_plots=False)

astro.to_mpc(positions_g, packed_desig="K26A02X", filter="r", note1="G", note2="C", obs_code=954, mags=mags, write_out=True)