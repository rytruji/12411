#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 22/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np

from .plotting import phot_stars_mag

from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

from astropy.stats import sigma_clipped_stats

import matplotlib.pyplot as plt

#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

class Photometry():
    def get_best_radius(self, data, position, rad_min=5, rad_max=50):
        radii = np.array([rad for rad in range(rad_min, rad_max)])

        snr = np.array([self.flux_from_aperture_annulus(data, 
                                                   position, 
                                                   aprad=rad)[1]
                    for rad in radii])
        
        self.best_rad = radii[np.argmax(snr)]

        return radii, snr


    def flux_from_aperture_annulus(self, data, position, aprad=None):
        if not aprad and self.best_rad:
            aprad = self.best_rad
        elif not aprad:
            aprad = 10

        anrad_min = aprad + 5
        anrad_max = aprad + 10

        ap = CircularAperture(positions=position, r=aprad)
        an = CircularAnnulus(positions=position, r_in=anrad_min, r_out=anrad_max)

        phot = aperture_photometry(data, ap)
        total_flux = phot['aperture_sum'][0]

        back_mask = an.to_mask()
        back_values = back_mask.get_values(data)
        bkg_mean, _, _ = sigma_clipped_stats(back_values)

        total_back = bkg_mean * ap.area

        signal = total_flux - total_back
        err = np.sqrt(signal) / np.sqrt(ap.area)

        snr = signal / np.sqrt(signal + ap.area*(np.std(back_values))**2)

        return signal, snr, err

    def get_mag_zero(self, data, xy_mag_table, idx=1):
        inst_mags = []
        offsets = []
        errs = []
        for source in xy_mag_table:
            inst_flux, _, err = self.flux_from_aperture_annulus(data, (source["x"], source["y"]))

            inst_mag = -2.5 * np.log10(inst_flux)
            inst_mag_err = (1 / (err * np.log(10)))**2 * err**2
            inst_offset = inst_mag - source["mag"]

            inst_mags.append(inst_mag)
            errs.append(inst_mag_err)
            offsets.append(inst_offset)

        offset = np.nanmedian(np.array(offsets))
        offset_err = np.nanstd(offsets) / np.sqrt(len(offsets))

        phot_stars_mag(mags=inst_mags, errs=errs, offset=offset, offset_err=offset_err, xy_mag_table=xy_mag_table, name=idx)

        return offset, np.nanstd(offsets) / np.sqrt(len(offsets))