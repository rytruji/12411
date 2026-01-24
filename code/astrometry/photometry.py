#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 22/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

import numpy as np

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

        snr = signal / np.sqrt(signal + ap.area*(np.std(back_values))**2)

        return signal, snr

    def get_mag_zero(self, data, xy_mag_table, idx=1):
        mags = []
        zeros = []
        for source in xy_mag_table:
            inst_flux = self.flux_from_aperture_annulus(data, (source["x"], source["y"]))[0]

            inst_mag = -2.5 * np.log10(inst_flux)
            zero = inst_mag - source["mag"]

            mags.append(inst_mag)
            zeros.append(zero)

        med_zero = np.nanmedian(np.array(zeros))

        fig, ax = plt.subplots(figsize=(8,5))

        ax.scatter([inst_mag - med_zero for inst_mag in mags],xy_mag_table["mag"],color="k")
        plt.savefig(f"./12_411/figures/mag_zero_{idx}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return med_zero, np.nanstd(zeros)