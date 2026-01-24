#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from .observation import Observation
from .tracking import cull_stationary, movement_search, linear_tracker
from .projection import fit_validator
from .plotting import *
from .query import gaia_conical, match_to_catalog
from .statistics import outlier_rejection, eq_residuals_from_line

from astropy.coordinates import SkyCoord
from astropy.io import fits as f
from astropy.time import Time
import astropy.units as u

from photutils.aperture import CircularAperture

import os
import glob
import subprocess

import numpy as np

from tqdm import tqdm
#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

class Astrometry():
    def __init__(self, dir, lindir, image_names=None, verbose=False, midtime_format=None):

        self.datadir = dir
        self.lindir = lindir
        self.plotdir = os.path.join(self.datadir, "plots/").replace("\\", "/")
        self.debugdir = os.path.join(self.datadir, "debug/").replace("\\", "/")

        self.observations = []
        self.verbose = verbose
        self.masked = False

        if midtime_format:
            assert type(midtime_format) == type(_default_midtime)
            self.midtime = midtime_format
        else:
            self.midtime = _default_midtime

        self.extend(image_names)

            
    def extend(self, image_names):

        # test for directory, call extend on contents if so
        if not image_names:
            # collect all files in datadir and recurvisely call extend
            new_files = glob.glob(os.path.join(self.datadir, "*.fits").replace("\\","/"))
            filenames = [os.path.basename(f) for f in new_files]
            assert len(filenames) != 0
            self.extend(filenames)

        # must be a filename, create obs instance using its path
        elif type(image_names) is str:
            full_path = os.path.join(self.datadir, image_names)
            self.observations.append(Observation(full_path, name=image_names, verbose=self.verbose))
        
        # is either invalid or a list. observation() will handle invalid
        else:
            for name in image_names:
                full_path = os.path.join(self.datadir, name)
                self.observations.append(Observation(full_path, name=name, verbose=self.verbose))

        self.total_count = len(self.observations)


    def center_mask_radius(self, rad):

        self.masked = True

        for obs in tqdm(self.observations, total=self.total_count, desc="Masking Observations"):
            ny, nx = obs.data.shape

            xc = nx // 2
            yc = ny // 2

            x_min = xc - rad
            x_max = xc + rad
            y_min = yc - rad
            y_max = yc + rad
            params = np.array([x_max, x_min, y_max, y_min])

            assert np.all(np.where(params >= 0, True, False))

            obs.set_mask(x_min, x_max, y_min, y_max)


    def calibrate_observations(self, biaspath, flatpath, darkpath=None):  

        with f.open(biaspath) as biashdul:
            biasdata = biashdul[0].data
        
        with f.open(flatpath) as flathdul:
            flatdata = flathdul[0].data

        if darkpath:
            with f.open(darkpath) as darkhdul:
                darkdata = darkhdul[0].data
        else:
            darkdata = None

        for obs in tqdm(self.observations, total=self.total_count, desc="Calibrating Observations"):
            obs.calibrate(biasdata, flatdata, darkdata)


    def get_xyls(self, fwhm=10, sigma=10, use_existing=False, make_plots=False):
        for obs in tqdm(self.observations, total=self.total_count, desc="Identifying Sources"):
            # get directory to write to
            out_xyls = os.path.join(self.datadir, f"xyls/{obs.name}.xyls").replace("\\","/")

            # skip if already exists (and not overwriting)
            if os.path.exists(out_xyls) and use_existing:
                obs.set_xyls(out_xyls)
                continue

            obs.sigma = sigma
            obs.fwhm = fwhm
                
            # get sources from data using DAO
            sources = obs.get_dao()

            # turn QTable into an xyls hdu
            xyls = obs.get_sources_xyls(sources)

            if make_plots:
                source_plot(data=obs.data, sources=sources, outdir=self.plotdir, name=f"source_plot_{obs.name}")
            
            # make file and write xyls
            os.makedirs(os.path.dirname(out_xyls), exist_ok=True)
            xyls.writeto(out_xyls, overwrite=True)
            obs.set_xyls(out_xyls)

        
    def get_plate_solve(self, xyls=False, use_existing=False, scale=None, silent=False):
        solveddir = os.path.join(self.lindir, "solved").replace("\\","/")
        os.makedirs(solveddir, exist_ok=True)

        for obs in tqdm(self.observations, total=self.total_count, desc="Requesting plate solutions from local Astrometry.net"):
            # get directory to .corr (that astrometry.net only writes if successful) 
            corr = os.path.join(self.datadir, f"solved/{obs.name}.corr").replace("\\","/")

            # skip if corr exists (and not overwriting)
            if os.path.exists(corr) and use_existing:
                continue

            if xyls:
                platedir = os.path.join(self.lindir, f"xyls/{obs.name}.xyls").replace("\\","/")
            else:
                platedir = os.path.join(self.lindir, obs.name).replace("\\","/")
            
            if scale:
                cmd = f'wsl ~ -e sh -c "solve-field {platedir} --overwrite --dir {solveddir} --no-plots --scale-low {scale-0.05} --scale-high {scale+0.05} --scale-units arcsecperpix"'
            else:
                cmd=  f'wsl ~ -e sh -c "solve-field {platedir} --overwrite --dir {solveddir} --no-plots --scale-units arcsecperpix"'

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except:
                proc.kill()
                stdout, stderr = proc.communicate()
            
            if not silent:
                os.makedirs(self.debugdir, exist_ok=True)

                stdout_path = os.path.join(self.debugdir, f"stdout_{obs.name}.txt").replace("\\","/")
                with open(stdout_path, "w") as txt:
                    txt.write(stdout)

                if not stderr: continue
                stderr_path = os.path.join(self.debugdir, f"stderr_{obs.name}.txt").replace("\\","/")
                with open(stderr_path, "w") as txt:
                    txt.write(stderr)


    def get_solutions(self, xyls=False, fwhm=10, sigma=10, use_existing=False, make_plots=False, scale=None, silent=False, degree=1):

        # catch parameter errors
        if not xyls:
            if self.masked:
                print("xyls=False is incompatible with data that has been masked.\n" +
                    "This is to ensure consistency between data and wcs.\n")
        if degree <= 0:
            raise AttributeError(f"Polynomial fit degree must be greater than or equal to one, but you input {degree}. Please input a higher value.")
        if type(degree) != int:
            raise AttributeError(f"Polynomial fit degree must be an integer, but you input {degree}. Please input an integer.")

        else:
            self.get_xyls(fwhm, sigma, use_existing=use_existing, make_plots=make_plots)

        self.get_plate_solve(xyls, use_existing, scale=scale, silent=silent)

        for obs in tqdm(self.observations, total=self.total_count, desc="Saving wcs and corr files; making converters"):
            try:
                obs.set_wcs(os.path.join(self.datadir, "solved", obs.name + ".wcs").replace("\\","/"))
                obs.set_corr(os.path.join(self.datadir, "solved", obs.name + ".corr").replace("\\","/"))
                obs.get_projection(degree)
            except:
                print(f"Field {obs.name} did not solve successfully and has been skipped.")
                obs.success = False
            

    def validate_fits(self, make_plots=False):
        residuals = []
        for obs in tqdm(self.observations, total=self.total_count, desc="Validating Fits"):
            if obs.success == False:
                continue
            
            residual = fit_validator(obs.projection, obs.corr)

            residuals.append(residual)
        if make_plots:
            residual_plot(residuals=residuals, outdir=self.plotdir)
        
    
    def collect_sources(self, fwhm=3, threshold=3, make_plots=False):
        '''
        Docstring for auto_tracker

        Collects detected sources, as SkyCoords, in all images. Uses image segmentation with source deblending.

        Returns:
            (1) all_sources, list of SkyCoord objects containing all source coordinates in observation order
            (2) all_times, list of Time objects of observation dates
        '''

        all_sources = []
        all_times = []

        for obs in tqdm(self.observations, total=self.total_count, desc="Collecting Sources"):
            if not obs.success:
                continue
            
            # get SourceCatalog and deblended segmentation map of data
            cat, segm_deblend = obs.get_segmentation(fwhm=fwhm, threshold=threshold)
            
            # turn cat into QTable that can be used to get source coordinates
            columns = ["xcentroid", "ycentroid"]
            sources = cat.to_table(columns=columns)

            if make_plots:
                segmentation_plot(obs.data, cat, segm_deblend, self.plotdir, name=f"segmentation_plot_{obs.name}")

            # get centroided x, y coords, turn into equatorial
            x_1 = sources["xcentroid"]
            y_1 = sources["ycentroid"]
            sc = SkyCoord([obs.px_to_eq(x, y) for x, y in zip(x_1, y_1)])    

            all_sources.append(sc)
            all_times.append(self.midtime(obs.header))

        return all_sources, all_times



    def auto_tracker(self, stationary_error, prediction_error, threshold, fwhm, depth, make_plots=False):
        '''
        Docstring for auto_tracker

        Scans observations for sources moving linearly and, if moving targets are identified, gives their equatorial position in each observation.
        
        Returns:
            (if successful) chains, list of moving target positions (SkyCoord) in each observation
            (else) None

        :param stationary_error: Angle, max angular separation between sources considered stationary
        :param prediction_error: Angle, max angular offset between predicted and true position of next source in the chain
        '''
        all_sources, all_times = self.collect_sources(threshold=threshold, fwhm=fwhm, make_plots=make_plots)
        culled_sources = cull_stationary(all_sources, stationary_error)
        print("Attempting Movement Search...")
        chains = movement_search(culled_sources, all_times, prediction_error, depth)
        print(f"Done. The number of identified tracks was {len(chains)}.")
        return(chains)
    

    def _coord_discrim(self, obs, i, jpl_id, coordinate, full_chains, tracker):
        if jpl_id: 
            return obs.get_ephemeris(jpl_id)
        elif coordinate:
            return SkyCoord(coordinate)
        elif full_chains:
            return full_chains[0][i]
        else:
            utc = self.midtime(obs.header)
            return tracker(utc)
        

    def reject_by_line(self, positions, degree=3):
        residuals = eq_residuals_from_line(positions, degree=degree)

        rejection = outlier_rejection(residuals[0]) | outlier_rejection(residuals[1])

        offset = 0
        for i, obs in enumerate(self.observations):
            if not obs.success:
                offset += 1
                continue
            if rejection[i-offset]:
                obs.success = False

        positions = np.array(positions)[~rejection]

        return positions


    def track_objects(self, rad=10, make_plots=False, jpl_id=None, coordinate=None,
                      stationary_error=None, prediction_error=None, threshold=3, fwhm=3, depth=3, rejection=False):
        
        # initialize tracker variables as None
        tracker = None
        full_chains = None

        if not jpl_id and not coordinate:
            full_chains = self.auto_tracker(stationary_error=stationary_error, 
                                            prediction_error=prediction_error, 
                                            threshold=threshold, 
                                            fwhm=fwhm, 
                                            make_plots=make_plots,
                                            depth=depth)
            if not full_chains:
                print("Auto Tracking failed. Switching to manual linear tracking...")
                tracker = linear_tracker(self.datadir, self.observations[0], self.observations[-1])
                
        
        # initialize empty positions
        positions_g, positions_m = [], []

        # temp soln for failed frames
        offset = 0

        for i, obs in tqdm(enumerate(self.observations), total=self.total_count, desc="Tracking"):
            
            # skip frames with a failed astrometric solution
            if not obs.success:
                offset += 1
                continue

            coord = self._coord_discrim(obs=obs, 
                                        i=i-offset, 
                                        jpl_id=jpl_id, 
                                        coordinate=coordinate, 
                                        full_chains=full_chains, 
                                        tracker=tracker)


            # skip frames with a failed track (coord == None)
            if not coord:
                obs.success = False
                continue

            pred_pos = []
            fit_pos = []

            x, y = obs.eq_to_px(coord)
            
            # make aperture at predicted position
            ap = CircularAperture((x,y), rad)

            # get observation (x_img, y_img) outputs from centroids
            g = obs.Gaussean2D_centroid(ap)
            m = obs.Moffat2D_centroid(ap)

            # for plotting: add predicted and dao output to relevant lists
            pred_pos.append((x,y))
            fit_pos.append(m)
            # for plotting: make array of form [(x_1, x_2, x_3), (y_1, y_2, y_3)]
            all_px = np.array([coords for coords in zip(g,m)])

            # add centroid results to relevant positions dicts
            positions_g.append(SkyCoord(obs.px_to_eq(g[0], g[1])))
            positions_m.append(SkyCoord(obs.px_to_eq(m[0], m[1])))

            if make_plots:
                fits_plot(data=obs.data, wcs=obs.wcs, outdir=self.plotdir, name=f"track_results_{obs.name}", pred_xy=pred_pos, fit_xy=fit_pos)
                centroid_test_plot(data=obs.data, ap=ap, centroid_results=all_px, outdir=self.plotdir, name=f"cent_fit_{obs.name}")

        if rejection:
            positions_g = self.reject_by_line(positions_g)
            positions_m = self.reject_by_line(positions_m)

        return positions_g, positions_m
    

    def get_magnitudes(self, positions, transform, make_plots=False):
        offset = 0
    
        mags = []

        print("Requesting Catalog Stars from Gaia...")
        obs = self.observations[len(self.observations) // 2]
        center = SkyCoord(obs.wcs.wcs.crval[0]*u.deg, obs.wcs.wcs.crval[1]*u.deg)
        r = gaia_conical(coordinate=center, transform=transform)
        
        for i, obs in tqdm(enumerate(self.observations), total=len(self.observations), desc="Performing photometry to get magnitudes"):
            if not obs.success:
                offset += 1
                continue

            matches = match_to_catalog(obs, r)

            highest_mag_coord = matches["x"][np.argmin(matches["mag"])], matches["y"][np.argmin(matches["mag"])]

            radii, curve = obs.phot.get_best_radius(obs.data, highest_mag_coord)

            zero, std = obs.phot.get_mag_zero(obs.data, matches, idx=obs.name)

            if make_plots:
                xy_matches = [(x,y) for x,y in zip(matches["x"], matches["y"])]
                fits_plot(obs.data, obs.wcs, self.plotdir, name=f"matches_{obs.name}", fit_xy=xy_matches)

                curve_of_growth(radii, curve, obs.phot.best_rad, self.plotdir, name=f"curve_of_growth_{obs.name}")

            target_position_xy = obs.eq_to_px(positions[i-offset])

            target_flux = obs.phot.flux_from_aperture_annulus(obs.data, target_position_xy)[0]

            inst_mag = -2.5 * np.log10(target_flux)
            stand_mag = inst_mag - zero
            mags.append(stand_mag)

        return np.array(mags)


    def to_mpc(self, positions, packed_desig, filter, obs_code, mags, mp_number="     ", discovery=False, note1=" ", note2=" ", write_out=True):
        mpc = ""
        offset = 0

        for i, obs in tqdm(enumerate(self.observations), total=len(self.observations), desc="Writing observations to 80-column format"):
            if not obs.success:
                offset += 1
                continue
            

            mpc += eighty_column(packed_desig=packed_desig,
                                 utc=self.midtime(obs.header),
                                 coord=positions[i-offset],
                                 mag=mags[i-offset],
                                 band=filter,
                                 obs_code=obs_code,
                                 packed_number=mp_number,
                                 discovery=discovery,
                                 note1=note1,
                                 note2=note2) + "\n"
            
        if write_out:
            outdir = os.path.join(self.datadir, "mpc_out.txt").replace("\\", "/")
            with open(outdir, "w") as txt:
                txt.write(mpc)
        
        return mpc


def _default_midtime(header):
    utc = Time(header.get("DATE-OBS"), scale='utc') 
    offset = (float(header.get("EXPTIME")) / 2) * u.second
    return utc + offset

def eighty_column(packed_desig, # cols 6-12 (7)
                  packed_number, # cols 1-5 (5)
                  discovery, # col 13 (1)
                  note1,  # col 14 (1)
                  note2, # col 15 (1))
                  utc, # cols 16-32 (17)
                  coord, # cols 33-56
                  mag,
                  band, 
                  obs_code):
    
    assert len(packed_number) == 5
    assert len(packed_desig) == 7
    assert type(discovery) == bool
    assert len(note1) == 1 and len(note2) == 1
    assert isinstance(utc, Time)
    assert isinstance(coord, SkyCoord)
    assert len(f"{mag:2.1f}") == 4 and isinstance(mag, float)
    assert len(band) == 1
    assert len(str(obs_code)) == 3 and isinstance(obs_code, int)

    packed_full = packed_number + packed_desig
    packed_full = f"{packed_full:<12}"
    
    if discovery:
        asterisk = "*"
    else:
        asterisk = " "

    jd = utc.utc.jd
    day_frac = jd % 1
    mpc_time = utc.utc.strftime("%Y %m %d.") + f"{day_frac:.6f}"[2:]
    mpc_time = f"{mpc_time:<17}"[:17]

    ra = coord.ra.to_string(unit=u.hour, sep=" ", pad=True, precision=2)
    dec = coord.dec.to_string(unit=u.deg, sep=" ", alwayssign=True, pad=True, precision=1)
    
    ra = f"{ra:<12}"[:12]
    dec = f"{dec:<12}"[:12]
    
    magband = f"{mag:>4.1f} {band:<1}"

    out = (
        f"{packed_full}"
        f"{asterisk}"
        f"{note1}{note2}"
        f"{mpc_time}"
        f"{ra}"
        f"{dec}"    
        f"{' '*9}"     
        f"{magband}" 
        f"{' '*6}" 
        f"{obs_code}"
    )

    print(out)

    assert len(out) == 80, len(out)

    return out