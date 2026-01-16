#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 09/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from .observation import Observation
from .tracking import cull_stationary, movement_search, linear_tracker
from .projection import fit_validator
from .plotting import fits_plot, centroid_test_plot, source_plot

from astropy.coordinates import SkyCoord
from astropy.io import fits as f
from astropy.table import Table
from astropy.time import Time

from photutils.aperture import CircularAperture

import os
import glob

import numpy as np

from tqdm import tqdm
#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

class Astrometry():
    def __init__(self, dir, lindir, image_names=None, verbose=False):

        self.datadir = dir
        self.lindir = lindir
        self.plotdir = os.path.join(self.datadir, "plots/").replace("\\", "/")

        self.observations = []

        self.verbose = verbose

        self.masked = False

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
        for obs in tqdm(self.observations, total=self.total_count, desc="Identifying Sources with DAOfind"):
            # get directory to write to
            out_xyls = os.path.join(self.datadir, f"xyls/{obs.name}.xyls").replace("\\","/")

            # skip if already exists (and not overwriting)
            if os.path.exists(out_xyls) and use_existing:
                continue

            obs.sigma = sigma
            obs.fwhm = fwhm

            # get, format sources from obs
            sources = obs.get_dao()
            xyls = obs.get_sources_xyls(sources)

            if make_plots:
                source_plot(data=obs.data, sources=sources, outdir=self.plotdir, name=f"source_plot_{obs.name}")
            
            # make file and write xyls
            os.makedirs(os.path.dirname(out_xyls), exist_ok=True)
            xyls.writeto(out_xyls, overwrite=True)

        
    def get_plate_solve(self, xyls=False, use_existing=False, scale=None):
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
                os.system(f'wsl ~ -e sh -c "solve-field {platedir} --overwrite --dir {solveddir} --no-plots --scale-low {scale-0.05} --scale-high {scale+0.05} --scale-units arcsecperpix"')
            else:
                os.system(f'wsl ~ -e sh -c "solve-field {platedir} --overwrite --dir {solveddir} --no-plots --scale-units arcsecperpix"')
                


    def get_solutions(self, xyls=False, fwhm=10, sigma=10, use_existing=False, make_plots=False, scale=None, degree=1):

        if not xyls and self.masked:
            print("xyls=False is incompatible with data that has been masked.\n" +
                  "This is to ensure consistency between data and wcs.\n")
            exit

        if xyls:
            self.get_xyls(fwhm, sigma, use_existing, make_plots=make_plots)

        self.get_plate_solve(xyls, use_existing, scale=scale)

        for obs in tqdm(self.observations, total=self.total_count, desc="Saving wcs and corr files; making converters"):
            try:
                obs.set_wcs(os.path.join(self.datadir, "solved", obs.name + ".wcs").replace("\\","/"))
                obs.set_corr(os.path.join(self.datadir, "solved", obs.name + ".corr").replace("\\","/"))
                obs.get_projection(degree)
            except:
                print(f"Field {obs.name} did not solve successfully and has been skipped.")
                obs.success = False
            

    def validate_fits(self, make_plots=False):
        for obs in tqdm(self.observations, total=self.total_count, desc="Validating Fits"):
            if obs.success == False:
                continue
            
            dra_arcsec, ddec_arcsec, dx_pix, dy_pix = fit_validator(obs.projection, obs.corr)

            print(dra_arcsec, ddec_arcsec, dx_pix, dy_pix)
        
    
    def collect_sources(self):
        '''
        Docstring for auto_tracker

        Collects detected sources, as SkyCoords, in all images. 

        Returns:
            (1) all_sources, list of SkyCoord objects containing all source coordinates in observation order
            (2) all_times, list of Time objects of observation dates
        '''
        all_sources = []
        all_times = []

        for obs in tqdm(self.observations, total=self.total_count, desc="Collecting Sources"):
            if not obs.success:
                continue
            # open xyls from file
            xyls_dir = os.path.join(self.datadir, f"xyls/{obs.name}.xyls").replace("\\","/")
            xyls = Table.read(xyls_dir)

            # get 
            x_1 = xyls["X"]
            y_1 = xyls["Y"]
            sc = SkyCoord([obs.px_to_eq(x, y) for x, y in zip(x_1, y_1)])    

            all_sources.append(sc)
            all_times.append(Time(obs.header.get("DATE-OBS"), scale='utc'))

        return all_sources, all_times



    def auto_tracker(self, stationary_error, prediction_error):
        '''
        Docstring for auto_tracker

        Scans observations for sources moving linearly and, if moving targets are identified, gives their equatorial position in each observation.
        
        Returns:
            (if successful) chains, list of moving target positions (SkyCoord) in each observation
            (else) None

        :param stationary_error: Angle, max angular separation between sources considered stationary
        :param prediction_error: Angle, max angular offset between predicted and true position of next source in the chain
        '''
        all_sources, all_times = self.collect_sources()
        culled_sources = cull_stationary(all_sources, stationary_error)
        chains = movement_search(culled_sources, all_times, prediction_error)
        return(chains)
    


    def track_objects(self, ids, rad=10, make_plots=False, use_jpl=False, coordinate=None, stationary_error=None, prediction_error=None):
        positions = {id: [] for id in ids}

        if not use_jpl and not coordinate:
            full_chains = self.auto_tracker(stationary_error=stationary_error, prediction_error=prediction_error)
            if not full_chains:
                print("Auto Tracking failed. Switching to manual linear tracking...")
                tracker = linear_tracker(self.datadir, self.observations[0], self.observations[-1])
                

        for i, obs in tqdm(enumerate(self.observations), total=self.total_count, desc="Tracking"):

            if not obs.success: continue

            if use_jpl: 
                ephem = obs.get_ephemeris(ids)
            
            elif coordinate:
                ephem = {id: coordinate for id in ids}

            elif full_chains:
                ephem = {id: full_chains[0][i] for id in ids}

            else:
                utc = Time(obs.header.get("DATE-OBS"), scale='utc') #  + "T" + obs.header.get("EXP-STRT")
                ephem = {id: tracker(utc) for id in ids}
                print(ephem)

            pred_pos = []
            fit_pos = []

            for id, coord in ephem.items():
                x, y = obs.eq_to_px(coord)
                print(x,y)
                ap = CircularAperture((x,y), rad)

                g = obs.Gaussean2D_centroid(ap)
                m = obs.Moffat2D_centroid(ap)
                dao = obs.dao_centroid(ap)

                all_px = np.array([coords for coords in zip(g,m,dao)])

                pred_pos.append((x,y))
                fit_pos.append(dao)

                positions[id].append(SkyCoord([obs.px_to_eq(xy[0], xy[1]) for xy in [g,m,dao]]))

            if make_plots:
                fits_plot(data=obs.data, wcs=obs.wcs, outdir=self.plotdir, name=f"track_results_{obs.name}", pred_xy=pred_pos, fit_xy=fit_pos)
                centroid_test_plot(data=obs.data, ap=ap, centroid_results=all_px, outdir=self.plotdir, name=f"cent_fit_{obs.name}")

        return positions