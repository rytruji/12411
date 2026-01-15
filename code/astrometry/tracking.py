#########################################################################################################
#-------------------------------------------------------------------------------------------------------#
#------------------------------#   created 15/01/2026 by truji@mit.edu   #------------------------------#
#-------------------------------------------------------------------------------------------------------#
#########################################################################################################

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

import re
from tqdm import tqdm

import numpy as np

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

def get_user_xy(prompt):
    pattern = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)")

    text = input(prompt)
    match = pattern.search(text)

    if match:
        x, y = map(float, match.groups())
    else:
        raise ValueError("Please input a valid (x, y) pair.")

    return x, y

def linear_tracker(obs_init, obs_end):

    x_init, y_init = get_user_xy(f"Please input the (x,y) coordinates of the object in {obs_init.name}: ")
    x_end, y_end = get_user_xy(f"Please input the (x,y) coordinates of the object in {obs_end.name}: ")

    coord_init = obs_init.converter(x_init, y_init)
    coord_end = obs_end.converter(x_end, y_end)

    utc_init = Time(obs_init.header.get("DATE-OBS"), scale='utc')
    utc_end = Time(obs_end.header.get("DATE-OBS"), scale='utc')

    delta_T = (utc_end - utc_init).to(u.second)

    dra_dt = (coord_end.ra.deg - coord_init.ra.deg) / delta_T
    ddec_dt = (coord_end.dec.deg - coord_init.dec.deg) / delta_T

    def tracker(utc_obs):
        delta_T_obs = utc_obs - utc_init

        ra_obs = (delta_T_obs * dra_dt) + coord_init.ra.deg
        dec_obs = (delta_T_obs * ddec_dt) + coord_init.dec.deg

        return SkyCoord(ra_obs * u.deg, dec_obs * u.deg, frame="icrs")

    return tracker

def cull_stationary(all_sources, tol):
    for i in tqdm(range(len(all_sources)), desc="Culling Stationary Sources"):
        ref = all_sources[i]

        ref_is_static = np.zeros(len(ref), dtype=bool)

        for k in range(len(all_sources)):
            if k == i: continue

            sc = all_sources[k]

            idx_ref, idx_sc, _, _ = sc.search_around_sky(ref, tol)

            ref_is_static[idx_ref] = True

            keep_sc = np.ones(len(sc), dtype=bool)
            keep_sc[idx_sc] = False
            all_sources[k] = sc[keep_sc]

        all_sources[i] = ref[~ref_is_static]

    return all_sources

def validity_check(chain, tol, all_times):
    if len(chain) <= 2:
        # is a source or pair. Always return True
        return True

    # define list indices for obs n-2, n-1, and n
    nm2 = len(chain) - 3
    nm1 = len(chain) - 2
    n = len(chain) - 1

    # get d/dt from n-2 to n-1
    delta_T21 = (all_times[nm1] - all_times[nm2]).to(u.second)

    dra12, ddec12 = (chain[nm2].spherical_offsets_to(chain[nm1]))

    dra12_dt = dra12 / delta_T21
    ddec12_dt = ddec12 / delta_T21

    # get deltaT for n-1 to n
    delta_T23 = (all_times[n] - all_times[nm1]).to(u.second)

    # predict new ra, dec
    pred_delta_ra = dra12_dt * delta_T23
    pred_delta_dec = ddec12_dt * delta_T23

    # make a skycoord with the predicted ra, dec offsets
    pred_sc = chain[nm1].spherical_offsets_by(pred_delta_ra, pred_delta_dec)

    # get angular sep from true coord to pred coord, return True if small enough
    offset = pred_sc.separation(chain[n])
    if offset < tol:
        return True

    # Failed. Return False.
    return False

def movement_search(all_sources, all_times, tolerance):
    '''
    
    '''
    # initialize chains with all sources in first observation
    queue = [[source] for source in all_sources[0]]

    # possible successes
    success = []

    # continue until nothing left in chains
    while queue:

        # get 'oldest' chain on the stack
        curr = queue.pop(0)

        # add new combinations to the queue if they pass filter
        for source in all_sources[len(curr)]:
            new_chain = curr + [source]

            valid = validity_check(new_chain, tolerance, all_times)

            if valid and len(new_chain) == len(all_sources):
                # completed chain that is valid
                success.append(new_chain)
            elif valid:
                queue.append(new_chain)

    if success:
        return success
    else:
        return None