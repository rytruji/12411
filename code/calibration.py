'''
Created 13/01/2026 by truji@mit.edu

Useful functions for image calibration, including master calibration file creation
'''

######################################################################################
#------------------------------------------------------------------------------------#
######################################################################################

from astropy.io import fits as f

import os

import numpy as np

######################################################################################
#------------------------------------------------------------------------------------#
######################################################################################

def calibrate(lightdata, biasdir, flatdir, darkdir=None):
    '''
    Docstring for calibrate

    Calubrates the given data using master calibration files. Dark is optional, will use simplified calibration

        result = (light - bias) / flat

    if no master dark is given. Returns:
        (1) np.ndarray, resulting calibrated data.
    
    :param lightdata: np.ndarray, Image data to calibrate.
    :param biasdir: str, path to master bias fits file. 
    :param flatdata: str, path to master flat fits file. 
    :param dark: (optional, default None) str, path to master dark fits file. 
    '''

    biashdul = f.open(biasdir)
    flathdul = f.open(flatdir)
    
    biasdata = biashdul[0].data
    flatdata = flathdul[0].data

    if darkdir:
        print("Sorry, I haven't implimented the full calibration yet...")

    return (lightdata - biasdata) / flatdata

def master_bias(outdir, bias_files, name="master_bias"):
    '''
    Docstring for master_bias

    Creates and saves a master bias file. Each resulting pixel is the median of the pixel in the given bias files. Returns:
        (1) None

    :param outdir: str, directory to write output to
    :param bias_files: list, contains paths to all bias files
    :param name: (Optional, default master_bias) str, name to give output file
    '''

    biases = np.array([f.open(file)[0].data for file in bias_files])

    out = f.PrimaryHDU(np.median(biases, axis=0))

    out.writeto(os.path.join(outdir, f"{name}.fits"), overwrite=True)

def master_flat(outdir, flat_files, biasdir, name="master_flat"):
    '''
    Docstring for master_flat

    Creates and saves a master flat file. Each resulting pixel is the median of the pixel in the given flat files after normalization and bias subtraction. Returns:
        (1) None

    :param outdir: str, directory to write output to
    :param flat_files: list, contains paths to all bias files
    :param biasdir: str, path to master bias fits file. 
    :param name: (Optional, default master_flat) str, name to give output file
    '''

    biashdul = f.open(biasdir)
    biasdata = biashdul[0].data

    unbiased_flats = np.array([f.open(file)[0].data - biasdata for file in flat_files])

    normalized_flats = np.array([flat / np.median(flat) for flat in unbiased_flats])

    out = f.PrimaryHDU(np.median(normalized_flats, axis=0))

    out.writeto(os.path.join(outdir, f"{name}.fits").replace("\\","/"), overwrite=True)