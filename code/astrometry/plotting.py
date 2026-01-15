'''
created 15/01/2026 by truji@mit.edu

Contains useful plotting methods for astrometry.
'''

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

import numpy as np

from photutils.aperture import CircularAperture

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

def residual_plot(dra_arcsec, ddec_arcsec, dy_pix, dx_pix, outdir, name="residual_plot"):
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    ax1.scatter(dra_arcsec, ddec_arcsec, s=10, alpha=0.7)
    ax2.scatter(dy_pix, dx_pix, s=10, alpha=0.7)

    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1)
    ax1.set_xlabel(r"$\Delta$RA (arcsec)")
    ax1.set_ylabel(r"$\Delta$Dec (arcsec)")
    ax1.set_title("field to index")
    ax1.set_aspect("equal", adjustable="box")

    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)
    ax2.set_xlabel(r"$\Delta y$ (pix)")
    ax2.set_ylabel(r"$\Delta x$ (pix)")
    ax2.set_title("index to field")
    ax2.set_aspect("equal", adjustable="box")

    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def source_plot(data, sources, outdir, name="detected_sources"):
    fig = plt.figure(figsize=(14, 8.0))
    ax = fig.add_subplot()

    im = ax.imshow(data, cmap="gray_r", norm='log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

    apertures = CircularAperture(positions, r=7.0)

    apertures.plot(ax=ax, color='blue', lw=0.5, alpha=0.5)
    
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def centroid_test_plot(data, ap, centroid_results, outdir, name="centroid_test_plot"):
    fig = plt.figure()
    ax = fig.add_subplot()
    
    mask = ap.to_mask(method="center")
    values = mask.cutout(data, fill_value=np.nan)
    y0, x0 = mask.bbox.iymin, mask.bbox.ixmin

    ax.imshow(
        values,
        cmap="gray",
        norm=LogNorm(vmin=1, vmax=np.nanmax(data))
    )

    x_pos, y_pos = centroid_results

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    x_pos -= x0
    y_pos -= y0

    assert len(x_pos) == len(y_pos)

    ax.scatter(x_pos, y_pos, marker="x", c="b")

    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def fits_plot(data, wcs, outdir, name="fits_plot", pred_aps=None, fit_aps=None):

    fig = plt.figure(figsize=(9, 5.0))
    ax = fig.add_subplot(projection=wcs)

    im = ax.imshow(
        data,
        cmap="gray_r",
        norm=LogNorm(vmin=1, vmax=np.nanmax(data))
    )

    ra  = ax.coords[0]
    dec = ax.coords[1]

    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')

    ra.display_minor_ticks(True)
    dec.display_minor_ticks(True)

    ra.set_minor_frequency(4)
    dec.set_minor_frequency(4)

    ra.set_ticklabel(exclude_overlapping=False, simplify=False)
    dec.set_ticklabel(exclude_overlapping=False, simplify=False)

    ra.set_axislabel("RA (HH MM SS)")
    dec.set_axislabel("Dec (DD MM SS)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts")

    handles = []

    if fit_aps:
        apertures_f = CircularAperture(fit_aps, 25)
        apertures_f.plot(ax=ax, color='r')
        handles.append(Line2D([0], [0], color='r', label='Fit Position'))

    if pred_aps:
        apertures_p = CircularAperture(pred_aps, 50)
        apertures_p.plot(ax=ax, color='k', ls='--')
        handles.append(Line2D([0], [0], color='k', ls='--', label='Predicted Position'))

    ax.legend(handles=handles)

    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)