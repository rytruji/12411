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
from matplotlib.gridspec import GridSpec

###############################################################################################################
#-------------------------------------------------------------------------------------------------------------#
###############################################################################################################

def residual_plot(residuals, outdir, name="residual_plot"):
    print("Plotting residuals...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), constrained_layout=True)

    cmap = plt.cm.viridis
    n = len(residuals)

    for i, (dra_arcsec, ddec_arcsec, dy_pix, dx_pix) in enumerate(residuals):
        color = cmap(i / max(n - 1, 1))
        ax1.scatter(dra_arcsec, ddec_arcsec, s=10, color=color, alpha=0.7)
        ax2.scatter(dx_pix, dy_pix, s=10, color=color, alpha=0.7)

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

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Done.")



def source_plot(data, sources, outdir, name="detected_sources"):
    fig = plt.figure(figsize=(14, 8.0))
    ax = fig.add_subplot()

    im = ax.imshow(data, cmap="gray_r", norm='log')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

    apertures = CircularAperture(positions, r=7.0)

    apertures.plot(ax=ax, color='blue', lw=0.5, alpha=0.5)
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def centroid_test_plot(data, ap, centroid_results, outdir, name="centroid_test_plot"):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.imshow(
        data,
        cmap="gray_r",
        norm=LogNorm(1, vmax=np.nanmax(data))
    )

    nx, ny = data.shape
    
    mask = ap.to_mask(method="center")
    values = mask.cutout(data, fill_value=np.nan)
    y0, x0, y1, x1 = mask.bbox.iymin, mask.bbox.ixmin, mask.bbox.iymax, mask.bbox.ixmax

    if y0 > ny // 2 and x0 > nx // 2:
        bounds=(0.6, 0.6, 0.35, 0.35)
    else:
        bounds=(0.1, 0.1, 0.35, 0.35)

    axins = ax.inset_axes(
        bounds=bounds
    )

    axins.set_xlim(x0, x1)
    axins.set_ylim(y1, y0)

    axins.imshow(
        values,
        cmap="gray",
        norm=LogNorm(vmin=np.nanmin(values), vmax=np.nanmax(values)),
        extent=(x0, x1, y1, y0),
        origin="upper"
    )

    x_pos, y_pos = centroid_results

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    assert len(x_pos) == len(y_pos)

    axins.scatter(x_pos, y_pos, marker="x", c="b")

    ax.indicate_inset_zoom(axins, edgecolor="black")

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def fits_plot(data, wcs, outdir, name="fits_plot", pred_xy=None, fit_xy=None):

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

    if fit_xy:
        apertures_f = CircularAperture(fit_xy, 25)
        apertures_f.plot(ax=ax, color='r')
        handles.append(Line2D([0], [0], color='r', label='Fit Position'))

    if pred_xy:
        apertures_p = CircularAperture(pred_xy, 50)
        apertures_p.plot(ax=ax, color='k', ls='--')
        handles.append(Line2D([0], [0], color='k', ls='--', label='Predicted Position'))

    ax.legend(handles=handles)

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def position_plot(positions, residuals, outdir, name="position_plot", obs_reference=None):
    fig = plt.figure(figsize=(10, 10))

    gs = GridSpec(2, 2, fig, width_ratios=[4, 1], height_ratios=[4, 1])

    ax_dec = fig.add_subplot(gs[0,1])
    ax_ra = fig.add_subplot(gs[1,0])
    ax_resids = fig.add_subplot(gs[1,1])

    if obs_reference:
        ax = fig.add_subplot(gs[0,0], projection=obs_reference.wcs)

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

        im = ax.imshow(
                obs_reference.data,
                cmap="gray_r",
                norm=LogNorm(vmin=1, vmax=np.nanmax(obs_reference.data))
            )
    else:
        ax = fig.add_subplot(gs[0,0])
        

    ax_ra.sharex(ax)
    ax_dec.sharey(ax)

    ax_resids.sharex(ax_dec)
    ax_resids.sharey(ax_ra)

    ax.scatter([coord.ra.deg for coord in positions],
                [coord.dec.deg for coord in positions],
                s=30,
                marker="o",
                alpha=0.7,
                color="k"
                )
        
    ra_offset, dec = [pair for pair in zip(*residuals[0])]
    dec_offset, ra = [pair for pair in zip(*residuals[1])]

    ra_offset = np.array([offset.value for offset in ra_offset])
    dec_offset = np.array([offset.value for offset in dec_offset])

    y_err=np.std(dec_offset)
    x_err=np.std(ra_offset)

    ax_resids.scatter(ra_offset, dec_offset, color="k")

    ax_ra.errorbar(ra, dec_offset, yerr=y_err, color="k", fmt="o", capsize=3)
    ax_dec.errorbar(ra_offset, dec, xerr=x_err, color="k", fmt="o", capsize=3)

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def segmentation_plot(data, cat, segment_map, outdir, name="segmentation_plot"):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))

    ax1.imshow(data, origin='upper', cmap='grey_r', norm=LogNorm(vmin=1, vmax=np.nanmax(data)))
    ax1.set_title('Background-subtracted Data')

    ax2.imshow(segment_map, origin='upper', cmap=segment_map.cmap,
            interpolation='nearest')
    ax2.set_title('Segmentation Image')

    cat.plot_kron_apertures(ax=ax1, color='white', lw=0.5, alpha=0.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=0.5, alpha=0.5)
    
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{name}.pdf").replace("\\","/"), dpi=300, bbox_inches="tight")
    plt.close(fig)