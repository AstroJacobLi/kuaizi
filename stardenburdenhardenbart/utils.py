from __future__ import division, print_function
import os, sys
import sep
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
from contextlib import contextmanager

from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from .display import display_single, SEG_CMAP, ORG

@contextmanager
def suppress_stdout():
    """Suppress the output.

    Based on: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def set_env(project='HSC', name='HSC_LSBG'):
    import os
    
    # Master directory
    try:
        data_dir = os.path.join(
            os.getenv('HOME'), 'Research/Data/', project, name)
    except:
        raise Exception("Can not recognize this dataset!")
        
    os.chdir(data_dir)
    
    return data_dir


def set_matplotlib(usetex=True, fontsize=20):
    '''
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.
    '''

    import matplotlib.pyplot as plt
    from matplotlib.colorbar import Colorbar
    from matplotlib import rcParams
    plt.rc('text', usetex=usetex)
    plt.rc('image', cmap='inferno', interpolation='none', origin='lower')
    '''
    rcParams.update({'axes.linewidth': 1.3})
    rcParams.update({'xtick.direction': 'in'})
    rcParams.update({'ytick.direction': 'in'})
    rcParams.update({'xtick.minor.visible': 'True'})
    rcParams.update({'ytick.minor.visible': 'True'})
    rcParams.update({'xtick.major.pad': '7.0'})
    rcParams.update({'xtick.major.size': 5.0})
    rcParams.update({'xtick.major.width': '1.5'})
    rcParams.update({'xtick.minor.pad': '7.0'})
    rcParams.update({'xtick.minor.size': 3.0})
    rcParams.update({'xtick.minor.width': 1.0})
    rcParams.update({'ytick.major.pad': '7.0'})
    rcParams.update({'ytick.major.size': 5.0})
    rcParams.update({'ytick.major.width': '1.5'})
    rcParams.update({'ytick.minor.pad': '7.0'})
    rcParams.update({'ytick.minor.size': 3.0})
    rcParams.update({'ytick.minor.width': 1.0})
    rcParams.update({'axes.titlepad': '10.0'})
    '''
    rcParams.update({'font.size': fontsize})


def extract_obj(img, mask=None, b=64, f=3, sigma=5, pixel_scale=0.168, minarea=5, 
    deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0, 
    sky_subtract=False, flux_auto=True, flux_aper=None, show_fig=False, 
    verbose=True, logger=None):
    '''
    Extract objects for a given image using ``sep`` (a Python-wrapped ``SExtractor``). 
    For more details, please check http://sep.readthedocs.io and documentation of SExtractor.

    Parameters:
        img (numpy 2-D array): input image
        mask (numpy 2-D array): image mask
        b (float): size of box
        f (float): size of convolving kernel
        sigma (float): detection threshold
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        minarea (float): minimum number of connected pixels
        deblend_nthresh (float): Number of thresholds used for object deblending
        deblend_cont (float): Minimum contrast ratio used for object deblending. Set to 1.0 to disable deblending. 
        clean_param (float): Cleaning parameter (see SExtractor manual)
        sky_subtract (bool): whether subtract sky before extract objects (this will affect the measured flux).
        flux_auto (bool): whether return AUTO photometry (see SExtractor manual)
        flux_aper (list): such as [3, 6], which gives flux within [3 pix, 6 pix] annulus.

    Returns:
        :
            objects: `astropy` Table, containing the positions,
                shapes and other properties of extracted objects.

            segmap: 2-D numpy array, segmentation map
    '''
    import sep
    # Subtract a mean sky value to achieve better object detection
    b = b  # Box size
    f = f  # Filter width
    try:
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    except ValueError as e:
        img = img.byteswap().newbyteorder()
        bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    
    data_sub = img - bkg.back()
    
    sigma = sigma
    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img

    objects, segmap = sep.extract(input_data,
                                sigma,
                                mask=mask,
                                err=bkg.globalrms,
                                segmentation_map=True,
                                filter_type='matched',
                                deblend_nthresh=deblend_nthresh,
                                deblend_cont=deblend_cont,
                                clean=True,
                                clean_param=clean_param,
                                minarea=minarea)

    if verbose:
        if logger is not None:
            logger.info("    - Detected %d objects" % len(objects))
        else:
            print("# Detected %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)), name='index'))
    # Maximum flux, defined as flux within 6 * `a` (semi-major axis) in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], 
                                    6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'. 
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2* np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)), 
                              name='fwhm_custom'))
    
    # Measure R30, R50, R80
    temp = sep.flux_radius(input_data, objects['x'], objects['y'], 6. * objects['a'], [0.3, 0.5, 0.8])[0]
    objects.add_column(Column(data=temp[:, 0], name='R30'))
    objects.add_column(Column(data=temp[:, 1], name='R50'))
    objects.add_column(Column(data=temp[:, 2], name='R80'))

    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'], objects['y'], 
                                          objects['a'], objects['b'], 
                                          objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data, objects['x'], objects['y'], 
                                            2.5 * (kronrad), subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data, objects['x'][use_circle], objects['y'][use_circle],
                                                r_min, subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))
        
    if flux_aper is not None:
        if len(flux_aper) != 2:
            raise ValueError('"flux_aper" must be a list with length = 2.')
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0], flux_aper[1])[0], name='flux_ann'))

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        if min(input_data.shape) * pixel_scale < 30:
            scale_bar_length = 5
        elif min(input_data.shape) * pixel_scale > 100:
            scale_bar_length = 61
        else:
            scale_bar_length = 10
        ax[0] = display_single(input_data, ax=ax[0], scale_bar_length=scale_bar_length, pixel_scale=pixel_scale)
        if mask is not None:
            ax[0].imshow(mask.astype(float), origin='lower', alpha=0.1, cmap='Greys_r')
        from matplotlib.patches import Ellipse
        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=5 * obj['a'],
                        height=5 * obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP , ax=ax[1], scale_bar_length=scale_bar_length)
        plt.savefig('./extract_obj.png', bbox_inches='tight')
        return objects, segmap, fig
    return objects, segmap


def image_gaia_stars(image, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                     verbose=False, visual=False, size_buffer=1.4,
                     tap_url=None):
    """
    Search for bright stars using GAIA catalog. From https://github.com/dr-guangtou/kungpao.

    Parameters:
        image (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scaling factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        visual (bool): whether display the matched Gaia stars.
        
    Return: 
        gaia_results (`astropy.table.Table` object): a catalog of matched stars.
    """
    # Central coordinate
    ra_cen, dec_cen = wcs.wcs_pix2world(image.shape[0] / 2,
                                        image.shape[1] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel_scale * (image.shape)[0] * size_buffer, u.arcsec)
    img_search_y = Quantity(pixel_scale * (image.shape)[1] * size_buffer, u.arcsec)

    # Search for stars
    if tap_url is not None:
        with suppress_stdout():
            from astroquery.gaia import TapPlus, GaiaClass
            Gaia = GaiaClass(TapPlus(url=tap_url))

            gaia_results = Gaia.query_object_async(
                coordinate=img_cen_ra_dec,
                width=img_search_x,
                height=img_search_y,
                verbose=verbose)
    else:
        with suppress_stdout():
            from astroquery.gaia import Gaia

            gaia_results = Gaia.query_object_async(
                coordinate=img_cen_ra_dec,
                width=img_search_x,
                height=img_search_y,
                verbose=verbose)

    if gaia_results:
        # Convert the (RA, Dec) of stars into pixel coordinate
        ra_gaia = np.asarray(gaia_results['ra'])
        dec_gaia = np.asarray(gaia_results['dec'])
        x_gaia, y_gaia = wcs.wcs_world2pix(ra_gaia, dec_gaia, 0)

        # Generate mask for each star
        rmask_gaia_arcsec = mask_a * np.exp(
            -gaia_results['phot_g_mean_mag'] / mask_b)

        # Update the catalog
        gaia_results.add_column(Column(data=x_gaia, name='x_pix'))
        gaia_results.add_column(Column(data=y_gaia, name='y_pix'))
        gaia_results.add_column(
            Column(data=rmask_gaia_arcsec, name='rmask_arcsec'))

        if visual:
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111)

            ax1 = display_single(image, ax=ax1)
            # Plot an ellipse for each object
            for star in gaia_results:
                smask = mpl_ellip(
                    xy=(star['x_pix'], star['y_pix']),
                    width=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    height=(2.0 * star['rmask_arcsec'] / pixel_scale),
                    angle=0.0)
                smask.set_facecolor(ORG(0.2))
                smask.set_edgecolor(ORG(1.0))
                smask.set_alpha(0.3)
                ax1.add_artist(smask)

            # Show stars
            ax1.scatter(
                gaia_results['x_pix'],
                gaia_results['y_pix'],
                color=ORG(1.0),
                s=100,
                alpha=0.9,
                marker='+')

            ax1.set_xlim(0, image.shape[0])
            ax1.set_ylim(0, image.shape[1])

            return gaia_results

        return gaia_results

    return None


def gaia_star_mask(img, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                   size_buffer=1.4, gaia_bright=18.0,
                   factor_b=1.3, factor_f=1.9):
    """Find stars using Gaia and mask them out if necessary. From https://github.com/dr-guangtou/kungpao.
    
    Using the stars found in the GAIA TAP catalog, we build a bright star mask following
    similar procedure in Coupon et al. (2017).

    We separate the GAIA stars into bright (G <= 18.0) and faint (G > 18.0) groups, and
    apply different parameters to build the mask.

    Parameters:
        img (numpy 2-D array): input image.
        wcs (`astropy.wcs` object): WCS of the input image.
        pixel_scale (float): default is 0.168 (HSC pixel size). This only affect the figure scale bar.
        mask_a (float): a scale factor for the size of the plotted star, larger value means larger circle will be plotted.
        mask_b (float): a scale size for the plotted star, larger value gives larger circle. 
        gaia_bright (float): a threshold above which are classified as bright stars.
        factor_b (float): a scale size of mask for bright stars. Larger value gives smaller mask size.
        factor_f (float): a scale size of mask for faint stars. Larger value gives smaller mask size.
        
    Return: 
        msk_star (numpy 2-D array): the masked pixels are marked by one.

    """
    gaia_stars = image_gaia_stars(img, wcs, pixel_scale=pixel_scale,
                                  mask_a=mask_a, mask_b=mask_b,
                                  verbose=False, visual=False,
                                  size_buffer=size_buffer)
    print(f'#{len(gaia_stars)} stars from GAIA are masked!')
    # Make a mask image
    msk_star = np.zeros(img.shape).astype('uint8')

    if gaia_stars is not None:
        gaia_b = gaia_stars[gaia_stars['phot_g_mean_mag'] <= gaia_bright]
        sep.mask_ellipse(msk_star, gaia_b['x_pix'], gaia_b['y_pix'],
                        gaia_b['rmask_arcsec'] / factor_b / pixel_scale,
                        gaia_b['rmask_arcsec'] / factor_b / pixel_scale, 0.0, r=1.0)

        gaia_f = gaia_stars[gaia_stars['phot_g_mean_mag'] > gaia_bright]
        sep.mask_ellipse(msk_star, gaia_f['x_pix'], gaia_f['y_pix'],
                        gaia_f['rmask_arcsec'] / factor_f / pixel_scale,
                        gaia_f['rmask_arcsec'] / factor_f / pixel_scale, 0.0, r=1.0)

        return gaia_stars, msk_star

    return None, msk_star