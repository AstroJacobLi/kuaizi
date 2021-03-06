from __future__ import division, print_function
import os
import sys
import sep
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
from contextlib import contextmanager

from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column, vstack
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


def set_env(project='HSC', name='HSC_LSBG', data_dir='Research/Data/'):
    import os

    # Master directory
    try:
        data_dir = os.path.join(
            os.getenv('HOME'), data_dir, project, name)
    except:
        raise Exception("Can not recognize this dataset!")

    os.chdir(data_dir)
    return data_dir


def set_matplotlib(style='JL', usetex=True, fontsize=13, figsize=(6, 5)):
    '''
    Default matplotlib settings, borrowed from Song Huang. I really like his plotting style.

    Parameters:
        style (str): options are "JL", "SM" (supermongo-like).
    '''

    import matplotlib.pyplot as plt
    from matplotlib.colorbar import Colorbar
    from matplotlib import rcParams
    import kuaizi
    # Use JL as a template
    pkg_path = kuaizi.__path__[0]
    plt.style.use(os.path.join(pkg_path, 'mplstyle/JL.mplstyle'))
    rcParams.update({'font.size': fontsize,
                     'figure.figsize': "{0}, {1}".format(figsize[0], figsize[1]),
                     'text.usetex': usetex})

    if style == 'SM':
        rcParams.update({
            "figure.figsize": "6, 6",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.3,
            "font.family": "monospace",
            "font.stretch": "semi-expanded",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.bf": "monospace:bold",
            "mathtext.cal": "monospace:bold",
            "mathtext.it": "monospace:italic",
            "mathtext.rm": "monospace",
            "mathtext.sf": "monospace",
            "mathtext.tt": "monospace",
            "mathtext.fallback": "cm",
            "mathtext.default": 'it'
        })

        if usetex is True:
            rcParams.update({
                "text.latex.preamble": '\n'.join([
                    '\\usepackage{amsmath}'
                    '\\usepackage[T1]{fontenc}',
                    '\\usepackage{courier}',
                    '\\usepackage[variablett]{lmodern}',
                    '\\usepackage[LGRgreek]{mathastext}',
                    '\\renewcommand{\\rmdefault}{\\ttdefault}'
                ])
            })

    if style == 'nature':
        rcParams.update({
            "font.family": "sans-serif",
            # The default edge colors for scatter plots.
            "scatter.edgecolors": "black",
            "mathtext.fontset": "stixsans"
        })


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
            logger.info("    Detected %d objects" % len(objects))
        else:
            print("    Detected %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)), name='index'))
    # Maximum flux, defined as flux within 6 * `a` (semi-major axis) in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'],
                                                  6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'.
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2 * np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)),
                              name='fwhm_custom'))

    # Measure R30, R50, R80
    temp = sep.flux_radius(
        input_data, objects['x'], objects['y'], 6. * objects['a'], [0.3, 0.5, 0.8])[0]
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
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        if min(input_data.shape) * pixel_scale < 30:
            scale_bar_length = 5
        elif min(input_data.shape) * pixel_scale > 100:
            scale_bar_length = 61
        else:
            scale_bar_length = 10
        ax[0] = display_single(
            input_data, ax=ax[0], scale_bar_length=scale_bar_length, pixel_scale=pixel_scale)
        if mask is not None:
            ax[0].imshow(mask.astype(float), origin='lower',
                         alpha=0.1, cmap='Greys_r')
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
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP,
                               ax=ax[1], scale_bar_length=scale_bar_length)
        # plt.savefig('./extract_obj.png', bbox_inches='tight')
        return objects, segmap, fig
    return objects, segmap


def _image_gaia_stars_tigress(image, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                              verbose=False, visual=False, size_buffer=1.4):
    """
    Search for bright stars using GAIA catalogs on Tigress (`/tigress/HSC/refcats/htm/gaia_dr2_20200414`).
    For more information, see https://community.lsst.org/t/gaia-dr2-reference-catalog-in-lsst-format/3901.
    This function requires `lsstpipe`.

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
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[1] / 2,
                                        image.shape[0] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_ra_size = Quantity(pixel_scale * (image.shape)
                           [1] * size_buffer, u.arcsec).to(u.degree)
    img_dec_size = Quantity(pixel_scale * (image.shape)
                            [0] * size_buffer, u.arcsec).to(u.degree)

    # Search for stars in Gaia catatlogs, which are stored in
    # `/tigress/HSC/refcats/htm/gaia_dr2_20200414`.
    try:
        from lsst.meas.algorithms.htmIndexer import HtmIndexer
        import lsst.geom as geom

        def getShards(ra, dec, radius):
            htm = HtmIndexer(depth=7)

            afw_coords = geom.SpherePoint(
                geom.Angle(ra, geom.degrees),
                geom.Angle(dec, geom.degrees))

            shards, onBoundary = htm.getShardIds(
                afw_coords, radius * geom.degrees)
            return shards

    except ImportError as e:
        # Output expected ImportErrors.
        print(e.__class__.__name__ + ": " + e.message)
        print('LSST Pipe must be installed to query Gaia stars on Tigress.')

    # find out the Shard ID of target area in the HTM (Hierarchical triangular mesh) system
    print('    Taking Gaia catalogs stored in `Tigress`')

    shards = getShards(ra_cen, dec_cen, max(
        img_ra_size, img_dec_size).to(u.degree).value)
    cat = vstack([Table.read(
        f'/tigress/HSC/refcats/htm/gaia_dr2_20200414/{index}.fits') for index in shards])
    cat['coord_ra'] = cat['coord_ra'].to(u.degree)
    # why GAIA coordinates are in RADIAN???
    cat['coord_dec'] = cat['coord_dec'].to(u.degree)

    # Trim this catalog a little bit
    # Ref: https://github.com/MerianSurvey/caterpillar/blob/main/caterpillar/catalog.py
    if cat:  # if not empty
        gaia_results = cat[
            (cat['coord_ra'] > img_cen_ra_dec.ra - img_ra_size / 2) &
            (cat['coord_ra'] < img_cen_ra_dec.ra + img_ra_size / 2) &
            (cat['coord_dec'] > img_cen_ra_dec.dec - img_dec_size / 2) &
            (cat['coord_dec'] < img_cen_ra_dec.dec + img_dec_size / 2)
        ]
        gaia_results.rename_columns(['coord_ra', 'coord_dec'], ['ra', 'dec'])

        gaia_results['phot_g_mean_mag'] = -2.5 * \
            np.log10(
                (gaia_results['phot_g_mean_flux'] / (3631 * u.Jy)))  # AB magnitude

        # Convert the (RA, Dec) of stars into pixel coordinate using WCS
        x_gaia, y_gaia = wcs.all_world2pix(gaia_results['ra'],
                                           gaia_results['dec'], 0)

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

            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])

        return gaia_results

    return None


def image_gaia_stars(image, wcs, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                     verbose=False, visual=False, size_buffer=1.4, tap_url=None):
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
    ra_cen, dec_cen = wcs.all_pix2world(image.shape[0] / 2,
                                        image.shape[1] / 2,
                                        0)
    img_cen_ra_dec = SkyCoord(
        ra_cen, dec_cen, unit=('deg', 'deg'), frame='icrs')

    # Width and height of the search box
    img_search_x = Quantity(pixel_scale * (image.shape)
                            [0] * size_buffer, u.arcsec)
    img_search_y = Quantity(pixel_scale * (image.shape)
                            [1] * size_buffer, u.arcsec)

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
        x_gaia, y_gaia = wcs.all_world2pix(ra_gaia, dec_gaia, 0)

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

    return None


def gaia_star_mask(img, wcs, gaia_stars=None, pixel_scale=0.168, mask_a=694.7, mask_b=3.5,
                   size_buffer=1.4, gaia_bright=18.0,
                   factor_b=1.3, factor_f=1.9, tigress=False):
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
        tigress (bool): whether take Gaia catalogs on Tigress

    Return: 
        msk_star (numpy 2-D array): the masked pixels are marked by one.

    """
    if gaia_stars is None:
        if tigress:
            gaia_stars = _image_gaia_stars_tigress(img, wcs, pixel_scale=pixel_scale,
                                                   mask_a=mask_a, mask_b=mask_b,
                                                   verbose=False, visual=False,
                                                   size_buffer=size_buffer)
        else:
            gaia_stars = image_gaia_stars(img, wcs, pixel_scale=pixel_scale,
                                          mask_a=mask_a, mask_b=mask_b,
                                          verbose=False, visual=False,
                                          size_buffer=size_buffer)
        if gaia_stars is not None:
            print(f'    {len(gaia_stars)} stars from Gaia are masked!')
        else:  # does not find Gaia stars
            print('    No Gaia stars are masked.')
    else:
        print(f'    {len(gaia_stars)} stars from Gaia are masked!')

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


def padding_PSF(psf_list):
    '''
    If the sizes of HSC PSF in all bands are not the same, this function pads the smaller PSFs.

    Parameters:
        psf_list: a list returned by `unagi.task.hsc_psf` function

    Returns:
        psf_pad: a list including padded PSFs. They now share the same size.
    '''
    # Padding PSF cutouts from HSC
    max_len = max([max(psf[0].data.shape) for psf in psf_list])
    psf_pad = []
    for psf in psf_list:
        y_len, x_len = psf[0].data.shape
        dy = (max_len - y_len) // 2
        dx = (max_len - x_len) // 2
        temp = np.pad(psf[0].data.astype('float'), ((dy, dy),
                                                    (dx, dx)), 'constant', constant_values=0)
        if temp.shape == (max_len, max_len):
            psf_pad.append(temp)
        else:
            raise ValueError('Wrong size!')

    return psf_pad


# Save 2-D numpy array to `fits`
def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """
    Save numpy 2-D arrays to `fits` file. (from `kungpao` https://github.com/dr-guangtou/kungpao)

    Parameters:
        img (numpy 2-D array): The 2-D array to be saved.
        fits_file (str): File name of `fits` file.
        wcs (``astropy.wcs.WCS`` object): World coordinate system (WCS) of this image.
        header (``astropy.io.fits.header`` or str): header of this image.
        overwrite (bool): Whether overwrite the file. Default is True.

    Returns:
        img_hdu (``astropy.fits.PrimaryHDU`` object)
    """
    img_hdu = fits.PrimaryHDU(img)

    if header is not None:
        img_hdu.header = header
        if wcs is not None:
            hdr = copy.deepcopy(header)
            wcs_header = wcs.to_header()
            import fnmatch
            for i in hdr:
                if i in wcs_header:
                    hdr[i] = wcs_header[i]
                if 'PC*' in wcs_header:
                    if fnmatch.fnmatch(i, 'CD?_?'):
                        hdr[i] = wcs_header['PC' + i.lstrip('CD')]
            img_hdu.header = hdr
    elif wcs is not None:
        wcs_header = wcs.to_header()
        wcs_header = fits.Header({'SIMPLE': True})
        wcs_header.update(NAXIS1=img.shape[1], NAXIS2=img.shape[0])
        for card in list(wcs.to_header().cards):
            wcs_header.append(card)
        img_hdu.header = wcs_header
    else:
        img_hdu = fits.PrimaryHDU(img)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)
    return img_hdu

# Cutout image
def img_cutout(img, wcs, coord_1, coord_2, size=[60.0, 60.0], pixel_scale=0.168,
               pixel_unit=False, img_header=None, prefix='img_cutout',
               out_dir=None, save=True):
    """
    Generate image cutout with updated WCS information. (From ``kungpao`` https://github.com/dr-guangtou/kungpao) 

    Parameters:
        img (numpy 2-D array): image array.
        wcs (``astropy.wcs.WCS`` object): WCS of input image array.
        coord_1 (float): ``ra`` or ``x`` of the cutout center.
        coord_2 (float): ``dec`` or ``y`` of the cutout center.
        size (array): image size, such as (800, 1000), in arcsec unit by default.
        pixel_scale (float): pixel size, in the unit of "arcsec/pixel".
        pixel_unit (bool):  When True, ``coord_1``, ``coord_2`` becomes ``X``, ``Y`` pixel coordinates. 
            ``size`` will also be treated as in pixels.
        img_header: The header of input image, typically ``astropy.io.fits.header`` object.
            Provide the haeder in case you can save the infomation in this header to the new header.
        prefix (str): Prefix of output files.
        out_dir (str): Directory of output files. Default is the current folder.
        save (bool): Whether save the cutout image.

    Returns: 
        :
            cutout (numpy 2-D array): the cutout image.

            [cen_pos, dx, dy]: a list contains center position and ``dx``, ``dy``.

            cutout_header: Header of cutout image.
    """

    from astropy.nddata import Cutout2D
    if not pixel_unit:
        # img_size in unit of arcsec
        cutout_size = np.asarray(size) / pixel_scale
        cen_x, cen_y = wcs.all_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))
    dx = -1.0 * (cen_x - int(cen_x))
    dy = -1.0 * (cen_y - int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs,
                      mode='partial', fill_value=0)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        if 'COMMENT' in img_header:
            del img_header['COMMENT']
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(
                keyword, img_header[keyword], img_header.comments[keyword])

    if 'PC1_1' in dict(cutout_header).keys():
        cutout_header['CD1_1'] = cutout_header['PC1_1']
        #cutout_header['CD1_2'] = cutout_header['PC1_2']
        #cutout_header['CD2_1'] = cutout_header['PC2_1']
        cutout_header['CD2_2'] = cutout_header['PC2_2']
        cutout_header['CDELT1'] = cutout_header['CD1_1']
        cutout_header['CDELT2'] = cutout_header['CD2_2']
        cutout_header.pop('PC1_1')
        # cutout_header.pop('PC2_1')
        # cutout_header.pop('PC1_2')
        cutout_header.pop('PC2_2')
        # cutout_header.pop('CDELT1')
        # cutout_header.pop('CDELT2')

    # Build a HDU
    hdu = fits.PrimaryHDU(header=cutout_header)
    hdu.data = cutout.data
    #hdu.data = np.flipud(cutout.data)
    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout, [cen_pos, dx, dy], cutout_header


# Filter correction between surveys
def filter_corr_gunn_stryker(surveys, n_terms=4, x_upper_lims=[2, 5, 5], y_lim=None,
    skip_calc_mag=False, gunn_stryker_dir='/Users/jiaxuanli/Research/HSC_Dragonfly_DECaLS/Filters/GunnStryker/'):
    '''
    Calculate filter correction between two surveys (such as HSC-r and DECam-r) 
    based on synthetic photometry of stars (Gunn-Stryker library). 
    This funciton change the input surveys in-place. `sedpy` is required. 
    Gunn-Stryker spectra atlas: https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/gunn-stryker-atlas-list

    Parameters:
        surveys (list): elements are a dictionary contains the name and filters in `sedpy` format. For example:
            ```
            hsc = {
                'name': 'HSC',
                'filters':
                observate.load_filters(['hsc_g', 'hsc_r2', 'hsc_i2'],
                                    directory=f"{filter_path}/hsc_responses_all_rev3/")
            }
            ```
        n_terms (int): order of polynomial fitting.
        x_upper_lims (list): the upper limits of x-axis (i.e., color) when doing 
            polynomial fitting, in order to remove outliers. The length of `x_upper_lims` 
            should be consistent with the number of possible combinations of the filters in the first survey, 
            i.e., `len(x_upper_lims) = len(combinations(range(len(survey1['filters'])), 2))`.
        y_lim (list): y_lim for the figure. Default is None.
        skip_calc_mag (bool): if you have already calculated the synthetic magnitudes based 
            on Gunn-Stryker catalog, you may want to skip this to save some time.
        gunn_stryker_dir (str): the directory of Gunn-Stryker atals of spectra. 

    Returns:
        surveys (list): the input list of surveys. 
            The synthetic magnitudes of each survey is updated in-place. 
    '''
    import re
    from itertools import combinations
    colorset = plt.rcParams['axes.prop_cycle'].by_key()['color']

    spec_cat = Table.read(os.path.join(gunn_stryker_dir, 'gsspectype.ascii'), 
                          format='ascii.no_header')

    if len(surveys) > 2:
        raise ValueError("Please only input two surveys!")

    # Calculate magnitude in each filter, for each star in Gunn-Stryker catalog
    print('Calculating synthetic magnitudes based on Gunn-Stryker catalog')
    if skip_calc_mag:
        # Skip calculating synthetic mag, check whether the input "survey" already has synthetic mags
        if not hasattr(surveys[0]['filters'][0], 'mag'):
            print('The input `survey` does not have synthetic magnitudes. Calculate them again.')
            skip_calc_mag = False
    
    if not skip_calc_mag:
        for survey in surveys:
            mag_dict = {}
            for filt in survey['filters']:
                temp = []
                for obj in spec_cat: # iterate over stars
                    filename = obj['col1'].rstrip('.tab') + '.ascii'
                    spec = Table.read(os.path.join(gunn_stryker_dir, filename), format='ascii')
                    mag = filt.ab_mag(spec['col1'], spec['col2'])
                    temp.append(mag)
                filt.mag = np.array(temp)

    # Fit a relation between color and filter difference
    survey_filters = [[re.sub('\d', '', re.sub('^\w+_', '', item.name)) for item in survey['filters']] for survey in surveys]
    common_filters = np.intersect1d(*survey_filters) # such as ['g', 'r']

    print(f"Deriving the filter correction between {surveys[0]['name']} and {surveys[1]['name']} in {len(common_filters)} bands: {common_filters}")

    survey1, survey2 = surveys
    fig, axes = plt.subplots(len(common_filters), len(survey1['filters']), 
                            figsize=(5 * len(survey1['filters']), 4. * len(common_filters)), 
                            sharex=True, sharey=True)

    for i, filt in enumerate(common_filters):
        survey1_filt = [item for item in survey1['filters'] if filt in item.name][0]
        survey2_filt = [item for item in survey2['filters'] if filt in item.name][0]
        
        for j, color_ind in enumerate(combinations(range(len(survey1['filters'])), 2)):
            y = survey1_filt.mag - survey2_filt.mag
            # y-axis is the mag difference of the same bands in two surveys 
            x = survey1['filters'][color_ind[0]].mag - survey1['filters'][color_ind[1]].mag 
            # x-axis is color (such as g-i)
            negative = (np.sum(y < 0) / len(y) > 0.6) # most of y is negative or not
            
            # sort by x
            zipped = sorted(zip(x, y), key=lambda x: x[0])
            x, y = list(zip(*zipped))
            x, y = map(lambda t: np.array(t), [x, y])
            
            # fit polynomial
            flag = (x < x_upper_lims[j])
            poly_params = np.polyfit(x[flag], y[flag], n_terms)
            func = np.poly1d(poly_params)
            y_poly = func(x)
            
            # Plot figure
            ax = axes[i, j]
            title = Polynomial_to_LaTeX(func)
            #title = '$y = ' + '+'.join([r'{0:.3f} x^{1:d}'.format(item, n_term - num) for num, item in enumerate(poly_params)]) + '$'
            ax.scatter(x, y, color='k')
            ylim = ax.get_ylim()
            if y_lim is not None: ylim = y_lim
            xlim = ax.get_xlim()
            
            ax.plot(x, y_poly, color=colorset[j], lw=2)
            ax.set_xlabel(survey1['filters'][color_ind[0]].name + ' - ' + survey1['filters'][color_ind[1]].name)
            if j == 0: ax.set_ylabel(survey1_filt.name + ' - ' + survey2_filt.name)
            if negative:
                ax.text((xlim[0] + xlim[1]) * 0.5, (0.85 * ylim[0] + 0.15 * ylim[1]), title, 
                        ha='center', va='center', fontsize=9)
            else:
                ax.text((xlim[0] + xlim[1]) * 0.5, (0.15 * ylim[0] + 0.85 * ylim[1]), title, 
                        ha='center', va='center', fontsize=9)
            ax.hlines(0, xlim[0], xlim[1], color='gray', linestyle='-.')
            #ax.set_title(title, fontsize=8)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
    
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.show()

    return surveys


################# HDF5 related ##################
# Print attributes of a HDF5 file
def h5_print_attrs(f):
    '''
    Print all attributes of a HDF5 file.

    Parameters:
    ----------
    f: HDF5 file.

    Returns:
    --------
    All attributes of 'f'
    '''
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print("    %s: %s" % (key, val))

    f.visititems(print_attrs)

# Rewrite dataset
def h5_rewrite_dataset(mother_group, key, new_data):
    '''
    Rewrite the given dataset of a HDF5 group.

    Parameters:
    ----------
    mother_group: HDF5 group class.
    key: string, the name of the dataset to be writen into.
    new_data: The data to be written into.
    '''
    if any([keys == key for keys in mother_group.keys()]):
        mother_group.__delitem__(key)
        dt = mother_group.create_dataset(key, data=new_data)
    else:
        dt = mother_group.create_dataset(key, data=new_data)
    return dt

# Create New group
def h5_new_group(mother_group, key):
    '''
    Create a new data_group

    Parameters:
    ----------
    mother_group: HDF5 group class.
    key: string, the name of the dataset to be writen into.
    new_data: The data to be written into.
    '''
    if not any([keys == key for keys in mother_group.keys()]):
        new_grp = mother_group.create_group(key)
    else:
        new_grp = mother_group[key]
    return new_grp

# String to dictionary
def str2dic(string):
    '''
    This function is used to load string dictionary and convert it into python dictionary.
    '''
    import yaml
    return yaml.load(string)

def Polynomial_to_LaTeX(p):
    """
    Small function to print nicely the polynomial p as we write it in maths, in LaTeX code.
    From https://perso.crans.org/besson/publis/notebooks/Demonstration%20of%20numpy.polynomial.Polynomial%20and%20nice%20display%20with%20LaTeX%20and%20MathJax%20(python3).html

    Parameter:
        p (numpy Polynomial object): such as p = np.poly1d(poly_params)
    
    Return:
        Latex string.
    """
    coefs = p.coef  # List of coefficient, sorted by increasing degrees
    coefs = [round(item, 3) for item in coefs]
    res = ""  # The resulting string
    for i, a in enumerate(coefs):
        if int(a) == a:  # Remove the trailing .0
            a = int(a)
        if i == 0:  # First coefficient, no need for X
            if a > 0:
                res += "{a} + ".format(a=a)
            elif a < 0:  # Negative a is printed like (a)
                res += "({a}) + ".format(a=a)
            # a = 0 is not displayed 
        elif i == 1:  # Second coefficient, only X and not X**i
            if a == 1:  # a = 1 does not need to be displayed
                res += "X + "
            elif a > 0:
                res += "{a} \;X + ".format(a=a)
            elif a < 0:
                res += "({a}) \;X + ".format(a=a)
        else:
            if i == 3 and (a != 0 | len(coefs) > 4):
                res += '$ \n $' # line break
            if a == 1:
                # A special care needs to be addressed to put the exponent in {..} in LaTeX
                res += "X^{i} + ".format(i="{%d}" % i)
            elif a > 0:
                res += "{a} \;X^{i} + ".format(a=a, i="{%d}" % i)
            elif a < 0:
                res += "({a}) \;X^{i} + ".format(a=a, i="{%d}" % i)

    return "$ Y= " + res[:-3] + "$" if res else ""