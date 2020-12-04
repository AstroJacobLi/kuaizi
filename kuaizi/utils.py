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
    if gaia_stars is not None:
        print(f'# {len(gaia_stars)} stars from GAIA are masked!')
    else:
        print('No GAIA stars are masked.')
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
        temp = np.pad(psf[0].data.astype('float'), ((dy, dy), (dx, dx)), 'constant', constant_values=0)
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
        cen_x, cen_y = wcs.wcs_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))
    dx = -1.0 * (cen_x - int(cen_x))
    dy = -1.0 * (cen_y - int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs, mode='partial', fill_value=0)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        if 'COMMENT' in img_header:
            del img_header['COMMENT']
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(keyword, img_header[keyword], img_header.comments[keyword])
    
    if 'PC1_1' in dict(cutout_header).keys():
        cutout_header['CD1_1'] = cutout_header['PC1_1']
        #cutout_header['CD1_2'] = cutout_header['PC1_2']
        #cutout_header['CD2_1'] = cutout_header['PC2_1']
        cutout_header['CD2_2'] = cutout_header['PC2_2']
        cutout_header['CDELT1'] = cutout_header['CD1_1']
        cutout_header['CDELT2'] = cutout_header['CD2_2']
        cutout_header.pop('PC1_1')
        #cutout_header.pop('PC2_1')
        #cutout_header.pop('PC1_2')
        cutout_header.pop('PC2_2')
        #cutout_header.pop('CDELT1')
        #cutout_header.pop('CDELT2')
    
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

# Determine the coefficient when converting image flux to `sigma map` for HSC
def calc_sigma_coeff(images, variance_map):
    '''
    This function empirically calculate the coefficient used to convert `image flux` to `sigma map`.
    This only works for single band now. 
    
    `sigma map = coeff * image_flux`, where `sigma map = sqrt(variance map)`. 

    Parameters:
        images (numpy 2-D array or 3-D array for multiple bands): image array. Band is along the first axis.
        variance_map (numpy 2-D array or 3-D array for multiple bands): variance array, 
            i.e., the third layer of HSC cutout (`hdu[3].data`).

    Return:
        A_best (numpy array): coeff in each band
    '''
    from kuaizi.utils import extract_obj
    from astropy.convolution import convolve, Gaussian2DKernel

    if len(images.shape) == 2: # single band
        images = images[np.newaxis, :, :]
        variance_map = variance_map[np.newaxis, :, :]

    sigma_map = np.sqrt(variance_map)
    # Generate a mask, which only includes bright objects
    obj_cat, segmap = extract_obj(images.mean(axis=0), b=32, f=2, sigma=3, show_fig=False, verbose=False)
    mask = segmap > 0
    mask = convolve(mask, Gaussian2DKernel(2)) > 0.2
    ### Calculate the loss between `sigma_map` and `images`. `A` is a coefficient.
    sigma_map -= np.median(sigma_map, axis=(1, 2))[:, np.newaxis, np.newaxis] # remove a background
    mask = np.repeat(mask[np.newaxis, :, :], len(images), axis=0) # len(images) is the number of bands
    
    A_set = np.linspace(0, 0.01, 200)
    loss = np.zeros([len(images), len(A_set)])
    for i, A in enumerate(A_set):
        temp = images
        diff = (sigma_map - A * images)[mask].reshape(len(images), -1)
        loss[:, i] = np.linalg.norm(diff, axis=1) / np.sum(mask[0])
    A_best = A_set[np.argmin(loss, axis=1)]

    return A_best

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