import sep
import numpy as np
import scarlet
from scarlet.wavelet import Starlet

from .utils import extract_obj, image_gaia_stars, _image_gaia_stars_tigress
from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from kuaizi.mock import Data


def interpolate(data_lr, data_hr):
    ''' Interpolate low resolution data to high resolution

    Parameters
    ----------
    data_lr: Data
        low resolution Data
    data_hr: Data
        high resolution Data

    Result
    ------
    interp: numpy array
        the images in data_lr interpolated to the grid of data_hr
    '''
    frame_lr = scarlet.Frame(data_lr.images.shape,
                             wcs=data_lr.wcs, channels=data_lr.channels)
    frame_hr = scarlet.Frame(data_hr.images.shape,
                             wcs=data_hr.wcs, channels=data_hr.channels)

    coord_lr0 = (np.arange(data_lr.images.shape[1]), np.arange(
        data_lr.images.shape[1]))
    coord_hr = (np.arange(data_hr.images.shape[1]), np.arange(
        data_hr.images.shape[1]))
    coord_lr = scarlet.resampling.convert_coordinates(
        coord_lr0, frame_lr, frame_hr)

    interp = []
    for image in data_lr.images:
        interp.append(scarlet.interpolation.sinc_interp(
            image[None, :, :], coord_hr, coord_lr, angle=None)[0].T)
    return np.array(interp)

# Vanilla detection: SEP


def vanilla_detection(detect_image, mask=None, sigma=3, b=64, f=3, minarea=5,
                      convolve=False, conv_radius=None, deblend_nthresh=30,
                      deblend_cont=0.001, sky_subtract=True, show_fig=True, **kwargs):
    '''
    Source detection using Source Extractor (actually SEP).

    Parameters
    ----------
    detect_image: 2-D numpy array
        image
    mask: numpy 2-D array
        image mask
    sigma: float
        detection threshold
    b: float
        box size
    f: float
        kernel size
    minarea: float
        minimum area for a source
    sky_subtract: bool
        whether subtract the estimated sky from the input image, then detect sources
    show_fig: bool
        whether plot a figure showing objects and segmentation map
    **kwargs: see `utils.extract_obj`.

    Result
    ------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    fig: `matplotlib.pyplot.figure` object
    '''
    result = extract_obj(
        detect_image,
        mask=mask,
        b=b,
        f=f,
        sigma=sigma,
        minarea=minarea,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        sky_subtract=sky_subtract,
        convolve=convolve,
        conv_radius=conv_radius,
        show_fig=show_fig,
        **kwargs)

    obj_cat = result[0]
    arg_ind = obj_cat.argsort('flux', reverse=True)
    obj_cat.sort('flux', reverse=True)
    obj_cat['index'] = np.arange(len(obj_cat))
    segmap = result[1]
    segmap = np.append(-1, np.argsort(arg_ind))[segmap] + 1

    if show_fig is True:
        fig = result[2]
        return obj_cat, segmap, fig
    else:
        return obj_cat, segmap


def wavelet_detection(detect_image, mask=None, wavelet_lvl=4, low_freq_lvl=0, high_freq_lvl=1,
                      sigma=3, b=64, f=3, minarea=5, convolve=False, conv_radius=None, deblend_nthresh=30,
                      deblend_cont=0.001, sky_subtract=True, show_fig=True, **kwargs):
    '''
    Perform wavelet transform before detecting sources. This enable us to emphasize features with high frequency or low frequency.

    Parameters
    ----------
    detect_image: 2-D numpy array
        image
    mask: numpy 2-D array
        image mask
    wavelet_lvl: int
        the number of wavelet decompositions
    high_freq_lvl: int
        this parameter controls how much low-frequency features are wiped away. It should be smaller than `wavelet_lvl - 1`.
        `high_freq_lvl=0` means no low-freq features are wiped (equivalent to vanilla), higher number yields a image with less low-freq features.
    sigma: float
        detection threshold
    b: float
        box size
    f: float
        kernel size
    minarea: float
        minimum area for a source
    sky_subtract: bool
        whether subtract the estimated sky from the input image, then detect sources
    show_fig: bool
        whether plot a figure showing objects and segmentation map
    **kwargs: see `utils.extract_obj`.

    Result
    ------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    fig: `matplotlib.pyplot.figure` object

    '''
    Sw = Starlet.from_image(detect_image)  # wavelet decomposition
    # Now the number of levels are calculated automatically
    # Can be accessed as lvl = Sw.scales
    w = Sw.coefficients
    iw = Sw.image

    if high_freq_lvl != 0:
        w[(high_freq_lvl):, :, :] = 0  # remove low frequency features
        # w: from high to low

    if low_freq_lvl != 0:
        w[:(low_freq_lvl), :, :] = 0  # remove high frequency features

    # image with high-frequency features highlighted
    high_freq_image = Starlet.from_coefficients(w).image

    result = vanilla_detection(
        high_freq_image,
        mask=mask,
        sigma=sigma,
        b=b,
        f=f,
        minarea=minarea,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        sky_subtract=sky_subtract,
        convolve=convolve,
        conv_radius=conv_radius,
        show_fig=show_fig,
        **kwargs)

    if show_fig is True:
        obj_cat, segmap, fig = result
        return obj_cat, segmap, fig
    else:
        obj_cat, segmap = result
        return obj_cat, segmap


def makeCatalog(datas, mask=None, lvl=3, method='wavelet', convolve=False, conv_radius=5,
                match_gaia=True, show_fig=True, visual_gaia=True, tigress=False, layer_ind=None, **kwargs):
    ''' Creates a detection catalog by combining low and high resolution data.

    This function is used for detection before running scarlet.
    It is particularly useful for stellar crowded fields and for detecting high frequency features.

    Parameters
    ----------
    datas: array
        array of Data objects
    mask: numpy 2-D array
        image mask
    lvl: int
        detection lvl, i.e., sigma in SEP
    method: str
        Options:
            "wavelet" uses wavelet decomposition of images before combination, emphasizes high-frequency features
            "vanilla" directly detect objects using SEP
    match_gaia: bool
        whether matching the detection catalog with Gaia dataset
    show_fig: bool
        whether show the detection catalog as a figure
    visual_gaia: bool
        whether mark Gaia stars in the figure
    kwargs:
        See the arguments of 'utils.extract_obj'.

    Returns
    -------
    obj_cat: `astropy.table.Table` object
        catalog of detected sources
    segmap: numpy array
        segmentation map
    bg_rms: array
        background level for each dataset
    '''
    if 'logger' in kwargs:
        logger = kwargs['logger']
    else:
        logger = None

    if len(datas) == 1:
        hr_images = datas[0].images / \
            np.abs(np.sum(datas[0].images, axis=(1, 2)))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)
        # _weights = datas[0].weights.sum(axis=(1, 2)) / datas[0].weights.sum()
        # detect_image = (_weights[:, None, None] * datas[0].images).sum(axis=0)
    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalisation of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / \
            np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)

    if np.size(detect_image.shape) == 3:
        detect = detect_image.mean(axis=0)
    else:
        detect = detect_image

    if layer_ind is not None:
        detect = datas[0].images[layer_ind]

    # we better subtract background first, before convolve
    if method == 'wavelet':
        result = wavelet_detection(
            detect, mask=mask, sigma=lvl, show_fig=show_fig, convolve=convolve, conv_radius=conv_radius, **kwargs)
    else:
        result = vanilla_detection(
            detect, mask=mask, sigma=lvl, show_fig=show_fig, convolve=convolve, conv_radius=conv_radius, **kwargs)

    obj_cat = result[0]
    segmap = result[1]

    # RA and Dec
    if len(datas) == 1:
        ra, dec = datas[0].wcs.wcs_pix2world(obj_cat['x'], obj_cat['y'], 0)
        obj_cat.add_columns([Column(data=ra, name='ra'),
                             Column(data=dec, name='dec')])
    else:
        ra_lr, dec_lr = data_lr.wcs.wcs_pix2world(
            obj_cat['x'], obj_cat['y'], 0)
        ra_hr, dec_hr = data_hr.wcs.wcs_pix2world(
            obj_cat['x'], obj_cat['y'], 0)
        obj_cat.add_columns(
            [Column(data=ra_lr, name='ra_lr'), Column(data=dec_lr, name='dec_lr')])
        obj_cat.add_columns(
            [Column(data=ra_hr, name='ra_hr'), Column(data=dec_hr, name='dec_hr')])

    # Reorder columns
    colnames = obj_cat.colnames
    for item in ['dec', 'ra', 'y', 'x', 'index']:
        if item in colnames:
            colnames.remove(item)
            colnames.insert(0, item)
    obj_cat = obj_cat[colnames]
    obj_cat.add_column(
        Column(data=[None] * len(obj_cat), name='obj_type'), index=0)

    # if len(datas) == 1:
    #     bg_rms = mad_wavelet(detect)
    # else:
    #     bg_rms = []
    #     for data in datas:
    #         bg_rms.append(mad_wavelet(detect))

    if match_gaia:
        obj_cat.add_column(
            Column(data=[None] * len(obj_cat), name='gaia_coord'))
        if len(datas) == 1:
            w = datas[0].wcs
            pixel_scale = w.to_header()['PC2_2'] * 3600
        else:
            w = data_hr.wcs
            pixel_scale = w.to_header()['PC2_2'] * 3600

        # Retrieve GAIA catalog
        if tigress:
            gaia_stars = _image_gaia_stars_tigress(
                detect, w, pixel_scale=pixel_scale,
                verbose=True, visual=visual_gaia, logger=logger)
        else:
            gaia_stars = image_gaia_stars(
                detect, w, pixel_scale=pixel_scale,
                verbose=True, visual=visual_gaia)

        # Cross-match with SExtractor catalog
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        temp, dist, _ = match_coordinates_sky(SkyCoord(ra=gaia_stars['ra'], dec=gaia_stars['dec'], unit='deg'),
                                              SkyCoord(ra=obj_cat['ra'], dec=obj_cat['dec'], unit='deg'), nthneighbor=1)
        flag = dist < 5 * u.arcsec
        star_mag = gaia_stars['phot_g_mean_mag'].data
        psf_ind = temp[flag]
        star_mag = star_mag[flag]
        bright_star_flag = star_mag < 19.0
        obj_cat['obj_type'][psf_ind[bright_star_flag]
                            ] = scarlet.source.ExtendedSource
        obj_cat['obj_type'][psf_ind[~bright_star_flag]
                            ] = scarlet.source.PointSource
        # we also use the coordinates from Gaia for bright stars
        obj_cat['gaia_coord'][psf_ind] = np.array(
            gaia_stars[['ra', 'dec']])[flag]

        # Cross-match for a second time: to deal with splitted bright stars
        temp_cat = obj_cat.copy(copy_data=True)
        temp_cat.remove_rows(psf_ind)
        temp2, dist2, _ = match_coordinates_sky(SkyCoord(ra=gaia_stars['ra'], dec=gaia_stars['dec'], unit='deg'),
                                                SkyCoord(ra=temp_cat['ra'], dec=temp_cat['dec'], unit='deg'), nthneighbor=1)
        flag2 = dist2 < 1 * u.arcsec
        psf_ind2 = temp_cat[temp2[flag2]]['index'].data
        # we also use the coordinates from Gaia for bright stars
        obj_cat.remove_rows(psf_ind2)
        # obj_cat['gaia_coord'][psf_ind2] = np.array(gaia_stars[['ra', 'dec']])[flag2]
        # obj_cat['obj_type'][psf_ind2] = scarlet.source.PointSource
        if logger:
            logger.info(f'    Matched {len(psf_ind)} stars from GAIA')
        print(f'    Matched {len(psf_ind)} stars from GAIA')

    obj_cat['index'] = np.arange(len(obj_cat))

    # Visualize the results
    if show_fig and match_gaia:
        from matplotlib.patches import Ellipse as mpl_ellip
        from .display import ORG, GRN

        fig = result[2]
        ax1 = fig.get_axes()[0]
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        # Plot an ellipse for each object
        for star in gaia_stars[flag]:
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
            gaia_stars['x_pix'],
            gaia_stars['y_pix'],
            color=GRN(1.0),
            s=100,
            alpha=0.9,
            marker='+')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

    return obj_cat, segmap, 0  # bg_rms
