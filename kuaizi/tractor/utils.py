from __future__ import division, print_function

import numpy as np

from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

from ..display import display_single, IMG_CMAP, SEG_CMAP
import kuaizi


import os
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def makeCatalog(datas, layer_ind=None, mask=None, lvl=3, method='wavelet', convolve=False, conv_radius=5,
                match_gaia=True, show_fig=True, visual_gaia=True, **kwargs):
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
    from kuaizi.detection import vanilla_detection, wavelet_detection

    if layer_ind is not None:
        detect_image = datas[0].images[layer_ind]
    else:
        hr_images = datas[0].images / \
            np.sum(datas[0].images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)

    if np.size(detect_image.shape) == 3:
        detect = detect_image.mean(axis=0)
    else:
        detect = detect_image

    if convolve:
        from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel
        detect = convolve(detect.astype(float), Gaussian2DKernel(conv_radius))

    if method == 'wavelet':
        result = wavelet_detection(
            detect, mask=mask, sigma=lvl, show_fig=show_fig, **kwargs)
    else:
        result = vanilla_detection(
            detect, mask=mask, sigma=lvl, show_fig=show_fig, **kwargs)

    obj_cat = result[0]
    segmap = result[1]

    ## RA and Dec
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
            [Column(data=ra_hr, name='ra_hr'), Column(data=dec_lr, name='dec_hr')])

    # Reorder columns
    colnames = obj_cat.colnames
    for item in ['dec', 'ra', 'y', 'x', 'index']:
        if item in colnames:
            colnames.remove(item)
            colnames.insert(0, item)
    obj_cat = obj_cat[colnames]
    obj_cat.add_column(
        Column(data=[None] * len(obj_cat), name='obj_type'), index=0)

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
        #obj_cat['gaia_coord'][psf_ind2] = np.array(gaia_stars[['ra', 'dec']])[flag2]
        #obj_cat['obj_type'][psf_ind2] = scarlet.source.PointSource
        print(f'# Matched {len(psf_ind)} stars from GAIA')

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

    return obj_cat, segmap


#########################################################################
########################## The Tractor related ##########################
#########################################################################


# Add sources to tractor
def add_tractor_sources(obj_cat, sources, w, shape_method='manual', band='r'):
    '''
    Add tractor sources to the sources list. SHOULD FOLLOW THE ORDER OF OBJECTS IN THE INPUT CATALOG.

    Parameters:
    ----------
    obj_cat: astropy Table, objects catalogue.
    sources: list, to which we will add objects.
    w: wcs object.
    shape_method: string, 'manual' or 'decals' or 'hsc'. 
        If 'manual', it will adopt the manually measured shapes. 
        If 'decals', it will adopt shapes in 'DECaLS' tractor catalog.
        If 'hsc', it will adopt shapes in HSC CModel catalog.

    Returns:
    --------
    sources: list of sources.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE
    from tractor.sersic import SersicGalaxy, SersicIndex

    # if shape_method is 'manual' or 'decals':
    obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))
    # comp_galaxy = obj_cat[obj_type == 'COMP']
    # dev_galaxy = obj_cat[obj_type == 'DEV']
    # exp_galaxy = obj_cat[obj_type == 'EXP']
    # rex_galaxy = obj_cat[obj_type == 'REX']
    # ser_galaxy = obj_cat[obj_type == 'SER']
    # psf_galaxy = obj_cat[np.logical_or(obj_type == 'PSF', obj_type == '   ')]

    # elif shape_method is 'hsc':
    #     star_mask = obj_cat['{}_extendedness'.format(band)] < 0.5
    #     psf_galaxy = obj_cat[star_mask]

    #     fracdev = obj_cat['cmodel_fracdev'].values
    #     dev_galaxy = obj_cat[(fracdev >= 0.5) & (~star_mask)]
    #     exp_galaxy = obj_cat[(fracdev < 0.5) & (~star_mask)]
    # else:
    #     raise ValueError('Only "manual", "decals", or "hsc" is supported now.')

    if shape_method is 'manual':
        # Using manually measured shapes
        if sources is None:
            sources = []

        for i, obj in enumerate(obj_cat):
            if obj_type[i] == 'COMP':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    CompositeGalaxy(
                        PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                        GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
                                    90.0 + obj['theta'] * 180.0 / np.pi),
                        Flux(0.6 * obj['flux']),
                        GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
                                    90.0 + obj['theta'] * 180.0 / np.pi)))

            elif obj_type[i] == 'DEV':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    DevGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                    (90.0 + obj['theta'] * 180.0 / np.pi))))

            elif obj_type[i] == 'EXP':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    ExpGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                    (90.0 + obj['theta'] * 180.0 / np.pi))))
            elif obj_type[i] == 'SER':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    SersicGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                    (90.0 + obj['theta'] * 180.0 / np.pi)),
                        SersicIndex(1.0)
                    )
                )
            elif obj_type[i] == 'REX':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['a_arcsec'], 0, 0)
                )
                src.shape.freezeParam('e1')
                src.shape.freezeParam('e2')
                sources.append(src)
            else:
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(PointSource(
                    PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print(" - Now you have %d sources" % len(sources))

    elif shape_method is 'decals':
        # Using DECaLS shapes
        if sources is None:
            sources = []

        for i, obj in enumerate(obj_cat):
            if obj_type[i] == 'COMP':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    CompositeGalaxy(
                        PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                        EllipseE(obj['shape_r'], obj['shape_e1'],
                                 obj['shape_e2']), Flux(0.6 * obj['flux']),
                        EllipseE(obj['shape_r'], obj['shape_e1'],
                                 obj['shape_e2'])))
            elif obj_type[i] == 'DEV':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    DevGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        EllipseE(obj['shape_r'], obj['shape_e1'],
                                 -obj['shape_e2'])))
            elif obj_type[i] == 'EXP':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    ExpGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        EllipseE(obj['shape_r'], obj['shape_e1'],
                                 -obj['shape_e2'])))
            elif obj_type[i] == 'REX':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shape_r'], 0, 0)
                )
                src.shape.freezeParam('e1')
                src.shape.freezeParam('e2')
                sources.append(src)
            elif obj_type[i] == 'SER':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(
                    SersicGalaxy(
                        PixPos(pos_x, pos_y), Flux(obj['flux']),
                        EllipseE(obj['shape_r'], obj['shape_e1'],
                                 -obj['shape_e2']),
                        SersicIndex(1.0)
                    )
                )
            else:
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                sources.append(PointSource(
                    PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print(" - Now you have %d sources" % len(sources))

    elif shape_method is 'hsc':
        from unagi import catalog
        # Using HSC CModel catalog
        if sources is None:
            sources = []
        for obj in obj_cat:
            if obj_type[i] == 'SER':
                pos_x, pos_y = obj['x'], obj['y']
                flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
                r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                    obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
                    to_pixel=False, update=False)  # arcsec, degree
                sources.append(
                    SersicGalaxy(
                        PixPos(pos_x, pos_y), Flux(flux),
                        GalaxyShape(r_gal, ba_gal, pa_gal + 90),
                        SersicIndex(1.0)
                    )
                )
            elif obj_type[i] == 'DEV':
                pos_x, pos_y = obj['x'], obj['y']
                flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
                r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                    obj, shape_type='cmodel_dev_ellipse', axis_ratio=True,
                    to_pixel=False, update=False)  # arcsec, degree
                sources.append(
                    DevGalaxy(
                        PixPos(pos_x, pos_y), Flux(flux),
                        GalaxyShape(r_gal, ba_gal, pa_gal + 90),
                    )
                )
            elif obj_type[i] == 'EXP':
                pos_x, pos_y = obj['x'], obj['y']
                flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
                r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                    obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
                    to_pixel=False, update=False)  # arcsec, degree
                sources.append(
                    ExpGalaxy(
                        PixPos(pos_x, pos_y), Flux(flux),
                        GalaxyShape(r_gal, ba_gal, pa_gal + 90)
                    )
                )
            elif obj_type[i] == 'REX':
                pos_x, pos_y = obj['x'], obj['y']
                flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
                r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                    obj, shape_type='cmodel_ellipse', axis_ratio=True,
                    to_pixel=False, update=False)  # arcsec, degree
                src = ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(flux),
                    GalaxyShape(r_gal, 1, 0)
                )
                src.shape.freezeParam('ab')
                src.shape.freezeParam('phi')
                sources.append(src)
            else:
                pos_x, pos_y = obj['x'], obj['y']
                flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)

                sources.append(PointSource(PixPos(pos_x, pos_y), Flux(flux)))

        print(" - Now you have %d sources" % len(sources))
    else:
        raise ValueError('Cannot use this shape method')
    return sources

# Do tractor iteration


def tractor_iteration(obj_cat, w, img_data, invvar, psf_obj, pixel_scale,
                      shape_method='manual', freeze_dict=None, ref_source=None,
                      kfold=4, first_num=50, band_name=None,
                      fig_name=None, verbose=False):
    '''
    Run tractor iteratively.

    Parameters:
    -----------
    obj_cat (astropy.table.Table): objects catalogue.
    w (astropy.wcs.WCS): wcs object.
    img_data (numpy 2-D array): image of a certain band.
    invvar (numpy 2-D array): inverse variance matrix of the input image.
    psf_obj (tractor.psf.PixelizedPSF): PSF object, defined by tractor.psf.PixelizedPSF() class.
    pixel_scale (float): pixel scale in the unit of arcsec/pixel.
    shape_method (str): if 'manual', then adopt manually measured shape. 
        If 'decals', then adopt DECaLS shape from tractor files. If 'cmodel', use HSC CModel catalog of S18A.
    freeze_dict (dict): for the target galaxy, whether freeze position, shape, flux, Sersic index, etc. 
        Example `freeze_dict` is like: `{'pos': True, 'shape': False, 'sersicindex': False}`.
    ref_source (tractor source): Tractor source as a reference when doing forced photometry. 
        The target galaxy in other bands all takes parameters from this `ref_source`.
    kfold (int): how many iterations you want to run.
    first_num (int): number of sources in the first run. 
    band_name (str): name of the filter.
    fig_name (str): if not None, it will save the tractor subtracted image to the given path.
    verbose (bool): if true, it will print out everything...

    Returns:
    -----------
    sources (list): a list of sources modeled by `tractor`.
    trac_obj: optimized tractor object.
    fig (matplotlib.pyplot.figure): figure showing optimized model.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE

    if len(obj_cat) < 1:
        raise ValueError(
            "The length of `obj_cat` is less than 1. Please check your catalog!")
    elif len(obj_cat) == 1:
        # when there's only one object, you don't need to run Tractor for several rounds.
        kfold = 1
    else:
        step = int((len(obj_cat) - first_num) / (kfold - 1))

    if freeze_dict is None:
        freeze_dict = {}

    if 'target' in obj_cat.colnames:
        target_index = np.where(obj_cat['target'] == 1)[0][0]
    else:
        target_index = 0  # regard the first object as target

    for i in range(kfold):
        if i == 0:
            obj_small_cat = obj_cat[:first_num]
            sources = add_tractor_sources(
                obj_small_cat, None, w, shape_method=shape_method)
        else:
            obj_small_cat = obj_cat[first_num +
                                    step * (i - 1): first_num + step * (i)]
            sources = add_tractor_sources(
                obj_small_cat, sources, w, shape_method=shape_method)

        # our target galaxy is now included in `sources`
        if target_index < len(sources):
            if ref_source is not None:
                sources[target_index] = ref_source

            cen_src = sources[target_index]
            [cen_src.freezeParam(key) for key in freeze_dict if freeze_dict[key]
             is True and key in cen_src.namedparams]
            # `src.namedparams` indicates the name of available parameters for this source
            # We may only want to freeze the params for the target galaxy??? Don't need to freeze for all.
            print('Target source:', sources[target_index])

        with HiddenPrints():  # suppress annoying messages
            tim = Image(data=img_data,
                        invvar=invvar,
                        psf=psf_obj,
                        wcs=NullWCS(pixscale=pixel_scale),
                        sky=ConstantSky(0.0),
                        photocal=NullPhotoCal()
                        )
            trac_obj = Tractor([tim], sources)
            trac_obj.freezeParam('images')
            trac_obj.optimize_loop()

        ########################
        plt.rc('font', size=20)
        if i % 2 == 1 or i == (kfold - 1):
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 8))

            with HiddenPrints():
                trac_mod_opt = trac_obj.getModelImage(
                    0, minsb=0., srcs=sources[:])

            if band_name is None:
                _ = kuaizi.display.display_multiple(
                    [img_data, trac_mod_opt, img_data - trac_mod_opt],
                    text=['raw\ image', 'tractor\ model', 'residual'],
                    ax=[ax1, ax2, ax3], scale_bar_y_offset=0.4, text_fontsize=20)
            else:
                _ = kuaizi.display.display_multiple(
                    [img_data, trac_mod_opt, img_data - trac_mod_opt],
                    text=[f'{band_name}-band\ raw\ image',
                          'tractor\ model', 'residual'],
                    ax=[ax1, ax2, ax3], scale_bar_y_offset=0.4, text_fontsize=20)

            # ax1 = display_single(img_data, ax=ax1, scale_bar=False)
            # if band_name is not None:
            #     ax1.set_title(f'{band_name}-band raw image')
            # else:
            #     ax1.set_title('raw image')
            # ax2 = display_single(trac_mod_opt, ax=ax2, scale_bar=False, contrast=0.02)
            # ax2.set_title('tractor model')
            # ax3 = display_single(abs(img_data - trac_mod_opt), ax=ax3, scale_bar=False, color_bar=True, contrast=0.05)
            # ax3.set_title('residual')

            with HiddenPrints():
                chi2 = (trac_obj.getChiImage()**2).mean()

            if i == (kfold - 1):
                if fig_name is not None:
                    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
                    plt.show()
                    print('   The chi-square is', chi2)
                    # print('   The chi-square is', np.sqrt(
                    #     np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data))
            elif verbose:
                plt.show()
                print('   The chi-square is', chi2)
            else:
                plt.close()
                print('   The chi-square is', chi2)

        #trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[1:])
        #ax4 = display_single(img_data - trac_mod_opt, ax=ax4, scale_bar=False, color_bar=True, contrast=0.05)
        #ax4.set_title('remain central galaxy')

    return sources, trac_obj, fig

# Do tractor fitting in a blob-by-blob way


def tractor_blob_by_blob(obj_cat, w, img_data, invvar, psf_obj, pixel_scale,
                         shape_method='manual', freeze_dict=None, ref_source=None,
                         band_name=None, show_fig=False, fig_name=None, verbose=False):
    '''
    Run tractor in a blob-by-blob way. See "Goodness-of-Fits and Morphological type" in https://www.legacysurvey.org/dr9/catalogs/#id9.

    Parameters:
    -----------
    obj_cat (astropy.table.Table): objects catalogue.
    w (astropy.wcs.WCS): wcs object.
    img_data (numpy 2-D array): image of a certain band.
    invvar (numpy 2-D array): inverse variance matrix of the input image.
    psf_obj (tractor.psf.PixelizedPSF): PSF object, defined by tractor.psf.PixelizedPSF() class.
    pixel_scale (float): pixel scale in the unit of arcsec/pixel.
    shape_method (str): if 'manual', then adopt manually measured shape. 
        If 'decals', then adopt DECaLS shape from tractor files. If 'cmodel', use HSC CModel catalog of S18A.
    freeze_dict (dict): for the target galaxy, whether freeze position, shape, flux, Sersic index, etc. 
        Example `freeze_dict` is like: `{'pos': True, 'shape': False, 'sersicindex': False}`.
    ref_source (tractor source): Tractor source as a reference when doing forced photometry. 
        The target galaxy in other bands all takes parameters from this `ref_source`.  
    band_name (str): name of the filter
    fig_name (str): if not None, it will save the tractor subtracted image to the given path.
    verbose (bool): if true, it will print out everything...

    Returns:
    -----------
    blob_sources (list): a list of sources modeled by `tractor`.
    trac_obj: optimized tractor object.
    fig (matplotlib.pyplot.figure): figure showing optimized model.

    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE
    import copy

    if len(obj_cat) < 1:
        raise ValueError(
            "The length of `obj_cat` is less than 1. Please check your catalog!")

    if freeze_dict is None:
        freeze_dict = {}

    blob_sources = []

    dchisq = []
    type_list = ['PSF', 'REX', 'DEV', 'EXP', 'SER']

    image = copy.deepcopy(img_data)

    for i, obj in enumerate(obj_cat):
        with HiddenPrints():
            tim = Image(data=image,
                        invvar=invvar,
                        psf=psf_obj,
                        wcs=NullWCS(pixscale=pixel_scale),
                        sky=ConstantSky(0.0),
                        photocal=NullPhotoCal()
                        )
            trac_obj = Tractor([tim], blob_sources)
            chi_0 = (trac_obj.getChiImage()**2).mean()

            chi_diff = []
            blobs = []
            # enable to iterate over types
            for Type in type_list:
                obj_temp = copy.deepcopy(obj)
                obj_temp['type'] = Type
                src = add_tractor_sources(
                    Table(obj_temp), None, w, shape_method=shape_method)
                # src[0].freezeParam('pos')
                # Free parameter for our target galaxy
                if obj['target'] == 1 and ref_source is not None:
                    src = [copy.deepcopy(ref_source)]
                    src[0].brightness = Flux(obj_temp['flux'])
                if obj['target'] == 1:
                    for item in freeze_dict:
                        if item == 'shape.ab' and freeze_dict[item]:
                            try:
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('ab')
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('e1')
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('e2')
                            except:
                                pass
                        if item == 'shape.phi' and freeze_dict[item]:
                            try:
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('phi')
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('e1')
                                src[0][src[0].getNamedParamIndex(
                                    'shape')].freezeParam('e2')
                            except:
                                pass
                        if item == 'shape.re' and freeze_dict[item]:
                            src[0][src[0].getNamedParamIndex(
                                'shape')].freezeParam('re')
                        if item == 'shape' and freeze_dict[item] and item in src[0].namedparams:
                            src[0].freezeParam(item)
                        if item in ['pos', 'sersicindex'] and freeze_dict[item] and item in src[0].namedparams:
                            src[0].freezeParam(item)
                    #[src[0].freezeParam(key) for key in freeze_dict if key in src[0].namedparams and freeze_dict[key] is True]
                trac_obj = Tractor([tim], src)
                trac_obj.freezeParam('images')
                try:
                    trac_obj.optimize_loop()
                    # object on the edge doesn't work
                    chi_1 = (trac_obj.getChiImage()**2).mean()
                except:
                    chi_1 = 99
                chi_diff.append(chi_0 - chi_1)
                blobs.append(src[0])

            dchisq.append(chi_diff)
            # use the object with smallest chi2
            blob_sources.append(blobs[np.nanargmax(chi_diff)])

            # subtract optimized object from image
            trac_mod_opt = trac_obj.getModelImage(0, minsb=0.)
            image -= trac_mod_opt

        if verbose:
            print(f'# Progress: {i+1} / {len(obj_cat)}')

    # ########################################
    # # Another round of global optimization #
    tim = Image(data=img_data,
                invvar=invvar,
                psf=psf_obj,
                wcs=NullWCS(pixscale=pixel_scale),
                sky=ConstantSky(0.0),
                photocal=NullPhotoCal()
                )
    trac_obj = Tractor([tim], blob_sources)
    trac_obj.freezeParam('images')
    with HiddenPrints():
        chi2_0 = (trac_obj.getChiImage()**2).mean()
        try:
            trac_obj.optimize_loop()
        except:
            pass
        chi2_1 = (trac_obj.getChiImage()**2).mean()
    if verbose:
        print('# Global optimization: Chi2 improvement = ', chi2_0 - chi2_1)

    ########################
    # Plot the residual #
    tim = Image(data=img_data,
                invvar=invvar,
                psf=psf_obj,
                wcs=NullWCS(pixscale=pixel_scale),
                sky=ConstantSky(0.0),
                photocal=NullPhotoCal()
                )
    trac_obj = Tractor([tim], blob_sources)
    with HiddenPrints():
        chi2 = (trac_obj.getChiImage()**2).mean()

    if show_fig:
        plt.rc('font', size=20)
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 8))

        with HiddenPrints():
            trac_mod_opt = trac_obj.getModelImage(
                0, minsb=0.)

        if band_name is None:
            _ = kuaizi.display.display_multiple(
                [img_data, trac_mod_opt, img_data - trac_mod_opt],
                text=['raw\ image', 'tractor\ model', 'residual'],
                ax=[ax1, ax2, ax3], scale_bar_y_offset=0.4, text_fontsize=20)
        else:
            _ = kuaizi.display.display_multiple(
                [img_data, trac_mod_opt, img_data - trac_mod_opt],
                text=[f'{band_name}-band\ raw\ image',
                      'tractor\ model', 'residual'],
                ax=[ax1, ax2, ax3], scale_bar_y_offset=0.4, text_fontsize=20)

        if fig_name is not None:
            plt.savefig(fig_name, dpi=200, bbox_inches='tight')
            plt.show()
            print('   The chi-square is', chi2)
            # print('   The chi-square is', np.sqrt(
            #     np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data))
        elif verbose:
            plt.show()
            print('   The chi-square is', chi2)
        else:
            plt.close()
            print('   The chi-square is', chi2)
    else:
        print('   The chi-square is', chi2)

    if show_fig:
        return blob_sources, trac_obj, fig
    else:
        return blob_sources, trac_obj


def tractor_hsc_sep_blob_by_blob(obj, filt, channels, data,
                                 freeze_dict=None, ref_source=None,
                                 show_fig=False, verbose=False):
    '''
    Run `the tractor` on HSC images, for Merian survey.
    BLOB-BY-BLOB way! Based on manual measurement. 

    Parameters:
        obj: a row in object catalog. Should contain 'ra', 'dec', 'name'. 
        filt (str): filter name, such as 'r'.
        channels (str): all filters, such as 'grizy'.
        data (kuaizi.detection.Data): an data structure which contains images, weights, wcs, PSFs, etc.
        freeze_dict (dict): for the target galaxy, whether freeze position, shape, flux, Sersic index, etc. 
            Example `freeze_dict` is like: `{'pos': True, 'shape': False, 'sersicindex': False}`.
        ref_source (tractor source): Tractor source as a reference when doing forced photometry. 
            The target galaxy in other bands all takes parameters from this `ref_source`.  
        verbose (bool): whether being verbose.

    Return: 
        trac_obj: tractor objects. Use `trac_obj.catalog.subs` to get models of sources, 
            use `trac_obj.getModelImage(0, minsb=0., srcs=sources[:])` to render the model image. 
    '''
    from kuaizi import HSC_pixel_scale
    from tractor.psf import PixelizedPSF

    obj_name = obj['name'].rstrip('_y')
    coord = SkyCoord(obj['ra'], obj['dec'], frame='icrs', unit='deg')

    if verbose:
        print("### `" + obj_name + f"` {filt}-band")
    layer_ind = channels.index(filt)

    obj_cat_sep, segmap_sep = makeCatalog(
        [data],
        layer_ind=layer_ind,
        lvl=2.0,
        mask=None,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=False,
        visual_gaia=False,
        b=32,
        f=3,
        pixel_scale=0.168,
        minarea=5,
        deblend_nthresh=48,
        deblend_cont=0.001,
        sky_subtract=True,
        verbose=verbose)

    obj_cat_sep['a_arcsec'] = obj_cat_sep['a'] * HSC_pixel_scale
    obj_cat_sep['b_arcsec'] = obj_cat_sep['b'] * HSC_pixel_scale
    obj_cat_sep['type'] = np.repeat(b'NAN', len(obj_cat_sep))

    obj_cat_sep.sort('flux', reverse=True)

    catalog_c = SkyCoord(obj_cat_sep['ra'], obj_cat_sep['dec'], unit='deg')
    dist = coord.separation(catalog_c)
    cen_obj_ind = np.argsort(dist)[0]
    cen_obj = obj_cat_sep[cen_obj_ind]
    ### Add a "target" column to indicate which object is the target galaxy ###
    obj_cat_sep['target'] = np.zeros(len(obj_cat_sep), dtype=int)
    obj_cat_sep['target'][cen_obj_ind] = 1

    # print(f'# Type of the central object is {cen_obj["type"]}')
    if verbose:
        print(f'# Total number of objects: {len(obj_cat_sep)}')
        print(f'# Central object index in {filt}-band: {cen_obj_ind}')

    psf_obj = PixelizedPSF(data.psfs[layer_ind])  # Construct PSF

    result = tractor_blob_by_blob(
        obj_cat_sep,
        data.wcs,
        data.images[layer_ind],
        data.weights[layer_ind],
        psf_obj,
        HSC_pixel_scale,
        shape_method='manual',
        freeze_dict=freeze_dict,
        ref_source=ref_source,
        show_fig=show_fig,
        fig_name=obj_name + '_sep_tractor_' + filt,
        verbose=verbose)

    if not show_fig:
        sources, trac_obj = result
    else:
        sources, trac_obj, fig = result
    trac_obj.target_ind = cen_obj_ind  # record the index of target galaxy
    if verbose:
        print(trac_obj.catalog[cen_obj_ind])
    return trac_obj


def _compute_invvars(allderivs):
    ivs = []
    for derivs in allderivs:
        chisq = 0
        for deriv, tim in derivs:
            h, w = tim.shape
            deriv.clipTo(w, h)
            ie = tim.getInvError()
            slc = deriv.getSlice(ie)
            chi = deriv.patch * ie[slc]
            chisq += (chi**2).sum()
        ivs.append(chisq)
    return ivs


def _regularize_attr(item):
    item = item.replace('pos.x', 'x')
    item = item.replace('pos.y', 'y')
    item = item.replace('brightness.Flux', 'flux')
    item = item.replace('shape.re', 're')
    item = item.replace('shape.ab', 'ab')
    item = item.replace('shape.phi', 'phi')
    item = item.replace('sersicindex.SersicIndex', 'sersic')
    return item


def getTargetProperty(trac_obj, wcs=None, pixel_scale=kuaizi.HSC_pixel_scale, zeropoint=kuaizi.HSC_zeropoint):
    '''
    Write the properties of our target galaxy (only) into a dictionary. 
    Note: this only gives you the properties in one band.

    Paarameters:
        trac_obj: should contain `target_ind`.
        wcs: wcs object of the input image. 
    Returns:
        source_output (dict): contains many attributes of the target galaxy in `trac_obj`. 
            Flux is in nanomaggy, effective radius (`re`) is in arcsec.

    '''
    trac_obj.thawAllRecursive()

    attri = trac_obj.catalog.getParamNames()
    attri = [_regularize_attr(item) for item in attri]

    values = dict(zip(attri, trac_obj.catalog.getParams()))
    with HiddenPrints():
        all_derivs = trac_obj.getDerivs()
        # first derivative is for sky bkg
        invvars = dict(zip(attri, _compute_invvars(all_derivs)[1:]))

    i = trac_obj.target_ind  # only extract information of our target galaxy
    src = trac_obj.catalog[i]

    keys_to_extract = [item for item in attri if f'source{i}.' in item]
    source_values = {key.replace(f'source{i}.', ''): values[key] for key in keys_to_extract}
    source_values['flux'] *= 10**((22.5 - zeropoint) / 2.5)  # in nanomaggy

    source_invvar = {key.replace(
        f'source{i}.', '') + '_ivar': invvars[key] for key in keys_to_extract}
    # in nanomaggy^-2
    source_invvar['flux_ivar'] *= 10**(- 2 * (22.5 - zeropoint) / 2.5)

    source_output = dict(**source_values, **source_invvar)

    if not 'sersic' in source_output.keys():  # doesn't have sersic index, need to assign according to its type
        if 'dev' in src.getSourceType().lower():
            source_output['sersic'] = 4.0
            source_output['sersic_ivar'] = 0.0
            source_output['type'] = 'DEV'

        if 'exp' in src.getSourceType().lower():
            source_output['sersic'] = 1.0
            source_output['sersic_ivar'] = 0.0
            rex_flag = (src.shape.getName() ==
                        'EllipseE' and src.shape.e1 == 0 and src.shape.e2 == 0)
            rex_flag |= (src.shape.getName() ==
                         'Galaxy Shape' and src.shape.ab == 1)

            if rex_flag:
                # Round exponential
                source_output['ab'] = 1.0
                source_output['phi'] = 0.0
                source_output['ab_ivar'] = 0.0
                source_output['phi_ivar'] = 0.0
                source_output['type'] = 'REX'
            else:
                source_output['type'] = 'EXP'

        if 'pointsource' in src.getSourceType().lower():
            source_output['sersic'] = 0.0
            source_output['sersic_ivar'] = 0.0
            source_output['type'] = 'PSF'
    else:
        source_output['type'] = 'SER'

    ## RA, DEC ##
    if wcs is not None:
        ra, dec = wcs.wcs_pix2world(source_output['x'], source_output['y'], 0)
        # Here we assume the WCS is regular and has no distortion!
        dec_ivar = source_output['y_ivar'] / (pixel_scale / 3600)**2
        ra_ivar = source_output['x_ivar'] / \
            (pixel_scale / 3600)**2 / np.cos(np.deg2rad(dec))
        source_output['ra'] = float(ra)
        source_output['dec'] = float(dec)
        source_output['ra_ivar'] = float(ra_ivar)
        source_output['dec_ivar'] = float(dec_ivar)

    # Pixel to arcsec
    if 're' in source_output.keys():
        source_output['re'] *= pixel_scale
        source_output['re_ivar'] /= pixel_scale**2

    return source_output


def _write_to_row(row, model_dict, channels=list('grizy')):
    '''
    Write the output of `makeMeasurement` to a Row of astropy.table.Table.

    Parameters:
        row (astropy.table.Row): one row of the table.
        measurement (dict): the output dictionary of `kuaizi.measure.makeMeasurement`

    Returns:
        row (astropy.table.Row)
    '''
    meas_dict = getTargetProperty(model_dict['i'])
    for key in ['x', 'x_ivar', 'y', 'y_ivar', 'ab', 'ab_ivar', 'phi', 'phi_ivar', 'sersic', 'sersic_ivar']:
        row[key] = meas_dict[key]

    # flux
    flux = np.zeros(len(channels))
    flux_ivar = np.zeros(len(channels))
    for i, filt in enumerate(channels):
        meas_dict = getTargetProperty(model_dict[filt])
        flux[i] = meas_dict['flux']
        flux_ivar[i] = meas_dict['flux_ivar']

    row['flux'] = flux
    row['flux_ivar'] = flux_ivar
    return row


def initialize_meas_cat(obj_cat, channels=list('grizy')):
    length = len(obj_cat)
    bands = len(channels)

    meas_cat = Table([
        Column(name='ID', length=length, dtype=int),
        Column(name='x', length=length, dtype=float),
        Column(name='x_ivar', length=length, dtype=float),
        Column(name='y', length=length, dtype=float),
        Column(name='y_ivar', length=length, dtype=float),
        Column(name='flux', length=length, shape=(bands,)),
        Column(name='flux_ivar', length=length, shape=(bands,)),
        Column(name='ab', length=length, dtype=float),
        Column(name='ab_ivar', length=length, dtype=float),
        Column(name='phi', length=length, dtype=float),
        Column(name='phi_ivar', length=length, dtype=float),
        Column(name='sersic', length=length, dtype=float),
        Column(name='sersic_ivar', length=length, dtype=float),
    ])
    # for col in meas_cat.columns:
    #     meas_cat[col] = np.nan * meas_cat[col]

    return meas_cat


# def _add_tractor_sources(obj_cat, sources, w, shape_method='manual', band='r'):
#     '''
#     DEPRECATED!!!

#     Add tractor sources to the sources list. SHOULD FOLLOW THE ORDER OF OBJECTS IN THE INPUT CATALOG.

#     Parameters:
#     ----------
#     obj_cat: astropy Table, objects catalogue.
#     sources: list, to which we will add objects.
#     w: wcs object.
#     shape_method: string, 'manual' or 'decals' or 'hsc'.
#         If 'manual', it will adopt the manually measured shapes.
#         If 'decals', it will adopt shapes in 'DECaLS' tractor catalog.
#         If 'hsc', it will adopt shapes in HSC CModel catalog.

#     Returns:
#     --------
#     sources: list of sources.
#     '''
#     from tractor import NullWCS, NullPhotoCal, ConstantSky
#     from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
#     from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
#     from tractor.ellipses import EllipseE
#     from tractor.sersic import SersicGalaxy, SersicIndex

#     # if shape_method is 'manual' or 'decals':
#     obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))
#     comp_galaxy = obj_cat[obj_type == 'COMP']
#     dev_galaxy = obj_cat[obj_type == 'DEV']
#     exp_galaxy = obj_cat[obj_type == 'EXP']
#     rex_galaxy = obj_cat[obj_type == 'REX']
#     ser_galaxy = obj_cat[obj_type == 'SER']
#     psf_galaxy = obj_cat[np.logical_or(obj_type == 'PSF', obj_type == '   ')]

#     # elif shape_method is 'hsc':
#     #     star_mask = obj_cat['{}_extendedness'.format(band)] < 0.5
#     #     psf_galaxy = obj_cat[star_mask]

#     #     fracdev = obj_cat['cmodel_fracdev'].values
#     #     dev_galaxy = obj_cat[(fracdev >= 0.5) & (~star_mask)]
#     #     exp_galaxy = obj_cat[(fracdev < 0.5) & (~star_mask)]
#     # else:
#     #     raise ValueError('Only "manual", "decals", or "hsc" is supported now.')

#     if shape_method is 'manual':
#         # Using manually measured shapes
#         if sources is None:
#             sources = []

#         for obj in comp_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 CompositeGalaxy(
#                     PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
#                     GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
#                                 90.0 + obj['theta'] * 180.0 / np.pi),
#                     Flux(0.6 * obj['flux']),
#                     GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
#                                 90.0 + obj['theta'] * 180.0 / np.pi)))
#         for obj in dev_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 DevGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#                                 (90.0 + obj['theta'] * 180.0 / np.pi))))
#         for obj in exp_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 ExpGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#                                 (90.0 + obj['theta'] * 180.0 / np.pi))))
#         for obj in ser_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 SersicGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#                                 (90.0 + obj['theta'] * 180.0 / np.pi)),
#                     SersicIndex(2.0)
#                 )
#             )
#         for obj in rex_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 ExpGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     GalaxyShape(obj['a_arcsec'], 1, 0)))

#         for obj in psf_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(PointSource(
#                 PixPos(pos_x, pos_y), Flux(obj['flux'])))

#         # for obj in obj_cat:
#         #     pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#         #     if obj['type'].rstrip(' ') == 'COMP':
#         #         sources.append(
#         #             CompositeGalaxy(
#         #                 PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
#         #                             90.0 + obj['theta'] * 180.0 / np.pi),
#         #                 Flux(0.6 * obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
#         #                             90.0 + obj['theta'] * 180.0 / np.pi)))
#         #     elif obj['type'].rstrip(' ') == 'DEV':
#         #         sources.append(
#         #             DevGalaxy(
#         #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#         #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
#         #     elif obj['type'].rstrip(' ') == 'EXP':
#         #         sources.append(
#         #             ExpGalaxy(
#         #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#         #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
#         #     elif obj['type'].rstrip(' ') == 'SER':
#         #         sources.append(
#         #             SersicGalaxy(
#         #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#         #                             (90.0 + obj['theta'] * 180.0 / np.pi)),
#         #                 SersicIndex(2.0)
#         #                 )
#         #         )
#         #     elif obj['type'].rstrip(' ') == 'REX':
#         #         sources.append(
#         #             ExpGalaxy(
#         #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
#         #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
#         #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
#         #     elif obj['type'].rstrip(' ') == 'PSF' or obj['type'].rstrip(' ') == '   ':
#         #         sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

#         print(" - Now you have %d sources" % len(sources))

#     elif shape_method is 'decals':
#         # Using DECaLS shapes
#         if sources is None:
#             sources = []
#         for obj in comp_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 CompositeGalaxy(
#                     PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
#                     EllipseE(obj['shape_r'], obj['shape_e1'],
#                              obj['shape_e2']), Flux(0.6 * obj['flux']),
#                     EllipseE(obj['shape_r'], obj['shape_e1'],
#                              obj['shape_e2'])))
#         for obj in dev_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 DevGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     EllipseE(obj['shape_r'], obj['shape_e1'],
#                              -obj['shape_e2'])))
#         for obj in exp_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 ExpGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     EllipseE(obj['shape_r'], obj['shape_e1'],
#                              -obj['shape_e2'])))
#         for obj in rex_galaxy:
#             # if obj['point_source'] > 0.0:
#             #            sources.append(PointSource(PixPos(w.wcs_world2pix([[obj['ra'], obj['dec']]],1)[0]),
#             #                                               Flux(obj['flux'])))
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             src = ExpGalaxy(
#                 PixPos(pos_x, pos_y), Flux(obj['flux']),
#                 EllipseE(obj['shape_r'], 0, 0)
#             )
#             src.shape.freezeParam('e1')
#             src.shape.freezeParam('e2')
#             sources.append(src)

#         for obj in ser_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(
#                 SersicGalaxy(
#                     PixPos(pos_x, pos_y), Flux(obj['flux']),
#                     EllipseE(obj['shape_r'], obj['shape_e1'],
#                              -obj['shape_e2']),
#                     SersicIndex(1.0)
#                 )
#             )

#         for obj in psf_galaxy:
#             pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
#             sources.append(PointSource(
#                 PixPos(pos_x, pos_y), Flux(obj['flux'])))

#         print(" - Now you have %d sources" % len(sources))

#     elif shape_method is 'hsc':
#         from unagi import catalog
#         # Using HSC CModel catalog
#         if sources is None:
#             sources = []

#         for obj in ser_galaxy:
#             pos_x, pos_y = obj['x'], obj['y']
#             flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
#             r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
#                 obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
#                 to_pixel=False, update=False)  # arcsec, degree
#             sources.append(
#                 SersicGalaxy(
#                     PixPos(pos_x, pos_y), Flux(flux),
#                     GalaxyShape(r_gal, ba_gal, pa_gal + 90),
#                     SersicIndex(1.0)
#                 )
#             )

#         for obj in dev_galaxy:
#             pos_x, pos_y = obj['x'], obj['y']
#             flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
#             r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
#                 obj, shape_type='cmodel_dev_ellipse', axis_ratio=True,
#                 to_pixel=False, update=False)  # arcsec, degree
#             sources.append(
#                 DevGalaxy(
#                     PixPos(pos_x, pos_y), Flux(flux),
#                     GalaxyShape(r_gal, ba_gal, pa_gal + 90),
#                 )
#             )

#         for obj in exp_galaxy:
#             pos_x, pos_y = obj['x'], obj['y']
#             flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
#             r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
#                 obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
#                 to_pixel=False, update=False)  # arcsec, degree
#             sources.append(
#                 ExpGalaxy(
#                     PixPos(pos_x, pos_y), Flux(flux),
#                     GalaxyShape(r_gal, ba_gal, pa_gal + 90)
#                 )
#             )

#         for obj in rex_galaxy:
#             pos_x, pos_y = obj['x'], obj['y']
#             flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
#             r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
#                 obj, shape_type='cmodel_ellipse', axis_ratio=True,
#                 to_pixel=False, update=False)  # arcsec, degree
#             src = ExpGalaxy(
#                 PixPos(pos_x, pos_y), Flux(flux),
#                 GalaxyShape(r_gal, 1, 0)
#             )
#             src.shape.freezeParam('ab')
#             src.shape.freezeParam('phi')
#             sources.append(src)

#         for obj in psf_galaxy:
#             pos_x, pos_y = obj['x'], obj['y']
#             flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)

#             sources.append(PointSource(PixPos(pos_x, pos_y), Flux(flux)))

#         print(" - Now you have %d sources" % len(sources))
#     else:
#         raise ValueError('Cannot use this shape method')
#     return sources


# def tractor_hsc_cmodel(obj_name, coord, s_ang, filt, channels, data, hsc_dr,
#                        use_cmodel_filt=None, freeze_dict=None, ref_source=None, verbose=False):
#     '''
#     Run `the tractor` on HSC images, for Merian survey.

#     Parameters:
#         obj_name (str): name of the object.
#         coord (astropy.coordinate.SkyCoord): Coordinate of the object.
#         s_ang (astropy.units.arcsec): searching (angular) radius.
#         filt (str): filter name, such as 'r'.
#         channels (str): all filters, such as 'grizy'.
#         data (kuaizi.detection.Data): an data structure which contains images, weights, wcs, PSFs, etc.
#         hsc_dr: archive of HSC data, such as using `pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')`.
#         use_cmodel_filt (str): if not None (such as `use_cmodel_filt='i'`),
#             models in all bands will be initialized using the CModel catalog in this band.
#         freeze_pos (bool): whether freezing the positions of objects during fitting.
#         verbose (bool): whether being verbose.

#     Return:
#         trac_obj: tractor objects. Use `trac_obj.catalog.subs` to get models of sources,
#             use `trac_obj.getModelImage(0, minsb=0., srcs=sources[:])` to render the model image.
#     '''
#     from tractor.psf import PixelizedPSF
#     import scarlet
#     from unagi import hsc, config
#     from unagi import plotting
#     from unagi import task, catalog

#     print("### `" + obj_name + f"` {filt}-band")
#     layer_ind = channels.index(filt)

#     if use_cmodel_filt is not None:
#         cmodel_filt = use_cmodel_filt
#         print(
#             f' - Using {cmodel_filt}-band CModel as initial guesses for all bands')
#     else:
#         cmodel_filt = filt

#     # Retrieve HSC catalog
#     cutout_objs = task.hsc_box_search(
#         coord, box_size=s_ang * 1.1, archive=hsc_dr,
#         verbose=True, psf=True, cmodel=True, aper=True, shape=True,
#         meas=cmodel_filt, flux=False, aper_type='3_20')

#     cutout_clean, clean_mask = catalog.select_clean_objects(
#         cutout_objs, return_catalog=True, verbose=False)  # Select "clean" images

#     # Convert `RA, Dec` to `x, y`
#     x, y = data.wcs.wcs_world2pix(cutout_clean['ra'], cutout_clean['dec'], 0)
#     cutout_clean['x'] = x
#     cutout_clean['y'] = y

#     # sort by magnitude
#     cutout_clean.sort(f'cmodel_mag')
#     # Remove weird objects: abs(psf_mag - cmodel_mag) > 1
#     cutout_clean = cutout_clean[(
#         cutout_clean['psf_mag'] - cutout_clean['cmodel_mag']) < 2.5]
#     # Remove faint objects satisfying `i_cmodel_mag > 26` or `i_psf_mag > 26`
#     cutout_clean = cutout_clean[(cutout_clean['i_cmodel_mag'] <= 26) & (
#         cutout_clean['i_psf_mag'] <= 26)]

#     # Plot HSC CModel catalog on top of the rgb image
#     if filt == 'i':
#         stretch = 1
#         Q = 0.5
#         channel_map = scarlet.display.channels_to_rgb(len(channels))

#         img_rgb = scarlet.display.img_to_rgb(
#             data.images,
#             norm=scarlet.display.AsinhMapping(
#                 minimum=-0.2, stretch=stretch, Q=Q),
#             channel_map=channel_map)

#         _ = plotting.cutout_show_objects(
#             img_rgb, cutout_clean, cutout_wcs=data.wcs, xsize=8, show_weighted=True)  # Exp is brown. Dev is dashed-white.
#         plt.savefig(obj_name + '_cmodel_i.png', bbox_inches='tight')

#     # Find out the target galaxy in CModel catalog
#     catalog.moments_to_shape(
#         cutout_clean,
#         shape_type='cmodel_ellipse',
#         axis_ratio=True,
#         to_pixel=False,
#         update=True)  # arcsec, degree

#     catalog_c = SkyCoord(cutout_clean['ra'], cutout_clean['dec'], unit='deg')
#     dist = coord.separation(catalog_c)
#     cen_obj_ind = np.argsort(dist)[0]
#     cen_obj = cutout_clean[cen_obj_ind]

#     # Assign types to each object in CModel catalog
#     obj_type = np.empty_like(cutout_clean['object_id'], dtype='S4')

#     star_mask = cutout_clean['{}_extendedness'.format(filt)] < 0.5
#     # If extendedness is less than 0.5: assign 'PSF' type
#     obj_type[star_mask] = 'PSF'

#     fracdev = cutout_clean['cmodel_fracdev']
#     ba = cutout_clean['cmodel_ellipse_ba']
#     # If b/a > 0.8 (round shape): assign 'REX'
#     obj_type[(ba >= 0.8) & (~star_mask)] = 'REX'  # round_exp_galaxy
#     # If 0.6 < b/a < 0.8 (not very round) and `fracdev >= 0.6`: assign 'DEV'
#     obj_type[(ba < 0.8) & (fracdev >= 0.6) & (
#         ~star_mask)] = 'DEV'  # dev_galaxy
#     # If 0.6 < b/a < 0.8 (not very round) and `fracdev < 0.6`: assign 'REX' (although it might not be very round)
#     obj_type[(ba < 0.8) & (ba > 0.6) & (fracdev < 0.6) &
#              (~star_mask)] = 'REX'  # round_exp_galaxy
#     # If b/a < 0.6 (elongated) and `fracdev < 0.6`: assign 'EXP'
#     obj_type[(ba <= 0.6) & (fracdev < 0.6) & (
#         ~star_mask)] = 'EXP'  # exp_galaxy

#     # Target object is always Sersic
#     obj_type[cen_obj_ind] = 'SER'

#     cutout_clean['type'] = obj_type
#     ### Add a "target" column to indicate which object is the target galaxy ###
#     cutout_clean['target'] = np.zeros(len(cutout_clean), dtype=int)
#     cutout_clean['target'][cen_obj_ind] = 1

#     print(
#         f'# Type of the central object is {cutout_clean["type"][cen_obj_ind]}')
#     print(f'# Total number of objects: {len(cutout_clean)}')
#     print(f'# Central object index in {filt}-band: {cen_obj_ind}')

#     psf_obj = PixelizedPSF(data.psfs[layer_ind])  # Construct PSF

#     kfold = 3
#     while True:
#         try:
#             if kfold == 1:
#                 break
#             sources, trac_obj, fig = tractor_iteration(
#                 cutout_clean,
#                 data.wcs,
#                 data.images[layer_ind],
#                 data.weights[layer_ind],
#                 psf_obj,
#                 kuaizi.HSC_pixel_scale,
#                 shape_method='hsc',
#                 freeze_dict=freeze_dict,
#                 ref_source=ref_source,
#                 kfold=kfold,
#                 first_num=cen_obj_ind + 1,
#                 band_name=filt,
#                 fig_name=obj_name + '_cmodel_tractor_' + filt,
#                 verbose=verbose)
#             trac_obj.target_ind = cen_obj_ind  # record the index of target galaxy

#         except Exception as e:
#             print('   ' + str(e))
#             if kfold == 3:
#                 kfold += 1
#             else:
#                 kfold -= 2
#             pass
#         else:
#             break

#     return trac_obj


# def tractor_hsc_sep(obj, filt, channels, data, brick_file='../survey-bricks-dr9.fits.gz',
#                     freeze_dict=None, ref_source=None, verbose=False):
#     '''
#     Run `the tractor` on HSC images, for Merian survey. This uses DECaLS catalog as prior for fitting.

#     Parameters:
#         obj: a row in object catalog. Should contain 'ra', 'dec', 'name'.
#         filt (str): filter name, such as 'r'.
#         channels (str): all filters, such as 'grizy'.
#         data (kuaizi.detection.Data): an data structure which contains images, weights, wcs, PSFs, etc.
#         brick_file (str): the directory of DECaLS brick catalog.
#         freeze_dict (dict): for the target galaxy, whether freeze position, shape, flux, Sersic index, etc.
#             Example `freeze_dict` is like: `{'pos': True, 'shape': False, 'sersicindex': False}`.
#         ref_source (tractor source): Tractor source as a reference when doing forced photometry.
#             The target galaxy in other bands all takes parameters from this `ref_source`.
#         verbose (bool): whether being verbose.

#     Return:
#         trac_obj: tractor objects. Use `trac_obj.catalog.subs` to get models of sources,
#             use `trac_obj.getModelImage(0, minsb=0., srcs=sources[:])` to render the model image.
#     '''
#     from tractor.psf import PixelizedPSF
#     from astropy.coordinates import match_coordinates_sky

#     obj_name = obj['name'].rstrip('_y')
#     coord = SkyCoord(obj['ra'], obj['dec'], frame='icrs', unit='deg')

#     print("### `" + obj_name + f"` {filt}-band")
#     layer_ind = channels.index(filt)

#     obj_cat_sep, segmap_sep = makeCatalog(
#         [data],
#         layer_ind=layer_ind,
#         lvl=2.5,
#         mask=None,
#         method='vanilla',
#         convolve=False,
#         match_gaia=False,
#         show_fig=False,
#         visual_gaia=False,
#         b=32,
#         f=3,
#         pixel_scale=0.168,
#         minarea=5,
#         deblend_nthresh=48,
#         deblend_cont=0.005,
#         sky_subtract=True)
#     # Download DECaLS tractor catalogs and match SEP detection with the tractor catalog

#     bricks_cat = Table.read(brick_file, format='fits')  # DR8 brick catalog
#     bricks_corr = SkyCoord(
#         ra=np.array(bricks_cat['RA']) * u.degree,
#         dec=np.array(bricks_cat['DEC']) * u.degree)
#     detect_coor = SkyCoord(
#         ra=obj_cat_sep['ra'] * u.degree, dec=obj_cat_sep['dec'] * u.degree)
#     # Match our detection catalog to see which bricks it belongs to
#     to_download = bricks_cat[np.unique(
#         match_coordinates_sky(detect_coor, bricks_corr)[0])]
#     bricknames = to_download['BRICKNAME'].data.astype(
#         str)  # in case that there are more than one `tractor` file
#     # Download tractor catalog of the corresponding brick
#     tractor_cat = kuaizi.download.download_decals_tractor_catalog(
#         bricknames, layer='dr9', overwrite=False)

#     # Match these galaxies with DECaLS tractor file and get their type
#     decals_corr = SkyCoord(
#         ra=np.array(tractor_cat['ra']) * u.degree,
#         dec=np.array(tractor_cat['dec']) * u.degree)
#     detect_coor = SkyCoord(
#         ra=obj_cat_sep['ra'] * u.degree, dec=obj_cat_sep['dec'] * u.degree)

#     temp = tractor_cat[match_coordinates_sky(detect_coor, decals_corr)[0]]
#     for columns in temp.columns:
#         obj_cat_sep.add_column(temp[columns], rename_duplicate=True)
#     obj_cat_sep.sort('flux', reverse=True)

#     catalog_c = SkyCoord(obj_cat_sep['ra'], obj_cat_sep['dec'], unit='deg')
#     dist = coord.separation(catalog_c)
#     cen_obj_ind = np.argsort(dist)[0]
#     cen_obj = obj_cat_sep[cen_obj_ind]
#     ### Add a "target" column to indicate which object is the target galaxy ###
#     obj_cat_sep['target'] = np.zeros(len(obj_cat_sep), dtype=int)
#     obj_cat_sep['target'][cen_obj_ind] = 1
#     #obj_cat_sex['type'][obj_cat_sex['type'] == 'PSF'] = 'REX'
#     obj_cat_sep['type'][cen_obj_ind] = 'SER'

#     print(f'# Type of the central object is {cen_obj["type"]}')
#     print(f'# Total number of objects: {len(obj_cat_sep)}')
#     print(f'# Central object index in {filt}-band: {cen_obj_ind}')

#     psf_obj = PixelizedPSF(data.psfs[layer_ind])  # Construct PSF

#     kfold = 3
#     while True:
#         try:
#             if kfold == 1:
#                 break
#             sources, trac_obj, fig = tractor_iteration(
#                 obj_cat_sep,
#                 data.wcs,
#                 data.images[layer_ind],
#                 data.weights[layer_ind],
#                 psf_obj,
#                 kuaizi.HSC_pixel_scale,
#                 shape_method='decals',
#                 freeze_dict=freeze_dict,
#                 ref_source=ref_source,
#                 kfold=kfold,
#                 first_num=cen_obj_ind + 1,
#                 band_name=filt,
#                 fig_name=obj_name + '_sep_tractor_' + filt,
#                 verbose=verbose)
#             trac_obj.target_ind = cen_obj_ind  # record the index of target galaxy

#         except Exception as e:
#             print('   ' + str(e))
#             kfold -= 1
#             pass
#         else:
#             break

#     return trac_obj
