import numpy as np

from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

import kuaizi
import os
import sys

from .utils import _freeze_params, _set_bounds, HiddenPrints, getTargetProperty, makeCatalog


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

    obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))

    if shape_method is 'manual':
        # Using manually measured shapes using SEP
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
                src = DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi)))
                sources.append(src)

            elif obj_type[i] == 'EXP':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi)))
                sources.append(src)
            elif obj_type[i] == 'SER':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = SersicGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi)),
                    SersicIndex(2.0)
                )
                sources.append(src)
            elif obj_type[i] == 'REX':
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['a_arcsec'], 0, 0)
                )
                sources.append(src)
            else:
                pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
                src = PointSource(
                    PixPos(pos_x, pos_y), Flux(obj['flux']))
                sources.append(src)

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
        raise ValueError(
            'Cannot use this shape method. Only "manual", "decals", or "hsc" is supported now.')

    return sources


def tractor_blob_by_blob(obj_cat, w, img_data, invvar, psf_obj, pixel_scale,
                         shape_method='manual', freeze_dict=None, ref_source=None,
                         point_source=False,
                         band_name=None, show_fig=False, fig_name=None, verbose=False):
    '''
    Run tractor in a blob-by-blob way. 
    See "Goodness-of-Fits and Morphological type" in 
    https://www.legacysurvey.org/dr9/catalogs/#id9.

    Method
    ------
    1. We run SEP to detect sources on the provided image (typically i-band, 
        which is deepest). Then we sort the detection catalog by flux.
    2. We start doing tractor fitting from the brightest source. We iterate 
        over different types of source, calculate chi2 for each type, and choose
        the one with smallest chi2. We subtract the best model of the source
        from the images. Then we fit the second brightest source...
    3. We repeat the process until we have all the sources optimized. Now we have
        a catalog which contains best models for all objects. 

    Parameters
    ----------
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
    ref_source (tractor source): If this is provided, the central source will be fixed to have the same shape
        as the reference source, while other sources in the field are free to move.
    point_source (bool): If True, then the galaxy is fixed to be a point source (PSF).
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
    from tractor.constrained_optimizer import ConstrainedOptimizer
    import copy

    if len(obj_cat) < 1:
        raise ValueError(
            "The length of `obj_cat` is less than 1. Please check your catalog!")

    if freeze_dict is None:
        freeze_dict = {}

    blob_sources = []
    dchisq = []

    type_list = ['PSF', 'DEV', 'EXP', 'SER']

    image = copy.deepcopy(img_data)

    # iterate over all objects
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
            if obj['target'] == 1 and ref_source is not None:
                # don't iterate over types if we have a ref_source for the target
                src = [ref_source.copy()]
                src[0].brightness = Flux(obj['flux'])
                print(src[0])
                src = _freeze_params(
                    src, freeze_dict, cen_ind=0, fix_all=False)
                src = _set_bounds(src)
                trac_obj = Tractor([tim], src)
                trac_obj.freezeParam('images')
                trac_obj.optimize_loop()
                blob_sources.append(src[0])
            else:
                # enable to iterate over types
                for Type in type_list:  # iterate over morph types
                    if ~point_source & (obj['target'] == 1) & (Type == 'PSF'):
                        continue  # target object cannot be a PSF!!
                    obj_temp = copy.deepcopy(obj)
                    obj_temp['type'] = Type
                    src = add_tractor_sources(
                        Table(obj_temp), None, w, shape_method=shape_method)
                    src = _set_bounds(src)  # disable tractor to go crazy
                    # Freeze parameter for our target galaxy
                    if obj['target'] == 1:
                        src = _freeze_params(src, freeze_dict)
                    trac_obj = Tractor([tim], src)
                    trac_obj.freezeParam('images')
                    try:
                        # trac_obj.optimize()
                        trac_obj.optimize_loop()
                        # trac_obj.optimize_loop(dchisq=0.1, shared_params=False)
                        # _ = trac_obj.optimize(
                        #     variance=True, just_variance=True, shared_params=False)
                        chi_1 = (trac_obj.getChiImage()**2).mean()
                    except:
                        chi_1 = 999

                    chi_diff.append(chi_0 - chi_1)
                    blobs.append(src[0])
                dchisq.append(chi_diff)
                # use the type with smallest chi2
                if point_source and (obj['target'] == 1):
                    blob_sources.append(blobs[0])  # PSF
                else:
                    blob_sources.append(blobs[np.nanargmax(chi_diff)])

            # subtract optimized object model from image
            trac_mod_opt = trac_obj.getModelImage(0, minsb=0.)
            image -= trac_mod_opt

        if verbose:
            print(f'# Progress: {i+1} / {len(obj_cat)}')

    #############################
    # Final global optimization #
    #############################
    tim = Image(data=img_data,
                invvar=invvar,
                psf=psf_obj,
                wcs=NullWCS(pixscale=pixel_scale),
                sky=ConstantSky(0.0),
                photocal=NullPhotoCal()
                )
    trac_obj = Tractor([tim], _set_bounds(blob_sources))
    trac_obj.freezeParam('images')
    with HiddenPrints():
        chi2_0 = (trac_obj.getChiImage()**2).mean()
        try:
            # trac_obj.optimize_loop()
            _ = trac_obj.optimize(variance=False, just_variance=False, shared_params=False)
        except:
            pass
        chi2_1 = (trac_obj.getChiImage()**2).mean()
    if verbose:
        print('# Global optimization: Chi2 improvement = ', chi2_0 - chi2_1)

    #####################
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


def tractor_fix_all(catalog, img_data, invvar, psf_obj, pixel_scale,
                    freeze_dict=None, band_name=None, show_fig=False,
                    fig_name=None, verbose=False):
    '''
    After having a catalog of models for all sources from `tractor_blob_by_blob`, 
    we might want to use these models to fit in other bands. This is called 
    "forced photometry". For example, we fix the model shapes and only let the amplitude
    and position vary. The position is allowed to vary a little bit, by posing a `sigma = 1 pix` 
    Gaussian prior on `x` and `y` coordinate of centroid (see `kuaizi.tractor.utils._freeze_source`).

    his methodology is described well in Section 3.2 of https://arxiv.org/pdf/2110.13923.pdf.
    Therefore, this `tractor_fix_all` function takes a tractor catalog, fix all shape parameters,
    and then fit the amplitude and position of all sources. 

    Parameters
    ----------
    catalog : `tractor.engine.Catalog`, which is a "Tractor" catalog from `tractor_blob_by_blob`.
    img_data : `numpy.ndarray`, the image data in a certain band.
    invvar : `numpy.ndarray`, the inverse variance of the image data.
    psf_obj : `tractor.PSF`, the PSF of the image.
    pixel_scale : `float`, the pixel scale of the image.
    freeze_dict : `dict`, the dictionary of parameters to be frozen.
    band_name : `str`, the name of the band.
    show_fig : `bool`, whether to show the figure.
    fig_name : `str`, the name of the figure.
    verbose : `bool`, whether to print the progress.

    Returns
    -------
    blob_sources : `list`, the list of tractor sources.
    trac_obj : `tractor.engine.Catalog`, the tractor catalog.
    fig : `matplotlib.figure.Figure`, the figure.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.psf import Image, Tractor

    if len(catalog) < 1:
        raise ValueError(
            "The length of `obj_cat` is less than 1. Please check your catalog!")

    if freeze_dict is None:
        freeze_dict = {}

    catalog = _set_bounds(catalog)

    with HiddenPrints():
        tim = Image(data=img_data,
                    invvar=invvar,
                    psf=psf_obj,
                    wcs=NullWCS(pixscale=pixel_scale),
                    sky=ConstantSky(0.0),
                    photocal=NullPhotoCal()
                    )
        catalog = _set_bounds(_freeze_params(
            catalog, freeze_dict, fix_all=True))

    trac_obj = Tractor([tim], catalog)
    chi2_0 = (trac_obj.getChiImage()**2).mean()
    trac_obj.freezeParam('images')
    # trac_obj.optimize_loop(dchisq=0.001, shared_params=False)
    trac_obj.optimize_loop()
    # trac_obj.optimize()
    # _ = trac_obj.optimize(variance=False, just_variance=False, shared_params=False)
    chi2 = (trac_obj.getChiImage()**2).mean()
    # if chi2_0 - chi2 == 0:
        
    print('# Global optimization: Chi2 improvement = ', chi2_0 - chi2)

    blob_sources = trac_obj.catalog
    ########################
    # Plot the residual #
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
                                 obj_cat=None, fix_all=False, tractor_cat=None,
                                 freeze_dict=None, ref_source=None, point_source=False,
                                 show_fig=False, verbose=False):
    '''
    This function wraps both `tractor_blob_by_blob` and `tractor_fix_all` to fit
    sources across all bands. 

    Parameters
    ----------
    obj: a row in the object catalog. Should contain 'ra', 'dec', 'name'. 
    filt (str): filter name, such as 'r'.
    channels (str): all filters, such as 'grizy'.
    data (kuaizi.detection.Data): an data structure which contains 
        images, weights, wcs, PSFs, etc.
    freeze_dict (dict): for the target galaxy, whether freeze position, 
        shape, flux, Sersic index, etc. 
        Example `freeze_dict` is like: `{'pos': True, 'shape': False, 'sersicindex': False}`.
    ref_source (tractor source): If this is provided, the central source will be fixed 
        to have the same shape as the reference source, while other sources in 
        the field are free to move. Only works if `fix_all = False`.
    point_source (bool): If this is True, the central source will be a point source.
    show_fig (bool): whether to show the figure.
    verbose (bool): whether to print the progress.

    Return
    ------
    trac_obj: tractor objects. Use `trac_obj.catalog.subs` to get models of sources, 
        and use `trac_obj.getModelImage(0, minsb=0., srcs=sources[:])` to render the model image. 
    '''
    from kuaizi import HSC_pixel_scale
    from tractor.psf import PixelizedPSF

    obj_name = obj['name']  # .replace('_y', '')
    coord = SkyCoord(obj['ra'], obj['dec'], frame='icrs', unit='deg')

    if verbose:
        print(f"### `{obj_name}` {filt}-band")
    layer_ind = channels.index(filt)

    if fix_all and tractor_cat is None:
        raise ValueError(
            'Tractor catalog must be provided in `fix_all` mode. ')

    if obj_cat is None:
        obj_cat_sep, segmap_sep, _ = makeCatalog(
            [data],
            mask=None,
            lvl=2.0,
            method='vanilla',
            # method='wavelet',
            # low_freq_lvl=0, 
            # high_freq_lvl=2,
            layer_ind=layer_ind,
            convolve=False,
            match_gaia=False,
            show_fig=False,
            visual_gaia=False,
            tigress=True,
            # b=32,
            b=12,
            f=2,
            pixel_scale=HSC_pixel_scale,
            minarea=5,
            deblend_nthresh=72,
            deblend_cont=1e-3,#0.0005,
            sky_subtract=True,
            verbose=verbose)

        obj_cat_sep['a_arcsec'] = obj_cat_sep['a'] * HSC_pixel_scale
        obj_cat_sep['b_arcsec'] = obj_cat_sep['b'] * HSC_pixel_scale
        obj_cat_sep['type'] = np.repeat(b'NAN', len(obj_cat_sep))

        obj_cat_sep.sort('flux', reverse=True)
        # obj_cat_sep.sort('flux', reverse=False)

        catalog_c = SkyCoord(obj_cat_sep['ra'], obj_cat_sep['dec'], unit='deg')
        dist = coord.separation(catalog_c)
        cen_obj_ind = np.argsort(dist)[0]
        cen_obj = obj_cat_sep[cen_obj_ind]
        ### Add a "target" column to indicate which object is the target galaxy ###
        obj_cat_sep['target'] = np.zeros(len(obj_cat_sep), dtype=int)
        obj_cat_sep['target'][cen_obj_ind] = 1

        obj_cat = obj_cat_sep

    else:
        cen_obj_ind = np.where(obj_cat['target'] == 1)[0][0]

    if verbose:
        print(f'# Total number of objects: {len(obj_cat)}')
        print(f'# Central object index in {filt}-band: {cen_obj_ind}')

    psf_obj = PixelizedPSF(data.psfs[layer_ind])  # Construct PSF

    if not fix_all:
        result = tractor_blob_by_blob(
            obj_cat,
            data.wcs,
            data.images[layer_ind],
            data.weights[layer_ind],
            psf_obj,
            HSC_pixel_scale,
            shape_method='manual',
            freeze_dict=freeze_dict,
            ref_source=ref_source,
            point_source=point_source,
            show_fig=show_fig,
            fig_name=f'{obj_name}_sep_tractor_{filt}',
            verbose=verbose)
    else:
        result = tractor_fix_all(
            tractor_cat,
            data.images[layer_ind],
            data.weights[layer_ind],
            psf_obj,
            HSC_pixel_scale,
            freeze_dict=freeze_dict,
            show_fig=show_fig,
            fig_name=f'{obj_name}_sep_tractor_{filt}',
            verbose=verbose
        )

    if not show_fig:
        sources, trac_obj = result
    else:
        sources, trac_obj, fig = result
    # if not fix_all:
    trac_obj.target_ind = cen_obj_ind  # record the index of target galaxy
    trac_obj.wcs = data.wcs
    # if verbose:
    #     print(trac_obj.catalog[cen_obj_ind])
    return trac_obj, obj_cat