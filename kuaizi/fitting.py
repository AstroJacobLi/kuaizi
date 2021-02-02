# Import packages
import os
import pickle
import time
import copy

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scarlet
import sep
from astropy import wcs
from astropy.convolution import Box2DKernel, Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import Column, Table

from astropy.utils.data import clear_download_cache, download_file
from IPython.display import clear_output

# Initialize `unagi`
from unagi import config, hsc, plotting
from unagi.task import hsc_cutout, hsc_psf

# Import kuaizi
import kuaizi as kz
from kuaizi import HSC_pixel_scale, HSC_zeropoint
from kuaizi.detection import Data
from kuaizi.display import SEG_CMAP, display_single

plt.rcParams['font.size'] = 15
plt.rc('image', cmap='inferno', interpolation='none', origin='lower')


def _fitting_single_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.0, compact_away_factor=0.4):
    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)
    index = lsbg['Seq']
    lsbg_coord = SkyCoord(ra=lsbg['RAJ2000'], dec=lsbg['DEJ2000'], unit='deg')

    if not os.path.isdir('./Images'):
        os.mkdir('./Images')
    if not os.path.isdir('./PSFs'):
        os.mkdir('./PSFs')

    size_ang = cutout_halfsize * u.arcmin
    channels = 'griz'
    cutout = hsc_cutout(
        lsbg_coord,
        cutout_size=size_ang,
        filters=channels,
        mask=True,
        variance=True,
        archive=hsc_dr,
        use_saved=True,
        output_dir='./Images/',
        prefix=f'{prefix}_{index:04d}_img',
        save_output=True)
    psf_list = hsc_psf(
        lsbg_coord,
        centered=True,
        filters=channels,
        img_type='coadd',
        verbose=True,
        archive=hsc_dr,
        save_output=True,
        use_saved=True,
        prefix=f'{prefix}_{index:04d}_psf',
        output_dir='./PSFs/')

    channels_list = list(channels)

    # Reconstructure data
    images = np.array([hdu[1].data for hdu in cutout])
    w = wcs.WCS(cutout[0][1].header)  # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
    psfs = scarlet.ImagePSF(np.array(psf_pad))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        w,
        pixel_scale=HSC_pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=1.4)

    # This detection (after blurring the original images) finds out what is the central object and its (estimated) size
    obj_cat_ori, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=8,
        method='wavelet',
        convolve=False,
        # conv_radius=2,
        wavelet_lvl=5,
        low_freq_lvl=3,
        high_freq_lvl=0,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    # print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # This step masks out high freq sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                       mask=msk_star,
                                                       lvl=2.5,
                                                       method='wavelet',
                                                       high_freq_lvl=1,
                                                       wavelet_lvl=3,
                                                       match_gaia=False,
                                                       show_fig=True,
                                                       visual_gaia=False,
                                                       b=32,
                                                       f=3,
                                                       pixel_scale=0.168,
                                                       minarea=5,
                                                       deblend_nthresh=30,
                                                       deblend_cont=0.001,
                                                       sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    for ind in np.where(dist < (compact_away_factor * cen_obj['fwhm_custom'] * HSC_pixel_scale) * u.arcsec)[0]:
        # we do not mask compact sources that are nearby to the center of target galaxy
        segmap[segmap == ind + 1] = 0

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=10,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=40,
        f=3,
        pixel_scale=0.168,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.001,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    arr = np.zeros_like(segmap).astype('uint8')
    sep.mask_ellipse(
        arr,
        cen_obj['x'],
        cen_obj['y'],
        cen_obj['a'],
        cen_obj['b'],
        cen_obj['theta'],
        r=large_away_factor)  # don't mask the target galaxy too much

    for ind, obj in enumerate(obj_cat):
        if arr[int(obj['y']), int(obj['x'])] == 1:
            segmap[segmap == ind + 1] = 0

    smooth_radius = 4
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=w,
        psfs=model_psf,
        channels=filters)
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=filters)
    observation = observation.match(model_frame)

    # Add sources
    from scarlet.initialization import build_initialization_coadd
    coadd, bg_cutoff = build_initialization_coadd(observation)
    coadd[(seg_mask_large + seg_mask + msk_star.astype(bool))] = 0.0
    sources = []
    src = obj_cat_ori[cen_indx]
    if HSC_zeropoint - 2.5 * np.log10(src['flux']) > 26.5:
        # If too faint, single component
        new_source = scarlet.source.SingleExtendedSource(model_frame, (src['ra'], src['dec']),
                                                         observation,
                                                         thresh=0.0,
                                                         shifting=False,
                                                         coadd=coadd,
                                                         coadd_rms=bg_cutoff)
    else:
        new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                        observation,
                                                        K=2,  # Two components
                                                        thresh=0.01,
                                                        shifting=False)
    sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(f'./Figures/{prefix}-{index:04d}-img.png', bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')

        with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
            blend = pickle.load(fp)[0]
            fp.close()

        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-fitting.png', bbox_inches='tight')

        return blend

    except Exception as e:
        print(e)
        return blend


def fitting_less_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.0, compact_away_factor=0.4):
    clear_output()
    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)
    index = lsbg['Seq']
    lsbg_coord = SkyCoord(ra=lsbg['RAJ2000'], dec=lsbg['DEJ2000'], unit='deg')

    if not os.path.isdir('./Images'):
        os.mkdir('./Images')
    if not os.path.isdir('./PSFs'):
        os.mkdir('./PSFs')

    size_ang = cutout_halfsize * u.arcmin
    channels = 'griz'
    cutout = hsc_cutout(
        lsbg_coord,
        cutout_size=size_ang,
        filters=channels,
        mask=True,
        variance=True,
        archive=hsc_dr,
        use_saved=True,
        output_dir='./Images/',
        prefix=f'{prefix}_{index:04d}_img',
        save_output=True)
    psf_list = hsc_psf(
        lsbg_coord,
        centered=True,
        filters=channels,
        img_type='coadd',
        verbose=True,
        archive=hsc_dr,
        save_output=True,
        use_saved=True,
        prefix=f'{prefix}_{index:04d}_psf',
        output_dir='./PSFs/')

    channels_list = list(channels)

    # Reconstructure data
    images = np.array([hdu[1].data for hdu in cutout])
    w = wcs.WCS(cutout[0][1].header)  # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
    psfs = scarlet.ImagePSF(np.array(psf_pad))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        w,
        pixel_scale=HSC_pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=0.6)

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        mask=msk_star,
        lvl=1.2,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.08,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    # print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # This detection (after blurring the original images) finds out what is the central object and its (estimated) size
    obj_cat, segmap_conv, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=8,
        method='wavelet',
        convolve=False,
        wavelet_lvl=5,
        low_freq_lvl=3,
        high_freq_lvl=0,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_conv = obj_cat_ori[np.argsort(dist)[0]]['index']

    # This step masks out HIGH FREQUENCY sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                       mask=msk_star,
                                                       lvl=2.5,
                                                       method='wavelet',
                                                       high_freq_lvl=3,
                                                       wavelet_lvl=4,
                                                       match_gaia=False,
                                                       show_fig=True,
                                                       visual_gaia=False,
                                                       b=32,
                                                       f=3,
                                                       pixel_scale=HSC_pixel_scale,
                                                       minarea=5,
                                                       deblend_nthresh=30,
                                                       deblend_cont=0.05,
                                                       sky_subtract=True)

    # the footprint of central object: an ellipse with 4 * a and 4 * b
    footprint = np.zeros_like(segmap, dtype=bool)
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'],
                     cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(
        zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        # we do not mask compact sources that are nearby to the center of target galaxy
        segmap[segmap == ind + 1] = 0
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=10,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=40,
        f=3,
        pixel_scale=0.168,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.001,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    arr = np.zeros_like(segmap).astype('uint8')
    sep.mask_ellipse(
        arr,
        cen_obj['x'],
        cen_obj['y'],
        cen_obj['a'],
        cen_obj['b'],
        cen_obj['theta'],
        r=large_away_factor)  # don't mask the target galaxy too much

    for ind, obj in enumerate(obj_cat):
        if arr[int(obj['y']), int(obj['x'])] == 1:
            segmap[segmap == ind + 1] = 0

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Remove objects that are masked from the compact obj catalog
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 2 * u.arcsec)[0])
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=w,
        psfs=model_psf,
        channels=filters)
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=filters)
    observation = observation.match(model_frame)

    # Add sources
    from scarlet.initialization import build_initialization_coadd

    # Filtered coadd removes noise! Very useful for faint objects (but slow)
    coadd, bg_cutoff = build_initialization_coadd(
        observation, filtered_coadd=True)
    coadd[(seg_mask_large + seg_mask + msk_star.astype(bool))] = 0.0
    sources = []
    src = obj_cat_ori[cen_indx]
    if HSC_zeropoint - 2.5 * np.log10(src['flux']) > 26.:
        # If too faint, single component
        new_source = scarlet.source.SingleExtendedSource(model_frame, (src['ra'], src['dec']),
                                                         observation,
                                                         thresh=0.001,
                                                         shifting=False,
                                                         coadd=coadd,
                                                         coadd_rms=bg_cutoff)
    else:
        new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                        observation,
                                                        K=2,  # Two components
                                                        thresh=0.01,
                                                        shifting=False,
                                                        coadd=coadd,
                                                        coadd_rms=bg_cutoff)
    sources.append(new_source)

    for k, src in enumerate(obj_cat_cpct):  # compact sources
        if src['fwhm_custom'] < 5:  # src['b'] / src['a'] > 0.9 and
            new_source = scarlet.source.PointSource(
                model_frame, (src['ra'], src['dec']), observation)
        else:
            try:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation, coadd=coadd, coadd_rms=bg_cutoff)
            except:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation)
        sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-img-less.png', bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        zoomin_size=50,
        minimum=-0.3,
        stretch=1,
        channels='griz',
        show_loss=True,
        show_mask=True,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-init-less.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                        pickle.dump(
                            [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                            pickle.dump(
                                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')

        # with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
        #     blend = pickle.load(fp)[0]
        #     fp.close()

        # Find out what compose a galaxy
        if len(blend.sources) > 1:
            mag_mat = np.array(
                [-2.5 * np.log10(kz.measure.flux(src)) + 27 for src in sources])
            # g - r, g - i, g - z
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:]
            color_dist = np.linalg.norm(
                color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            # np.argsort(color_dist)[:]  #
            sed_ind = np.where(color_dist < 0.2)[0]
            dist = np.array([
                np.linalg.norm(
                    src.center - blend.sources[0].center) * HSC_pixel_scale
                for src in np.array(blend.sources)[sed_ind]
            ])
            dist_flag = (
                dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
            point_flag = np.array([
                isinstance(src, scarlet.source.PointSource)
                for src in np.array(blend.sources)[sed_ind]
            ])
            near_cen_flag = [
                (segmap_conv == cen_indx_conv +
                 1)[int(src.center[1]), int(src.center[0])]
                for src in np.array(blend.sources)[sed_ind]
            ]
            sed_ind = sed_ind[(~point_flag) & near_cen_flag]

            if not 0 in sed_ind:
                # the central source must be included.
                sed_ind = np.array(list(set(sed_ind).union({0})))
        else:
            sed_ind = np.array([0])
        print(f'Components {sed_ind} are considered as the target galaxy.')
        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump(
                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}, sed_ind], fp)
            fp.close()

        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-fitting-less.png', bbox_inches='tight')
        fig = kz.display.display_scarlet_model(
            blend,
            show_ind=sed_ind,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-zoomin-less.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend


def fitting_single_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.0, compact_away_factor=0.4):
    clear_output()
    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)
    index = lsbg['Seq']
    lsbg_coord = SkyCoord(ra=lsbg['RAJ2000'], dec=lsbg['DEJ2000'], unit='deg')

    if not os.path.isdir('./Images'):
        os.mkdir('./Images')
    if not os.path.isdir('./PSFs'):
        os.mkdir('./PSFs')

    size_ang = cutout_halfsize * u.arcmin
    channels = 'griz'
    cutout = hsc_cutout(
        lsbg_coord,
        cutout_size=size_ang,
        filters=channels,
        mask=True,
        variance=True,
        archive=hsc_dr,
        use_saved=True,
        output_dir='./Images/',
        prefix=f'{prefix}_{index:04d}_img',
        save_output=True)
    psf_list = hsc_psf(
        lsbg_coord,
        centered=True,
        filters=channels,
        img_type='coadd',
        verbose=True,
        archive=hsc_dr,
        save_output=True,
        use_saved=True,
        prefix=f'{prefix}_{index:04d}_psf',
        output_dir='./PSFs/')

    channels_list = list(channels)

    # Reconstructure data
    images = np.array([hdu[1].data for hdu in cutout])
    w = wcs.WCS(cutout[0][1].header)  # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
    psfs = scarlet.ImagePSF(np.array(psf_pad))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        w,
        pixel_scale=HSC_pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=0.6)

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=1.2,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.08,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    # print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # This detection (after blurring the original images) finds out what is the central object and its (estimated) size
    obj_cat, segmap_conv, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=8,
        method='wavelet',
        convolve=False,
        wavelet_lvl=5,
        low_freq_lvl=3,
        high_freq_lvl=0,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_conv = obj_cat_ori[np.argsort(dist)[0]]['index']

    # This step masks out HIGH FREQUENCY sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                       mask=msk_star,
                                                       lvl=2.5,
                                                       method='wavelet',
                                                       high_freq_lvl=3,
                                                       wavelet_lvl=4,
                                                       match_gaia=False,
                                                       show_fig=True,
                                                       visual_gaia=False,
                                                       b=32,
                                                       f=3,
                                                       pixel_scale=HSC_pixel_scale,
                                                       minarea=5,
                                                       deblend_nthresh=30,
                                                       deblend_cont=0.05,
                                                       sky_subtract=True)

    # the footprint of central object: an ellipse with 4 * a and 4 * b
    footprint = np.zeros_like(segmap, dtype=bool)
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'],
                     cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(
        zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        # we do not mask compact sources that are nearby to the center of target galaxy
        segmap[segmap == ind + 1] = 0
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=10,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=40,
        f=3,
        pixel_scale=0.168,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.001,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    arr = np.zeros_like(segmap).astype('uint8')
    sep.mask_ellipse(
        arr,
        cen_obj['x'],
        cen_obj['y'],
        cen_obj['a'],
        cen_obj['b'],
        cen_obj['theta'],
        r=large_away_factor)  # don't mask the target galaxy too much

    for ind, obj in enumerate(obj_cat):
        if arr[int(obj['y']), int(obj['x'])] == 1:
            segmap[segmap == ind + 1] = 0

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Remove objects that are masked from the compact obj catalog
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 2 * u.arcsec)[0])
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=w,
        psfs=model_psf,
        channels=filters)
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=filters)
    observation = observation.match(model_frame)

    # Add sources
    from scarlet.initialization import build_initialization_coadd

    # Filtered coadd removes noise! Very useful for faint objects (but slow)
    coadd, bg_cutoff = build_initialization_coadd(
        observation, filtered_coadd=True)
    coadd[(seg_mask_large + seg_mask + msk_star.astype(bool))] = 0.0
    sources = []
    src = obj_cat_ori[cen_indx]
    if HSC_zeropoint - 2.5 * np.log10(src['flux']) > 26.5:
        # If too faint, single component
        new_source = scarlet.source.SingleExtendedSource(model_frame, (src['ra'], src['dec']),
                                                         observation,
                                                         thresh=0.001,
                                                         shifting=False,
                                                         coadd=coadd,
                                                         coadd_rms=bg_cutoff)
    else:
        new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                        observation,
                                                        K=2,  # Two components
                                                        thresh=0.01,
                                                        shifting=False,
                                                        coadd=coadd,
                                                        coadd_rms=bg_cutoff)
    sources.append(new_source)

    # for k, src in enumerate(obj_cat_cpct): # compact sources
    #     if src['fwhm_custom'] < 5:  # src['b'] / src['a'] > 0.9 and
    #         new_source = scarlet.source.PointSource(
    #             model_frame, (src['ra'], src['dec']), observation)
    #     else:
    #         try:
    #             new_source = scarlet.source.SingleExtendedSource(
    #                 model_frame, (src['ra'], src['dec']), observation, coadd=coadd, coadd_rms=bg_cutoff)
    #         except:
    #             new_source = scarlet.source.SingleExtendedSource(
    #                 model_frame, (src['ra'], src['dec']), observation)
    #     sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-img-sing.png', bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        zoomin_size=50,
        minimum=-0.3,
        stretch=1,
        channels='griz',
        show_loss=True,
        show_mask=True,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-init-sing.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                        pickle.dump(
                            [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                            pickle.dump(
                                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')

        # with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
        #     blend = pickle.load(fp)[0]
        #     fp.close()

        # Find out what compose a galaxy
        # seds = np.array([np.copy(src.parameters[0]) for src in blend.sources])
        # corr = np.corrcoef(seds)
        # sed_ind = np.argsort(corr[0, :])[::-1] # np.where(corr[0, :] > 0.99)[0]#
        # # dist = np.array([
        # #     np.linalg.norm(src.center - blend.sources[0].center) * HSC_pixel_scale
        # #     for src in np.array(blend.sources)[sed_ind]
        # # ])
        # # dist_flag = (dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
        # point_flag = np.array([isinstance(src, scarlet.source.PointSource) for src in np.array(blend.sources)[sed_ind]])
        # near_cen_flag = [(segmap_conv == cen_indx_conv + 1)[int(src.center[1]), int(src.center[0])] for src in np.array(blend.sources)[sed_ind]]
        # sed_ind = sed_ind[(~point_flag) & near_cen_flag] # & dist_flag]
        # if not 0 in sed_ind:
        #     sed_ind.append(0)  # the central source must be included.
        # print(f'Components {sed_ind} are considered as the target galaxy.')
        # with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
        #     pickle.dump([blend, {'e_rel': e_rel}, sed_ind], fp)
        #     fp.close()
        with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
            fp.close()

        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-fitting-sing.png', bbox_inches='tight')
        fig = kz.display.display_scarlet_model(
            blend,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-zoomin-sing.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend


def fitting_single_comp_mockgal(index=0, prefix='MockLSBG', large_away_factor=3.0, compact_away_factor=0.4, zp=HSC_zeropoint):
    clear_output()
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    index = index

    from kuaizi.mock import MockGal
    mgal = MockGal.read(f'./Models/MockGalModel/{prefix}-{index:04d}.pkl')
    print(f'Opening ./Models/MockGalModel/{prefix}-{index:04d}.pkl')
    channels = mgal.channels
    channels_list = list(channels)
    filters = channels_list
    lsbg_coord = SkyCoord(
        ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    psfs = scarlet.ImagePSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        w,
        pixel_scale=HSC_pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=0.6)

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        mask=msk_star,
        lvl=1.2,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.08,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    # print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # This detection (after blurring the original images) finds out what is the central object and its (estimated) size
    obj_cat, segmap_conv, bg_rms = kz.detection.makeCatalog(
        [data],
        mask=msk_star,
        lvl=8,
        method='wavelet',
        convolve=False,
        wavelet_lvl=5,
        low_freq_lvl=3,
        high_freq_lvl=0,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_conv = obj_cat_ori[np.argsort(dist)[0]]['index']

    # This step masks out HIGH FREQUENCY sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                       mask=msk_star,
                                                       lvl=2.5,
                                                       method='wavelet',
                                                       high_freq_lvl=3,
                                                       wavelet_lvl=4,
                                                       match_gaia=False,
                                                       show_fig=True,
                                                       visual_gaia=False,
                                                       b=32,
                                                       f=3,
                                                       pixel_scale=HSC_pixel_scale,
                                                       minarea=5,
                                                       deblend_nthresh=30,
                                                       deblend_cont=0.05,
                                                       sky_subtract=True)

    # the footprint of central object: an ellipse with 4 * a and 4 * b
    footprint = np.zeros_like(segmap, dtype=bool)
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'],
                     cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(
        zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        # we do not mask compact sources that are nearby to the center of target galaxy
        segmap[segmap == ind + 1] = 0
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=10,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=40,
        f=3,
        pixel_scale=0.168,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.001,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    arr = np.zeros_like(segmap).astype('uint8')
    sep.mask_ellipse(
        arr,
        cen_obj['x'],
        cen_obj['y'],
        cen_obj['a'],
        cen_obj['b'],
        cen_obj['theta'],
        r=large_away_factor)  # don't mask the target galaxy too much

    for ind, obj in enumerate(obj_cat):
        if arr[int(obj['y']), int(obj['x'])] == 1:
            segmap[segmap == ind + 1] = 0

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Remove objects that are masked from the compact obj catalog
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 2 * u.arcsec)[0])
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=w,
        psfs=model_psf,
        channels=filters)
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=filters)
    observation = observation.match(model_frame)

    # Add sources
    from scarlet.initialization import build_initialization_coadd

    # Filtered coadd removes noise! Very useful for faint objects (but slow)
    coadd, bg_cutoff = build_initialization_coadd(
        observation, filtered_coadd=True)
    coadd[(seg_mask_large + seg_mask + msk_star.astype(bool))] = 0.0
    sources = []
    src = obj_cat_ori[cen_indx]
    if zp - 2.5 * np.log10(src['flux']) > 26.:
        # If too faint, single component
        new_source = scarlet.source.SingleExtendedSource(model_frame, (src['ra'], src['dec']),
                                                         observation,
                                                         thresh=0.001,
                                                         shifting=False,
                                                         coadd=coadd,
                                                         coadd_rms=bg_cutoff)
    else:
        new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                        observation,
                                                        K=2,  # Two components
                                                        thresh=0.001,
                                                        shifting=False,
                                                        coadd=coadd,
                                                        coadd_rms=bg_cutoff)
    sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-img-sing.png', bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        zoomin_size=50,
        minimum=-0.3,
        stretch=1,
        channels='griz',
        show_loss=True,
        show_mask=True,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-init-sing.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                        pickle.dump(
                            [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                            pickle.dump(
                                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')

        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
            fp.close()

        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-fitting-sing.png', bbox_inches='tight')
        fig = kz.display.display_scarlet_model(
            blend,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-zoomin-sing.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend


def fitting_less_comp_mockgal(index=0, prefix='MockLSBG', large_away_factor=3.0, compact_away_factor=0.4, zp=HSC_zeropoint):
    clear_output()
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    index = index

    from kuaizi.mock import MockGal
    mgal = MockGal.read(f'./Models/MockGalModel/{prefix}-{index:04d}.pkl')
    print(f'Opening ./Models/MockGalModel/{prefix}-{index:04d}.pkl')
    channels = mgal.channels
    channels_list = list(channels)
    filters = channels_list
    lsbg_coord = SkyCoord(
        ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    psfs = scarlet.ImagePSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        w,
        pixel_scale=HSC_pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=0.6)

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        mask=msk_star,
        lvl=1.2,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.08,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    # print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    # x, y = cen_obj['x'], cen_obj['y']
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # This detection (after blurring the original images) finds out what is the central object and its (estimated) size
    obj_cat, segmap_conv, bg_rms = kz.detection.makeCatalog(
        [data],
        mask=msk_star,
        lvl=8,
        method='wavelet',
        convolve=False,
        wavelet_lvl=5,
        low_freq_lvl=3,
        high_freq_lvl=0,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=0.168,
        minarea=20,
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_conv = obj_cat_ori[np.argsort(dist)[0]]['index']

    # This step masks out HIGH FREQUENCY sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                       mask=msk_star,
                                                       lvl=2.5,
                                                       method='wavelet',
                                                       high_freq_lvl=3,
                                                       wavelet_lvl=4,
                                                       match_gaia=False,
                                                       show_fig=True,
                                                       visual_gaia=False,
                                                       b=32,
                                                       f=3,
                                                       pixel_scale=HSC_pixel_scale,
                                                       minarea=5,
                                                       deblend_nthresh=30,
                                                       deblend_cont=0.05,
                                                       sky_subtract=True)

    # the footprint of central object: an ellipse with 4 * a and 4 * b
    footprint = np.zeros_like(segmap, dtype=bool)
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'],
                     cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(
        zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        # we do not mask compact sources that are nearby to the center of target galaxy
        segmap[segmap == ind + 1] = 0
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=10,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=40,
        f=3,
        pixel_scale=0.168,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.001,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)

    arr = np.zeros_like(segmap).astype('uint8')
    sep.mask_ellipse(
        arr,
        cen_obj['x'],
        cen_obj['y'],
        cen_obj['a'],
        cen_obj['b'],
        cen_obj['theta'],
        r=large_away_factor)  # don't mask the target galaxy too much

    for ind, obj in enumerate(obj_cat):
        if arr[int(obj['y']), int(obj['x'])] == 1:
            segmap[segmap == ind + 1] = 0

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Remove objects that are masked from the compact obj catalog
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 2 * u.arcsec)[0])
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(filters))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=w,
        psfs=model_psf,
        channels=filters)
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=filters)
    observation = observation.match(model_frame)

    # Add sources
    from scarlet.initialization import build_initialization_coadd

    # Filtered coadd removes noise! Very useful for faint objects (but slow)
    coadd, bg_cutoff = build_initialization_coadd(
        observation, filtered_coadd=True)
    coadd[(seg_mask_large + seg_mask + msk_star.astype(bool))] = 0.0
    sources = []
    src = obj_cat_ori[cen_indx]
    if HSC_zeropoint - 2.5 * np.log10(src['flux']) > 26.:
        # If too faint, single component
        new_source = scarlet.source.SingleExtendedSource(model_frame, (src['ra'], src['dec']),
                                                         observation,
                                                         thresh=0.005,
                                                         shifting=False,
                                                         coadd=coadd,
                                                         coadd_rms=bg_cutoff)
    else:
        new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                        observation,
                                                        K=2,  # Two components
                                                        thresh=0.01,
                                                        shifting=False,
                                                        coadd=coadd,
                                                        coadd_rms=bg_cutoff)
    sources.append(new_source)

    for k, src in enumerate(obj_cat_cpct):  # compact sources
        if src['fwhm_custom'] < 5:  # src['b'] / src['a'] > 0.9 and
            new_source = scarlet.source.PointSource(
                model_frame, (src['ra'], src['dec']), observation)
        else:
            try:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation, coadd=coadd, coadd_rms=bg_cutoff)
            except:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation)
        sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-img-less.png', bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        zoomin_size=50,
        minimum=-0.3,
        stretch=1,
        channels='griz',
        show_loss=True,
        show_mask=True,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        f'./Figures/{prefix}-{index:04d}-init-less.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                        pickle.dump(
                            [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                            pickle.dump(
                                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')

        # with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
        #     blend = pickle.load(fp)[0]
        #     fp.close()

        # Find out what compose a galaxy
        if len(blend.sources) > 1:
            mag_mat = np.array(
                [-2.5 * np.log10(kz.measure.flux(src)) + 27 for src in sources])
            # g - r, g - i, g - z
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:]
            color_dist = np.linalg.norm(
                color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            # np.argsort(color_dist)[:]  #
            sed_ind = np.where(color_dist < 0.2)[0]
            dist = np.array([
                np.linalg.norm(
                    src.center - blend.sources[0].center) * HSC_pixel_scale
                for src in np.array(blend.sources)[sed_ind]
            ])
            dist_flag = (
                dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
            point_flag = np.array([
                isinstance(src, scarlet.source.PointSource)
                for src in np.array(blend.sources)[sed_ind]
            ])
            near_cen_flag = [
                (segmap_conv == cen_indx_conv +
                 1)[int(src.center[1]), int(src.center[0])]
                for src in np.array(blend.sources)[sed_ind]
            ]
            sed_ind = sed_ind[(~point_flag) & near_cen_flag]

            if not 0 in sed_ind:
                # the central source must be included.
                sed_ind = np.array(list(set(sed_ind).union({0})))
        else:
            sed_ind = np.array([0])
        print(f'Components {sed_ind} are considered as the target galaxy.')
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump(
                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}, sed_ind], fp)
            fp.close()

        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-fitting-less.png', bbox_inches='tight')
        fig = kz.display.display_scarlet_model(
            blend,
            show_ind=sed_ind,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            f'./Figures/{prefix}-{index:04d}-zoomin-less.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend


def _fitting_wavelet(data, coord, pixel_scale=HSC_pixel_scale, zp=HSC_zeropoint, prefix='mockgal', index=0, model_dir='./Model', figure_dir='./Figure'):
    '''
    This is a fitting function for internal use. It fits the galaxy using Starlet model, and apply a mask after fitting.

    '''
    from scarlet import Starlet

    lsbg_coord = coord
    print('# Query GAIA stars...')
    gaia_cat, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        data.wcs,
        pixel_scale=pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,
        factor_f=0.6)

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=1.2,
        mask=msk_star,
        method='vanilla',
        convolve=False,
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=128,
        f=3,
        pixel_scale=pixel_scale,
        minarea=20,
        deblend_nthresh=48,
        deblend_cont=0.07, # 0.07, I changed it to 0.1
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_ori = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx_ori]

    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec

    # We roughly guess the box size of the Starlet model
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(data.channels))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=data.wcs,
        psfs=model_psf,
        channels=list(data.channels))
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=list(data.channels))
    observation = observation.match(model_frame)

    cen_obj = obj_cat_ori[cen_indx_ori]
    starlet_source = scarlet.StarletSource(model_frame,
                                           (cen_obj['ra'], cen_obj['dec']),
                                           observation,
                                           min_grad=-0.4,  # the initial guess of box size is as large as possible
                                           starlet_thresh=1)
    starlet_extent = kz.display.get_extent(starlet_source.bbox)

    # Show the Starlet initial box
    fig = display_single(data.images.mean(axis=0))
    from matplotlib.patches import Rectangle
    box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}
    rect = Rectangle(
        (starlet_extent[0], starlet_extent[2]),
        starlet_extent[1] - starlet_extent[0],
        starlet_extent[3] - starlet_extent[2],
        **box_kwargs
    )
    ax = plt.gca()
    ax.add_patch(rect)
    plt.close()

    if gaia_cat is not None:
        star_flag = [(item[0] > starlet_extent[0]) & (item[0] < starlet_extent[1]) &
                     (item[1] > starlet_extent[2]) & (
                         item[1] < starlet_extent[3])
                     for item in np.asarray(
            data.wcs.wcs_world2pix(gaia_cat['ra'], gaia_cat['dec'], 0), dtype=int).T]
        # "star_cat" is a catalog for GAIA stars which fall in the Starlet box
        star_cat = gaia_cat[star_flag]

        _, msk_star = kz.utils.gaia_star_mask(  # Generate GAIA mask only for stars outside of the Starlet box
            data.images.mean(axis=0),
            data.wcs,
            gaia_stars=gaia_cat[~np.array(star_flag)],
            pixel_scale=pixel_scale,
            gaia_bright=19.5,
            mask_a=694.7,
            mask_b=3.8,
            factor_b=1.0,
            factor_f=0.6)
    else:
        star_cat = []

    # This step masks out high frequency sources by doing wavelet transformation
    obj_cat, segmap_highfreq, bg_rms = kz.detection.makeCatalog([data],
                                                                mask=msk_star,
                                                                lvl=2.5,
                                                                method='wavelet',
                                                                high_freq_lvl=2,  # 3
                                                                wavelet_lvl=4,
                                                                match_gaia=False,
                                                                show_fig=True,
                                                                visual_gaia=False,
                                                                b=24,
                                                                f=3,
                                                                pixel_scale=pixel_scale,
                                                                minarea=3,
                                                                deblend_nthresh=30,
                                                                deblend_cont=0.03,
                                                                sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_highfreq = obj_cat[np.argsort(dist)[0]]['index']

    # Don't mask out objects that fall in the segmap of the central object and the Starlet box
    segmap = segmap_highfreq.copy()
    # overlap_flag is for objects which fall in the footprint of central galaxy in the fist SEP detection
    overlap_flag = [(segmap_ori == (cen_indx_ori + 1))[item]
                    for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    # box_flat is for objects which fall in the initial Starlet box
    box_flag = np.unique(
        segmap[starlet_extent[0]:starlet_extent[1], starlet_extent[2]:starlet_extent[3]]) - 1
    box_flag = np.delete(np.sort(box_flag), 0)
    overlap_flag = np.array(overlap_flag)
    overlap_flag[box_flag] = True
    obj_cat_cpct = obj_cat[overlap_flag]

    # Remove the source if it is the central galaxy
    if dist[cen_indx_highfreq] < 1 * u.arcsec:
        obj_cat_cpct.remove_rows(
            np.where(obj_cat_cpct['index'] == cen_indx_highfreq)[0])

    for ind in np.where(overlap_flag)[0]:
        segmap[segmap == ind + 1] = 0

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap_big, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=4,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=True,
        visual_gaia=False,
        b=45,
        f=3,
        pixel_scale=pixel_scale,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx_big = obj_cat_ori[np.argsort(dist)[0]]['index']

    # mask out big objects that are not identified in the high_freq step
    segmap = segmap_big.copy()
    box_flag = np.unique(
        segmap[starlet_extent[0]:starlet_extent[1], starlet_extent[2]:starlet_extent[3]]) - 1
    box_flag = np.delete(np.sort(box_flag), 0)

    for ind in box_flag:
        segmap[segmap == ind + 1] = 0

    box_flag = np.delete(box_flag, np.where(box_flag == cen_indx_big)[
                         0])  # dont include the central galaxy
    obj_cat_big = obj_cat[box_flag]

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

        # Remove compact objects that are too close to the central
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 2 * u.arcsec)[0])
    # Remove objects that are already masked!
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])
    # Remove objects that are already masked!
    inside_flag = [
        (layer == 0)[item] for item in list(
            zip(obj_cat_big['y'].astype(int), obj_cat_big['x'].astype(int)))
    ]
    obj_cat_big.remove_rows(np.where(inside_flag)[0])

    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(data.channels))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=data.wcs,
        psfs=model_psf,
        channels=list(data.channels))
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psfs=data.psfs,
        weights=data.weights,
        channels=list(data.channels))
    observation = observation.match(model_frame)

    sources = []

    # Add central Starlet source
    src = obj_cat_ori[cen_indx_ori]
    # Find a better box, not too large, not too small
    for min_grad in np.arange(-0.3, 0.4, 0.05):
        starlet_source = scarlet.StarletSource(
            model_frame,
            (src['ra'], src['dec']),
            observation,
            min_grad=min_grad,
            starlet_thresh=1)
        starlet_extent = kz.display.get_extent(starlet_source.bbox)
        segbox = segmap_ori[starlet_extent[2]:starlet_extent[3],
                            starlet_extent[0]:starlet_extent[1]]
        contam_ratio = 1 - \
            np.sum((segbox == 0) | (segbox == cen_indx_ori + 1)) / \
            np.sum(np.ones_like(segbox))
        if contam_ratio < 0.08:
            break
    print(f'# min_grad = {min_grad:.2f}, contam_ratio = {contam_ratio:.2f}')
    starlet_source.center = (
        np.array(starlet_source.bbox.shape) // 2 + starlet_source.bbox.origin)[1:]
    sources.append(starlet_source)

    # Only model "real compact" sources
    if len(obj_cat_big) > 0:
        cpct = SkyCoord(
            ra=np.array(obj_cat_cpct['ra']) * u.degree,
            dec=np.array(obj_cat_cpct['dec']) * u.degree)
        big = SkyCoord(ra=obj_cat_big['ra'] * u.degree,
                       dec=obj_cat_big['dec'] * u.degree)
        tempid, sep2d, _ = match_coordinates_sky(big, cpct)
        cpct = obj_cat_cpct[np.setdiff1d(
            np.arange(len(obj_cat_cpct)), tempid[np.where(sep2d < 2 * u.arcsec)])]
    else:
        cpct = obj_cat_cpct

    for k, src in enumerate(cpct):
        if src['fwhm_custom'] < 5:
            new_source = scarlet.source.SingleExtendedSource(
                model_frame, (src['ra'], src['dec']), observation, compact=True)
        else:
            new_source = scarlet.source.SingleExtendedSource(
                model_frame, (src['ra'], src['dec']), observation, thresh=2)
        sources.append(new_source)

    # IF GAIA stars are within the box: exclude it from the big_cat
    if len(obj_cat_big) > 0:
        if len(star_cat) > 0:
            star = SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg')
            tempid, sep2d, _ = match_coordinates_sky(star, big)
            big_cat = obj_cat_big[np.setdiff1d(
                np.arange(len(obj_cat_big)), tempid[np.where(sep2d < 2 * u.arcsec)])]
        else:
            big_cat = obj_cat_big

        # [np.where(sep2d > 2 * u.arcsec)[0]]
        for k, src in enumerate(big_cat):
            if src['fwhm_custom'] > 12:
                new_source = scarlet.source.ExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation, K=2)
            else:
                # try:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation, thresh=2)
        #         except:
        #             new_source = scarlet.source.SingleExtendedSource(
        #                 model_frame, (src['ra'], src['dec']), observation, coadd=coadd, coadd_rms=bg_cutoff)
            sources.append(new_source)

    if len(star_cat) > 0:
        for k, src in enumerate(star_cat):
            new_source = scarlet.source.ExtendedSource(
                model_frame, (src['ra'], src['dec']), observation, K=2)
            sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=1,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(
        os.path.join(figure_dir, f'{prefix}-{index:04d}-src-wavelet.png'), bbox_inches='tight')

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        minimum=-0.3,
        stretch=1,
        channels='griz',
        show_loss=False,
        show_mask=False,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        os.path.join(figure_dir, f'{prefix}-{index:04d}-init-wavelet.png'), bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(os.path.join(model_dir, f'{prefix}-{index:04d}-trained-model-wavelet.pkl'), 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(
            f'  - Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(
                        f'  - Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(os.path.join(model_dir, f'{prefix}-{index:04d}-trained-model-wavelet.pkl'), 'wb') as fp:
                        pickle.dump(
                            [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        print(
                            f'  - I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(os.path.join(model_dir, f'{prefix}-{index:04d}-trained-model-wavelet.pkl'), 'wb') as fp:
                            pickle.dump(
                                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(
                        f'  - Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'    - Elapsed time for fitting: {end - start} s')

        # In principle, Now we don't need to find which components compose a galaxy. The central Starlet is enough!
        if len(blend.sources) > 1:
            mag_mat = np.array(
                [-2.5 * np.log10(kz.measure.flux(src, observation)) + 27 for src in sources])
            # g - r, g - i, g - z
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:]
            color_dist = np.linalg.norm(
                color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            # np.argsort(color_dist)[:]  #
            sed_ind = np.where(color_dist < 0.2)[0]
            dist = np.array([
                np.linalg.norm(
                    src.center - blend.sources[0].center) * HSC_pixel_scale
                for src in np.array(blend.sources)[sed_ind]
            ])
            dist_flag = (
                dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
            point_flag = np.array([
                isinstance(src, scarlet.source.PointSource)
                for src in np.array(blend.sources)[sed_ind]
            ])
            near_cen_flag = [
                (segmap_ori == cen_indx_ori +
                 1)[int(src.center[1]), int(src.center[0])]
                for src in np.array(blend.sources)[sed_ind]
            ]
            sed_ind = sed_ind[(~point_flag) & near_cen_flag]

            if not 0 in sed_ind:
                # the central source must be included.
                sed_ind = np.array(list(set(sed_ind).union({0})))
        else:
            sed_ind = np.array([0])
        print(f'Components {sed_ind} are considered as the target galaxy.')

        # Generate a VERY AGGRESSIVE mask, named "footprint"
        footprint = np.zeros_like(segmap_highfreq, dtype=bool)
        for ind in cpct['index']:  # mask ExtendedSources which are modeled
            footprint[segmap_highfreq == ind + 1] = 1

        footprint[segmap_highfreq == cen_indx_highfreq + 1] = 0
        sed_ind_pix = np.array([item.center for item in np.array(
            sources)[sed_ind]])  # the y and x of sed_ind objects
        # if any objects in `sed_ind` is in `segmap_highfreq`
        sed_corr_indx = segmap_highfreq[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
        for ind in sed_corr_indx:
            footprint[segmap_highfreq == ind] = 0

        smooth_radius = 1.5
        gaussian_threshold = 0.03
        mask_conv = np.copy(footprint)
        mask_conv[mask_conv > 0] = 1
        mask_conv = convolve(mask_conv.astype(
            float), Gaussian2DKernel(smooth_radius))
        footprint = (mask_conv >= gaussian_threshold)

        if len(obj_cat_big) > 0:
            footprint2 = np.zeros_like(segmap_big, dtype=bool)
            for ind in big_cat['index']:  # mask ExtendedSources which are modeled
                footprint2[segmap_big == ind + 1] = 1
                # if any objects in `sed_ind` is in `segmap_big`
            sed_corr_indx = segmap_big[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
            for ind in sed_corr_indx:
                footprint2[segmap_big == ind] = 0
            footprint2[segmap_big == cen_indx_big + 1] = 0

            smooth_radius = 4
            gaussian_threshold = 0.02
            mask_conv = np.copy(footprint2)
            mask_conv[mask_conv > 0] = 1
            mask_conv = convolve(mask_conv.astype(
                float), Gaussian2DKernel(smooth_radius))
            footprint2 = (mask_conv >= gaussian_threshold)

            footprint = footprint | footprint2  # This is the mask for everything except target galaxy

        with open(os.path.join(model_dir, f'{prefix}-{index:04d}-trained-model-wavelet.pkl'), 'wb') as fp:
            pickle.dump(
                [blend, {'e_rel': e_rel, 'loss': blend.loss[-1], 'sed_ind': sed_ind}, footprint], fp)
            fp.close()

        # Save fitting figure
        fig = kz.display.display_scarlet_model(
            blend,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=False,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            os.path.join(figure_dir, f'{prefix}-{index:04d}-fitting-wavelet.png'), bbox_inches='tight')

        # Save zoomin figure (non-agressively-masked, target galaxy only)
        fig = kz.display.display_scarlet_model(
            blend,
            show_ind=sed_ind,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=True,
            show_mask=False,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            os.path.join(figure_dir, f'{prefix}-{index:04d}-zoomin-wavelet.png'), bbox_inches='tight')
        
        # Save zoomin figure (agressively-masked, target galaxy only)
        new_weights = data.weights.copy()
        for layer in new_weights:
            layer[footprint.astype(bool)] = 0
        observation2 = scarlet.Observation(
            data.images,
            wcs=data.wcs,
            psfs=data.psfs,
            weights=new_weights,
            channels=list(data.channels))
        observation2 = observation2.match(model_frame)
        blend2 = scarlet.Blend(sources, observation2)

        fig = kz.display.display_scarlet_model(
            blend2,
            show_ind=sed_ind,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=False,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            os.path.join(figure_dir, f'{prefix}-{index:04d}-zoomin-mask-wavelet.png'), bbox_inches='tight')

        # Save high-freq-removed figure
        ## remove high-frequency features from the Starlet objects
        for src in np.array(blend2.sources)[sed_ind]:
            if isinstance(src, scarlet.StarletSource):
                # Cutout a patch of original image
                y_cen, x_cen = np.array(src.bbox.shape)[1:] // 2 + np.array(src.bbox.origin)[1:]
                size = np.array(src.bbox.shape)[1:] // 2
                img_ = observation.images[:, y_cen - size[0]:y_cen + size[0] + 1, x_cen - size[1]:x_cen + size[1] + 1]

                morph = src.children[1]
                stlt = Starlet(morph.get_model(), direct=True)
                c = stlt.coefficients
                c[:, :2, :, :] = 0 # Remove high-frequency features
                new_morph = copy.deepcopy(morph)
                new_src = copy.deepcopy(src)
                new_morph.__init__(morph.frame, img_, coeffs=c, bbox=morph.bbox)
                src.children[1] = new_morph
        #blend2 = scarlet.Blend(sources, observation2)
        fig = kz.display.display_scarlet_model(
            blend2,
            show_ind=sed_ind,
            zoomin_size=50,
            minimum=-0.3,
            stretch=1,
            channels='griz',
            show_loss=False,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(
            os.path.join(figure_dir, f'{prefix}-{index:04d}-zoomin-blur-wavelet.png'), bbox_inches='tight')

        return blend
    except Exception as e:
        print(e)
        return blend


def fitting_wavelet_observation(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', pixel_scale=HSC_pixel_scale,
                                zp=HSC_zeropoint, model_dir='./Models', figure_dir='./Figure'):
    clear_output()
    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)

    index = lsbg['Seq']
    lsbg_coord = SkyCoord(ra=lsbg['RAJ2000'], dec=lsbg['DEJ2000'], unit='deg')

    img_dir = './Images/'
    psf_dir = './PSFs/'

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(psf_dir):
        os.mkdir(psf_dir)

    size_ang = cutout_halfsize * u.arcmin
    channels = 'griz'
    cutout = hsc_cutout(
        lsbg_coord,
        cutout_size=size_ang,
        filters=channels,
        mask=True,
        variance=True,
        archive=hsc_dr,
        use_saved=True,
        save_output=True,
        output_dir=img_dir,
        prefix=f'LSBG_{index:04d}_img') # {prefix}

    psf_list = hsc_psf(
        lsbg_coord,
        centered=True,
        filters=channels,
        img_type='coadd',
        verbose=True,
        archive=hsc_dr,
        use_saved=True,
        save_output=True,
        output_dir=psf_dir,
        prefix=f'LSBG_{index:04d}_psf')

    channels_list = list(channels)

    # Reconstructure data
    images = np.array([hdu[1].data for hdu in cutout])
    w = wcs.WCS(cutout[0][1].header)  # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
    psfs = scarlet.ImagePSF(np.array(psf_pad))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)
    
    blend = _fitting_wavelet(
        data, lsbg_coord, prefix=prefix, index=index, pixel_scale=pixel_scale,
        zp=zp, model_dir=model_dir, figure_dir=figure_dir)
    return blend


def fitting_wavelet_mockgal(index=0, prefix='MockLSBG', pixel_scale=HSC_pixel_scale,
                            zp=HSC_zeropoint, model_dir='./Models', figure_dir='./Figure'):
    clear_output()
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    index = index

    from kuaizi.mock import MockGal
    mgal = MockGal.read(os.path.join(
        model_dir, f'MockGalModel/{prefix}-{index:04d}.pkl'))
    print('Loading', os.path.join(
        model_dir, f'MockGalModel/{prefix}-{index:04d}.pkl'))
    channels = mgal.channels
    channels_list = list(channels)
    filters = channels_list
    lsbg_coord = SkyCoord(
        ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    psfs = scarlet.ImagePSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    blend = _fitting_wavelet(
        data, lsbg_coord, prefix=prefix, index=index, pixel_scale=pixel_scale,
        zp=zp, model_dir=model_dir, figure_dir=figure_dir)
    return blend
