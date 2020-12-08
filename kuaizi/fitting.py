# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scarlet
import os
import sep
import time
import pickle 

import kuaizi as kz
from kuaizi.detection import Data
from kuaizi.display import display_single, SEG_CMAP
from kuaizi import HSC_pixel_scale, HSC_zeropoint

## Initialize `unagi`
from unagi import hsc, config
from unagi import plotting
from unagi.task import hsc_cutout, hsc_psf


import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
#from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache
from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel
from IPython.display import clear_output

plt.rcParams['font.size'] = 15
plt.rc('image', cmap='inferno', interpolation='none', origin='lower')


def _fitting_single_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.5, compact_away_factor=0.4):
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
    w = wcs.WCS(cutout[0][1].header) # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list) # Padding PSF cutouts from HSC
    psfs = scarlet.PSF(np.array(psf_pad))
    data = Data(images=images, weights=weights, wcs=w, psfs=psfs, channels=channels)


    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0), # averaged image
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
        #conv_radius=2,
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
    #print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(axis=0), cen_obj['x'], cen_obj['y'], 6)
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
        segmap[segmap == ind + 1] = 0   # we do not mask compact sources that are nearby to the center of target galaxy
    
    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask = (mask_conv >= gaussian_threshold)  # This `seg_mask` only masks compact sources

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
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask_large = (mask_conv >= gaussian_threshold)  # This `seg_mask_large` masks large bright sources

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0


    # Construct `scarlet` frames and observation
    from functools import partial
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))
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
        print(f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50: # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss: # better than the saved model
                        print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel}], fp)
                            fp.close()
                        break
                else:
                    print(f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(e_rel, len(blend.loss), -blend.loss[-1]))
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-fitting.png', bbox_inches='tight')

        return blend

    except Exception as e:
        print(e)
        return blend
  
def fitting_less_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.5, compact_away_factor=0.4):
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
    w = wcs.WCS(cutout[0][1].header) # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list) # Padding PSF cutouts from HSC
    psfs = scarlet.PSF(np.array(psf_pad))
    data = Data(images=images, weights=weights, wcs=w, psfs=psfs, channels=channels)


    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0), # averaged image
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
        deblend_cont=0.1,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    #print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(axis=0), cen_obj['x'], cen_obj['y'], 6)
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
                                                   high_freq_lvl=2,
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

    footprint = np.zeros_like(segmap, dtype=bool)  # the footprint of central object: an ellipse with 4 * a and 4 * b
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'], cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        segmap[segmap == ind + 1] = 0   # we do not mask compact sources that are nearby to the center of target galaxy
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask = (mask_conv >= gaussian_threshold)  # This `seg_mask` only masks compact sources

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
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask_large = (mask_conv >= gaussian_threshold)  # This `seg_mask_large` masks large bright sources


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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))
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
    coadd, bg_cutoff = build_initialization_coadd(observation, filtered_coadd=True) 

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

    for k, src in enumerate(obj_cat_cpct): # compact sources
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-img-less.png', bbox_inches='tight')
    
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-init-less.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50: # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss: # better than the saved model
                        print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')
        
        # with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
        #     blend = pickle.load(fp)[0]
        #     fp.close()

        # Find out what compose a galaxy
        if len(blend.sources) > 1:
            seds = np.array([np.copy(src.parameters[0]) for src in blend.sources])
            corr = np.corrcoef(seds)
            sed_ind = np.argsort(corr[0, :])[::-1] # np.where(corr[0, :] > 0.99)[0]#
            # dist = np.array([
            #     np.linalg.norm(src.center - blend.sources[0].center) * HSC_pixel_scale
            #     for src in np.array(blend.sources)[sed_ind]
            # ])
            # dist_flag = (dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
            point_flag = np.array([isinstance(src, scarlet.source.PointSource) for src in np.array(blend.sources)[sed_ind]])
            near_cen_flag = [(segmap_conv == cen_indx_conv + 1)[int(src.center[1]), int(src.center[0])] for src in np.array(blend.sources)[sed_ind]]
            sed_ind = sed_ind[(~point_flag) & near_cen_flag] # & dist_flag]
            if not 0 in sed_ind:
                sed_ind = np.array(list(set(sed_ind).union({0})))  # the central source must be included.
        else:
            sed_ind = np.array([0])
        print(f'Components {sed_ind} are considered as the target galaxy.')
        with open(f'./Models/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}, sed_ind], fp)
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-fitting-less.png', bbox_inches='tight')
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-zoomin-less.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend

def fitting_single_comp(lsbg, hsc_dr, cutout_halfsize=1.0, prefix='LSBG', large_away_factor=3.5, compact_away_factor=0.4):
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
    w = wcs.WCS(cutout[0][1].header) # note: all bands share the same WCS here
    filters = channels_list
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    psf_pad = padding_PSF(psf_list) # Padding PSF cutouts from HSC
    psfs = scarlet.PSF(np.array(psf_pad))
    data = Data(images=images, weights=weights, wcs=w, psfs=psfs, channels=channels)


    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0), # averaged image
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
        deblend_cont=0.1,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    #print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(axis=0), cen_obj['x'], cen_obj['y'], 6)
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
                                                   high_freq_lvl=2,
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

    footprint = np.zeros_like(segmap, dtype=bool)  # the footprint of central object: an ellipse with 4 * a and 4 * b
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'], cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        segmap[segmap == ind + 1] = 0   # we do not mask compact sources that are nearby to the center of target galaxy
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask = (mask_conv >= gaussian_threshold)  # This `seg_mask` only masks compact sources

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
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask_large = (mask_conv >= gaussian_threshold)  # This `seg_mask_large` masks large bright sources


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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))
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
    coadd, bg_cutoff = build_initialization_coadd(observation, filtered_coadd=True) 

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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-img-sing.png', bbox_inches='tight')
    
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-init-sing.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50: # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss: # better than the saved model
                        print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(e_rel, len(blend.loss), -blend.loss[-1]))
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-fitting-sing.png', bbox_inches='tight')
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-zoomin-sing.png', bbox_inches='tight')
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
    lsbg_coord = SkyCoord(ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    psfs = scarlet.PSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights, wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0), # averaged image
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
        deblend_cont=0.1,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    #print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(axis=0), cen_obj['x'], cen_obj['y'], 6)
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
                                                   high_freq_lvl=2,
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

    footprint = np.zeros_like(segmap, dtype=bool)  # the footprint of central object: an ellipse with 4 * a and 4 * b
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'], cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        segmap[segmap == ind + 1] = 0   # we do not mask compact sources that are nearby to the center of target galaxy
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask = (mask_conv >= gaussian_threshold)  # This `seg_mask` only masks compact sources

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
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask_large = (mask_conv >= gaussian_threshold)  # This `seg_mask_large` masks large bright sources


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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))
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
    coadd, bg_cutoff = build_initialization_coadd(observation, filtered_coadd=True) 

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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-img-sing.png', bbox_inches='tight')
    
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-init-sing.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50: # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss: # better than the saved model
                        print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-sing.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(e_rel, len(blend.loss), -blend.loss[-1]))
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-fitting-sing.png', bbox_inches='tight')
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-zoomin-sing.png', bbox_inches='tight')
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
    lsbg_coord = SkyCoord(ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    psfs = scarlet.PSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights, wcs=w, psfs=psfs, channels=channels)

    _, msk_star = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0), # averaged image
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
        deblend_cont=0.1,
        sky_subtract=True)
    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    #print(f'# Central object is #{cen_indx}.')
    # Better position for cen_obj
    x, y, _ = sep.winpos(data.images.mean(axis=0), cen_obj['x'], cen_obj['y'], 6)
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
                                                   high_freq_lvl=2,
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

    footprint = np.zeros_like(segmap, dtype=bool)  # the footprint of central object: an ellipse with 4 * a and 4 * b
    sep.mask_ellipse(footprint, cen_obj['x'], cen_obj['y'], cen_obj['a'], cen_obj['b'], cen_obj['theta'], r=4.0)
    inside_flag = [footprint[item] for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    for ind in np.where(inside_flag)[0]:
        segmap[segmap == ind + 1] = 0   # we do not mask compact sources that are nearby to the center of target galaxy
    obj_cat_cpct = obj_cat[inside_flag]   # catalog of compact sources

    smooth_radius = 2
    gaussian_threshold = 0.03
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask = (mask_conv >= gaussian_threshold)  # This `seg_mask` only masks compact sources

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
    mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
    seg_mask_large = (mask_conv >= gaussian_threshold)  # This `seg_mask_large` masks large bright sources


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
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 8, 8))
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
    coadd, bg_cutoff = build_initialization_coadd(observation, filtered_coadd=True) 

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
                                                        thresh=0.001,
                                                        shifting=False,
                                                        coadd=coadd, 
                                                        coadd_rms=bg_cutoff) 
    sources.append(new_source)

    for k, src in enumerate(obj_cat_cpct): # compact sources
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-img-less.png', bbox_inches='tight')
    
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
    plt.savefig(f'./Figures/{prefix}-{index:04d}-init-less.png', bbox_inches='tight')

    try:
        blend.fit(150, 1e-4)
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': 1e-4, 'loss': blend.loss[-1]}], fp)
            fp.close()
        last_loss = blend.loss[-1]
        print(f'Succeed for e_rel = 1e-4 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate([5e-4, 1e-5, 5e-5, 1e-6]):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50: # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                        pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss: # better than the saved model
                        print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
                            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}], fp)
                            fp.close()
                        break
                else:
                    print(f'Cannot achieve a global optimization with e_rel = {e_rel}.')

        print("Scarlet ran for {1} iterations to logL = {2}".format(e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsed time for fitting: {end - start} s')
        
        # with open(f"./Models/{prefix}-{index:04d}-trained-model.pkl", "rb") as fp:
        #     blend = pickle.load(fp)[0]
        #     fp.close()

        # Find out what compose a galaxy
        if len(blend.sources) > 1:
            mag_mat = np.array([-2.5 * np.log10(kz.measure.flux(src)) + 27 for src in sources])
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:] # g - r, g - i, g - z
            color_dist = np.linalg.norm(color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            sed_ind = np.where(color_dist < 0.2)[0] # np.argsort(color_dist)[:]  # 
            dist = np.array([
                np.linalg.norm(src.center - blend.sources[0].center) * HSC_pixel_scale
                for src in np.array(blend.sources)[sed_ind]
            ])
            dist_flag = (dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)
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
                sed_ind = np.array(list(set(sed_ind).union({0})))  # the central source must be included.
        else:
            sed_ind = np.array([0])
        print(f'Components {sed_ind} are considered as the target galaxy.')
        with open(f'./Models/MockGalScarlet/{prefix}-{index:04d}-trained-model-less.pkl', 'wb') as fp:
            pickle.dump([blend, {'e_rel': e_rel, 'loss': blend.loss[-1]}, sed_ind], fp)
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-fitting-less.png', bbox_inches='tight')
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
        plt.savefig(f'./Figures/{prefix}-{index:04d}-zoomin-less.png', bbox_inches='tight')
        return blend
    except Exception as e:
        print(e)
        return blend
