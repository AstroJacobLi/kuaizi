# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scarlet
import os
import sep
import time

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

plt.rcParams['font.size'] = 15
plt.rc('image', cmap='inferno', interpolation='none', origin='lower')


def fitting_single_comp(lsbg, hsc_dr, cutout_halfsize=0.6, prefix='LSBG', index=0, large_away_factor=3.5, compact_away_factor=0.4):

    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)
    lsbg_coord = SkyCoord(ra=lsbg['ra'], dec=lsbg['dec'], unit='deg')
    size_ang = cutout_halfsize * u.arcmin
    channels = 'griz'

    if not os.path.isdir('./Images'):
        os.mkdir('./Images')
    if not os.path.isdir('./PSFs'):
        os.mkdir('./PSFs')
    
    cutout = hsc_cutout(
        lsbg_coord,
        cutout_size=size_ang,
        filters=channels,
        mask=True,
        variance=True,
        archive=hsc_dr,
        use_saved=False,
        output_dir='./Images/',
        save_output=True)
    psf_list = hsc_psf(
        lsbg_coord,
        centered=True,
        filters=channels,
        img_type='coadd',
        prefix=None,
        verbose=True,
        archive=hsc_dr,
        save_output=True,
        use_saved=False,
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
    obj_cat_ori, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                        lvl=10,
                                                        method='vanilla',
                                                        convolve=True,
                                                        conv_radius=5,
                                                        match_gaia=False,
                                                        show_fig=True,
                                                        visual_gaia=False,
                                                        b=128,  # subtract a smooth background
                                                        f=3, 
                                                        pixel_scale=0.168,
                                                        minarea=5,
                                                        deblend_nthresh=30,
                                                        deblend_cont=0.001,
                                                        sky_subtract=True)

    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    cen_indx = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx]
    #print(f'# Central object is #{cen_indx}.')

    # This step masks out high freq sources after wavelet transformation
    obj_cat, segmap, bg_rms = kz.detection.makeCatalog([data],
                                                    mask=msk_star,
                                                    lvl=4,
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
        if arr[int(obj['x']), int(obj['y'])] == 1:
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
    new_source = scarlet.source.MultiExtendedSource(model_frame, (src['ra'], src['dec']),
                                                    observation,
                                                    K=2,   # two components
                                                    thresh=0.01,
                                                    shifting=False)   # I don't use the manual `coadd` and `bg_cutoff` here.
    sources.append(new_source)

    # Visualize our data and mask and source
    if not os.path.isdir('./Figures'):
        os.mkdir('./Figures/')
    fig = kz.display.display_scarlet_sources(
        data,
        sources,
        show_ind=None,
        stretch=2,
        Q=1,
        minimum=-0.3,
        show_mark=True,
        scale_bar_length=10,
        add_text=f'{prefix}-{index}')
    plt.savefig(f'./Figures/{prefix}-{index}.png', bbox_inches='tight')


    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)

    try:
        for e_rel in [1e-4, 1e-5, 1e-6]:
            blend.fit(150, e_rel)
            if len(blend.loss) > 20:
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
            else:
                recent_loss = np.mean(blend.loss[-5:])
                min_loss = np.min(blend.loss[:-5])
            if recent_loss < min_loss:
                print(f'Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
            elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                print(f'I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                break
            else:
                continue
        print("Scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
        end = time.time()
        print(f'Elapsing time: {end - start} s')

        fig = kz.display.display_scarlet_model(
            blend,
            observation,
            minimum=-0.3,
            stretch=2,
            channels=channels,
            show_loss=True,
            show_mask=True,
            show_mark=False,
            scale_bar=False)
        plt.savefig(f'./Figures/{prefix}-{index}-fit.png', bbox_inches='tight')

        return blend, observation
    except:
        return blend, observation
    