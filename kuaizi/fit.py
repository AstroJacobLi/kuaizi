"""
This file contains classes and functions for Scarlet fitting. I'm trying to make it more mordular.
Based on `fitting.py`.
"""
# Import packages
import os
import gc
import sys
import pickle
import dill
import time
import copy
import traceback

import sep
import numpy as np
import scarlet
import unagi  # for HSC saturation mask
from scarlet.source import StarletSource
from scarlet.operator import prox_monotonic_mask


from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.table import Column, Table
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord, match_coordinates_sky

import matplotlib.pyplot as plt

import kuaizi as kz
from kuaizi import HSC_pixel_scale, HSC_zeropoint
from kuaizi.detection import Data
from kuaizi.display import display_single, display_rgb, SEG_CMAP

sys.setrecursionlimit(10000)
plt.rcParams['font.size'] = 15
plt.rc('image', cmap='inferno', interpolation='none', origin='lower')


def _optimization(blend, bright=False, logger=None, bkg=False):
    logger.info('  - Optimizing scarlet model...')
    print('  - Optimizing scarlet model...')
    if bright:
        # , 1e-4  # otherwise it will take forever....
        e_rel_list = [1e-3, 1e-4, ]  #
        n_iter = [200, 200, ] if bkg else [100, 100, ]
    else:
        e_rel_list = [1e-4, 5e-4, 2e-4]  # , 2e-4 # , 5e-5, 1e-5
        n_iter = [150, 100, 100] if bkg else [
            100, 100, 50]  # , 50 [200, 200, 200]

    blend.fit(1, 1e-3)  # First iteration, just to get the inital loss

    best_model = blend
    best_logL = -blend.loss[-1]
    best_erel = 1e-3
    best_epoch = 20

    for i, e_rel in enumerate(e_rel_list):
        for k in range(n_iter[i] // 5):
            blend.fit(5, e_rel)
            if (-blend.loss[-1] > best_logL) and (len(blend.loss) > best_epoch):
                best_model = copy.deepcopy(blend)
                best_logL = -blend.loss[-1]
                best_erel = e_rel
                best_epoch = len(blend.loss)

        logger.info(
            f'    Optimizaiton: Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
        print(
            f'    Optimizaiton: Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')

        if len(blend.loss) <= 50:  # must have more than 50 iterations
            continue

    if len(blend.loss) < 50:
        logger.warning(
            '  ! Might be poor fitting! Iterations less than 50.')
        print('  ! Might be poor fitting! Iterations less than 50.')
    logger.info(
        "  - After {0} iterations, logL = {1:.2f}".format(best_epoch, best_logL))
    print(
        "  - After {0} iterations, logL = {1:.2f}".format(best_epoch, best_logL))

    return [best_model, best_logL, best_erel, best_epoch, blend]


class ScarletFittingError(Exception):
    """Exception raised for errors when fitting a galaxy using scarlet.

    Attributes:
        prefix, index, coord, channels
        message -- explanation of the error
    """

    def __init__(self, prefix, index, starlet_thresh, monotonic, channels, error):
        self.prefix = prefix
        self.index = index
        self.starlet_thresh = starlet_thresh
        self.monotonic = monotonic
        self.channels = channels
        self.error = error
        super().__init__(self.error)

    def __str__(self):
        return f'{self.prefix}-{self.index} in `{self.channels}` bands with `starlet_thresh = {self.starlet_thresh}, monotonic = {self.monotonic}` -> {self.error}'


class ScarletFitter(object):
    def __init__(self, method='vanilla', tigress=True, bright=False, min_grad=-0.03, bkg=True, **kwargs) -> None:
        """Initialize a ScarletFitter object.

        Args:
            method (str, optional): method used in scarlet fitting, either 'vanilla' or 'wavelet'.
                Defaults to 'vanilla'.
            tigress (bool, optional): whether to use Tigress data. Defaults to True.
            bright (bool, optional): whether treat this object as bright object. Defaults to False.
            bkg (bool, optional): whether to add a constant sky component. 
            **kwargs: keyword arguments for ScarletFitter, including
                ['log_dir', 'figure_dir', 'model_dir', 'prefix', 'index',
                 'pixel_scale', 'zeropoint', show_figure', starlet_thresh', 'monotonic', 'variance].
        """
        if method not in ['vanilla', 'wavelet', 'spergel']:
            raise ValueError(
                f'Method {method} is not supported! Only "vaniila" and "wavelet" are supported.')

        self.method = method
        self.tigress = tigress
        self.bright = bright
        self.min_grad = min_grad
        self.bkg = bkg

        self.log_dir = kwargs.get('log_dir', './log')
        self.figure_dir = kwargs.get('figure_dir', './Figure')
        self.model_dir = kwargs.get('model_dir', './Model')

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.figure_dir):
            os.makedirs(self.figure_dir)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self.prefix = kwargs.get('prefix', 'test')
        self.index = kwargs.get('index', 0)
        self.pixel_scale = kwargs.get('pixel_scale', HSC_pixel_scale)
        self.zeropoint = kwargs.get('zeropoint', HSC_zeropoint)
        self.show_figure = kwargs.get('show_figure', False)
        self.starlet_thresh = kwargs.get('starlet_thresh', 0.5)
        self.monotonic = kwargs.get('monotonic', True)
        self.variance = kwargs.get('variance', 0.03**2)
        self.scales = kwargs.get('scales', [0, 1, 2, 3, 4, 5, 6])

        self._set_logger()

    def load_data(self, data, coord):
        self.logger.info('    Load data')
        self.data = data
        self.coord = coord

    def _set_logger(self):
        from .utils import set_logger
        logger = set_logger(f'fitting_{self.method}', os.path.join(
            self.log_dir, f'{self.prefix}-{self.index}.log'))
        self.logger = logger

    def _first_gaia_search(self, scale_factor=1.0):
        print('    Query GAIA stars...')
        self.logger.info('    Query GAIA stars...')
        # Generate a mask for GAIA bright stars
        self.gaia_cat, self.msk_star_ori = kz.utils.gaia_star_mask(
            self.data.images.mean(axis=0),  # averaged image
            self.data.wcs,
            pixel_scale=self.pixel_scale,
            gaia_bright=19.5,
            mask_a=694.7 * scale_factor,
            mask_b=3.8,
            factor_b=1.0,  # 0.7,
            factor_f=1.4,  # 1.0,
            tigress=self.tigress,
            logger=self.logger)
        if self.gaia_cat is None:
            self.n_stars = 0
        else:
            self.n_stars = len(self.gaia_cat)

    def _first_detection(self, first_dblend_cont, conv_radius=2, b=80, f=3):
        obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
            [self.data],
            lvl=4,  # a.k.a., "sigma"
            mask=self.msk_star_ori,
            method='vanilla',
            convolve=True,
            conv_radius=conv_radius,
            match_gaia=False,
            show_fig=self.show_figure,
            visual_gaia=False,
            b=b,
            f=f,
            pixel_scale=self.pixel_scale,
            minarea=20,
            deblend_nthresh=48,
            deblend_cont=first_dblend_cont,  # 0.01, 0.05, 0.07, I changed it to 0.1
            sky_subtract=True,
            logger=self.logger)

        catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
        dist = self.coord.separation(catalog_c)
        # ori = original, i.e., first SEP run
        cen_indx_ori = obj_cat_ori[np.argsort(dist)[0]]['index']
        cen_obj = obj_cat_ori[cen_indx_ori]

        # Better position for cen_obj, THIS IS PROBLEMATIC!!!
        # x, y, _ = sep.winpos(self.data.images.mean(
        #     axis=0), cen_obj['x'], cen_obj['y'], 6)
        x, y = cen_obj['x'], cen_obj['y']
        ra, dec = self.data.wcs.wcs_pix2world(x, y, 0)
        cen_obj = dict(cen_obj)
        cen_obj['x'] = x
        cen_obj['y'] = y
        cen_obj['ra'] = ra
        cen_obj['dec'] = dec
        cen_obj['idx'] = cen_indx_ori
        cen_obj['coord'] = SkyCoord(cen_obj['ra'], cen_obj['dec'], unit='deg')
        # cen_obj_coord = SkyCoord(cen_obj['ra'], cen_obj['dec'], unit='deg')

        self.cen_obj = cen_obj
        self.obj_cat_ori = obj_cat_ori
        self.segmap_ori = segmap_ori

    def _estimate_box(self, cen_obj):
        # We roughly guess the box size of the Starlet model
        model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(self.data.channels))
        model_frame = scarlet.Frame(
            self.data.images.shape,
            wcs=self.data.wcs,
            psf=model_psf,
            channels=list(self.data.channels))
        observation = scarlet.Observation(
            self.data.images,
            wcs=self.data.wcs,
            psf=self.data.psfs,
            weights=self.data.weights,
            channels=list(self.data.channels))
        observation = observation.match(model_frame)

        if self.method == 'vanilla':
            min_grad = -0.01
        elif self.method == 'wavelet':
            min_grad = -0.1
        else:
            min_grad = -0.1

        starlet_source = StarletSource(model_frame,
                                       (cen_obj['ra'], cen_obj['dec']),
                                       observation,
                                       thresh=0.01,
                                       min_grad=min_grad,  # the initial guess of box size is as large as possible
                                       starlet_thresh=5e-3)
        # If the initial guess of the box is way too large (but not bright galaxy), set min_grad = 0.1.
        # The box is way too large
        if starlet_source.bbox.shape[1] > 0.9 * self.data.images[0].shape[0] and (self.bright):
            # The box is way too large
            min_grad = 0.03
            smaller_box = True
        elif starlet_source.bbox.shape[1] > 0.9 * self.data.images[0].shape[0] and (~self.bright):
            # not bright but large box: something must be wrong! min_grad should be larger
            min_grad = 0.05
            smaller_box = True
        elif starlet_source.bbox.shape[1] > 0.6 * self.data.images[0].shape[0] and (self.bright):
            # If box is large and gal is bright
            min_grad = 0.02
            smaller_box = True
        elif starlet_source.bbox.shape[1] > 0.6 * self.data.images[0].shape[0] and (~self.bright):
            # If box is large and gal is not bright
            min_grad = 0.02
            smaller_box = True
        else:
            smaller_box = False

        if smaller_box:
            starlet_source = scarlet.StarletSource(model_frame,
                                                   (cen_obj['ra'],
                                                    cen_obj['dec']),
                                                   observation,
                                                   thresh=0.01,
                                                   min_grad=min_grad,  # the initial guess of box size is as large as possible
                                                   starlet_thresh=5e-3)

        starlet_extent = kz.display.get_extent(
            starlet_source.bbox)  # [x1, x2, y1, y2]

        # extra padding, to enlarge the box
        starlet_extent[0] -= 10
        starlet_extent[2] -= 10
        starlet_extent[1] += 10
        starlet_extent[3] += 10

        self.starlet_extent = starlet_extent
        self.smaller_box = smaller_box

        if self.show_figure:
            # Show the Starlet initial box
            fig, ax = plt.subplots(figsize=(6, 6))
            ax = display_single(self.data.images.mean(axis=0), ax=ax)
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

    def _mask_stars_outside_box(self, scale_factor=1.0):
        if self.n_stars > 0:
            # Find stars within the wavelet box, and mask them.
            star_flag = [(item[0] > self.starlet_extent[0]) & (item[0] < self.starlet_extent[1]) &
                         (item[1] > self.starlet_extent[2]) & (
                             item[1] < self.starlet_extent[3])
                         for item in np.asarray(
                self.data.wcs.wcs_world2pix(self.gaia_cat['ra'], self.gaia_cat['dec'], 0), dtype=int).T]
            # "star_cat" is a catalog for GAIA stars which fall in the Starlet box
            self.star_cat = self.gaia_cat[star_flag]

            # Generate GAIA mask only for stars outside of the Starlet box
            _, self.msk_star = kz.utils.gaia_star_mask(
                self.data.images.mean(axis=0),
                self.data.wcs,
                gaia_stars=self.gaia_cat[~np.array(star_flag)],
                pixel_scale=self.pixel_scale,
                gaia_bright=19.5,
                mask_a=694.7 * scale_factor,
                mask_b=3.8,
                factor_b=0.8,
                factor_f=0.6,
                tigress=self.tigress,
                logger=self.logger)
        else:
            self.star_cat = []
            self.msk_star = np.copy(self.msk_star_ori)

    def _cpct_obj_detection(self, high_freq_lvl=2, wavelet_lvl=4, deblend_cont=0.03):
        # This step masks out high frequency sources (compact objects) by doing wavelet transformation
        obj_cat, segmap_cpct, bg_rms = kz.detection.makeCatalog(
            [self.data],
            mask=self.msk_star,
            lvl=2.5,
            method='wavelet',
            high_freq_lvl=high_freq_lvl,
            wavelet_lvl=wavelet_lvl,
            match_gaia=False,
            show_fig=self.show_figure,
            visual_gaia=False,
            b=24,
            f=3,
            pixel_scale=self.pixel_scale,
            minarea=3,
            deblend_nthresh=30,
            deblend_cont=deblend_cont,
            sky_subtract=True,
            logger=self.logger)

        catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
        dist = self.cen_obj['coord'].separation(catalog_c)
        cen_indx_highfreq = obj_cat[np.argsort(dist)[0]]['index']

        # Don't mask out objects that fall in the segmap of the central object and the Starlet box
        segmap = segmap_cpct.copy()
        segbox = segmap[self.starlet_extent[2]:self.starlet_extent[3],
                        self.starlet_extent[0]:self.starlet_extent[1]]
        # overlap_flag is for objects which fall in the footprint
        # of central galaxy in the fist SEP detection
        overlap_flag = [(self.segmap_ori == (self.cen_obj['idx'] + 1))[item]
                        for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
        overlap_flag = np.array(overlap_flag)

        # box_flat is for objects which fall in the initial Starlet box
        box_flag = np.unique(segbox) - 1
        if len(box_flag) > 0:
            box_flag = np.delete(np.sort(box_flag), 0)
            overlap_flag[box_flag] = True
        if len(overlap_flag) > 0:
            # obj_cat_cpct is the catalog for compact sources
            obj_cat_cpct = obj_cat[overlap_flag]

        # Remove the source from `obj_cat_cpct` if it is the central galaxy
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
        # This `seg_mask_cpct` only masks compact sources
        self.segmap_cpct = segmap_cpct
        self.seg_mask_cpct = (mask_conv >= gaussian_threshold)
        self.obj_cat_cpct = obj_cat_cpct

    def _big_obj_detection(self, lvl=4.0, b=48, f=3, deblend_cont=0.01):
        # This step masks out bright and large contamination, which is not well-masked in previous step
        obj_cat, segmap_big, bg_rms = kz.detection.makeCatalog(
            [self.data],
            lvl=lvl,  # relative agressive threshold
            method='vanilla',
            match_gaia=False,
            show_fig=self.show_figure,
            visual_gaia=False,
            b=b,
            f=f,
            pixel_scale=self.pixel_scale,
            minarea=20,   # only want large things
            deblend_nthresh=36,
            deblend_cont=deblend_cont,
            sky_subtract=True,
            logger=self.logger)

        catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
        dist = self.cen_obj['coord'].separation(catalog_c)
        cen_indx_big = obj_cat[np.argmin(dist)]['index'] if np.min(dist) < 1. * np.sqrt(
            self.cen_obj['a'] * self.cen_obj['b']) * self.pixel_scale * u.arcsec else -1
        # sometimes the central obj are not identified as big obj.

        # mask out big objects that are NOT identified in the high_freq step
        segmap = segmap_big.copy()
        segbox = segmap[self.starlet_extent[2]:self.starlet_extent[3],
                        self.starlet_extent[0]:self.starlet_extent[1]]
        box_flag = np.unique(segbox) - 1
        if len(box_flag) > 0:
            box_flag = np.delete(np.sort(box_flag), 0)
            for ind in box_flag:
                if np.sum(segbox == ind + 1) / np.sum(segmap == ind + 1) > 0.5:
                    segmap[segmap == ind + 1] = 0
            box_flag = np.delete(box_flag, np.where(box_flag == cen_indx_big)[
                0])  # dont include the central galaxy
            obj_cat_big = obj_cat[box_flag]
        else:
            obj_cat_big = obj_cat
        # `obj_cat_big` is catalog of the big and high SNR objects in the image

        smooth_radius = 5
        gaussian_threshold = 0.01
        mask_conv = np.copy(segmap)
        mask_conv[mask_conv > 0] = 1
        mask_conv = convolve(mask_conv.astype(
            float), Gaussian2DKernel(smooth_radius))
        # This `seg_mask_big` masks large bright sources
        self.segmap_big = segmap_big
        self.seg_mask_big = (mask_conv >= gaussian_threshold)
        self.obj_cat_big = obj_cat_big

    def _merge_catalogs(self):
        # Remove compact objects that are too close to the central
        # We don't want to shred the central galaxy
        catalog_c = SkyCoord(
            self.obj_cat_cpct['ra'], self.obj_cat_cpct['dec'], unit='deg')
        dist = self.cen_obj['coord'].separation(catalog_c)
        self.obj_cat_cpct.remove_rows(np.where(dist < 3 * u.arcsec)[0])

        # Remove objects in `obj_cat_cpct` that are already masked!
        # (since our final mask is combined from three masks)
        inside_flag = [
            self.seg_mask_big[item] for item in list(
                zip(self.obj_cat_cpct['y'].astype(int), self.obj_cat_cpct['x'].astype(int)))
        ]
        self.obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

        # Remove big objects that are toooo near to the target
        catalog_c = SkyCoord(
            self.obj_cat_big['ra'], self.obj_cat_big['dec'], unit='deg')
        dist = self.cen_obj['coord'].separation(catalog_c)
        # obj_cat_big.remove_rows(np.where(dist < 3 * u.arcsec)[0])
        self.obj_cat_big.remove_rows(np.where(
            dist < 1. * np.sqrt(self.cen_obj['a'] * self.cen_obj['b']) * self.pixel_scale * u.arcsec)[0])  # 2 times circularized effective radius

        # Remove objects in `obj_cat_big` that are already masked!
        inside_flag = [
            (self.data.weights[0] == 0)[item] for item in list(
                zip(self.obj_cat_big['y'].astype(int), self.obj_cat_big['x'].astype(int)))
        ]
        self.obj_cat_big.remove_rows(np.where(inside_flag)[0])

    def _construct_obs_frames(self):
        # Set weights of masked pixels to zero
        for layer in self.data.weights:
            layer[self.msk_star.astype(bool)] = 0
            layer[self.seg_mask_cpct.astype(bool)] = 0
            layer[self.seg_mask_big.astype(bool)] = 0

        # Construct `scarlet` frames and observation
        model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(self.data.channels))
        self.model_frame = scarlet.Frame(
            self.data.images.shape,
            wcs=self.data.wcs,
            psf=model_psf,
            channels=list(self.data.channels))
        observation = scarlet.Observation(
            self.data.images,
            wcs=self.data.wcs,
            psf=self.data.psfs,
            weights=self.data.weights,
            channels=list(self.data.channels))
        self.observation = observation.match(self.model_frame)
        # self.variance = np.array(
        #     np.mean(self.observation.noise_rms, axis=(1, 2))).mean()**2
        # convolve the `observation` with a gaussian kernel, to blur it
        # if self.method == 'vanilla':
        import sep
        from astropy.convolution import convolve, Gaussian2DKernel
        conv_data = np.zeros_like(self.data.images)
        for i in range(len(self.data.images)):
            if self.method == 'vanilla':
                input_data = convolve(
                    self.data.images[i].astype(float), Gaussian2DKernel(1.5))
                bkg = sep.Background(input_data, bw=50, bh=50, fw=3.5, fh=3.5)
            else:
                input_data = convolve(
                    self.data.images[i].astype(float), Gaussian2DKernel(4))
                bkg = sep.Background(input_data, bw=80,
                                     bh=80, fw=2.5, fh=2.5)
            input_data -= bkg.back()
            conv_data[i] = input_data
        observation = scarlet.Observation(
            conv_data,
            wcs=self.data.wcs,
            psf=self.data.psfs,
            weights=self.data.weights,
            channels=list(self.data.channels))
        self._conv_observation = observation.match(self.model_frame)

        # STARLET_MASK!!! contains the mask for irrelavant objects (also very nearby objects),
        # as well as larger bright star mask
        # This is used to help getting the SED initialization correct.
        # When estimating the starlet box and SED, we need to mask out saturated pixels and nearby bright stars.
        self.starlet_mask = ((np.sum(self.observation.weights == 0, axis=0)
                              != 0) + self.msk_star_ori + (~((self.segmap_ori == 0) | (self.segmap_ori == self.cen_obj['idx'] + 1)))).astype(bool)

    def _add_central_source(self, K=2, min_grad=0.02, thresh=0.1,
                            shifting=True, monotonic=True, variance=0.03**2,
                            scales=[0, 1, 2, 3, 4, 5, 6]):
        '''
        Add central source. The central source type depends on `self.method`.

        Parameters:
        -----------
        K (int): Number of components if MultiExtendedSource is invoked. Typically K=2.
        min_grad (float): Minimum gradient of profile in initializing the source.
        thresh (float): Threshold (# of std over background noise) in initializing the source.
        shifting (bool): Whether to allow the center of object to shift or not.
        monotonic (bool): Whether to enforce monotonicity of the profile. Only works for wavelet source.
        variance (float): Variance (actually std) below which non-monotnoicity is allowed.
            Only works for wavelet source.
        scales (list): For these wavelet scales, enforce monotonicity constraint, positive constraint,
            and L0 penalty (controlled by starlet_thresh).
            For other scales, we only enforce monotonicity, but remove the positive constraint.
            Only works for wavelet source and if `monotonic` is True.
        '''

        sources = []
        src = self.cen_obj

        self.monotonic = monotonic
        self.variance = variance
        self.scales = scales

        if self.method == 'vanilla':
            # Add central Vanilla source
            new_source = scarlet.source.ExtendedSource(
                self.model_frame, (src['ra'], src['dec']),
                self._conv_observation,
                satu_mask=self.data.masks,
                K=K, thresh=thresh, shifting=shifting, min_grad=min_grad)
            sources.append(new_source)
        elif self.method == 'wavelet':  # wavelet
            assert self.monotonic == monotonic, 'monotonic must be consistent with fitter.monotonic'
            # Find a better box, not too large, not too small
            upper = 0.1 if self.monotonic else 0.3
            min_grad_range = np.arange(
                0.02, upper, 0.02) if self.bright else np.arange(min_grad, upper, 0.02)

            # We calculate the ratio of contaminants' area over the box area
            # Then the box size is decided based on this ratio.
            contam_ratio_list = []
            for k, min_grad in enumerate(min_grad_range):
                if k == len(min_grad_range) - 1:
                    # if min_grad reaches its maximum, and `contam_ratio` is still very large,
                    # we choose the min_grad with the minimum `contam_ratio`
                    min_grad = min_grad_range[np.argmin(contam_ratio_list)]

                starlet_source = StarletSource(
                    self.model_frame,
                    (src['ra'], src['dec']),
                    self._conv_observation,  # self._conv_observation,
                    star_mask=self.starlet_mask,  # bright stars are masked when estimating morphology
                    satu_mask=self.data.masks,  # saturated pixels are masked when estimating SED
                    thresh=thresh,
                    min_grad=min_grad,
                    monotonic=monotonic,
                    starlet_thresh=self.starlet_thresh,
                    variance=variance,
                    scales=scales)
                starlet_extent = kz.display.get_extent(starlet_source.bbox)
                segbox = self.segmap_ori[starlet_extent[2]:starlet_extent[3],
                                         starlet_extent[0]:starlet_extent[1]]
                contam_ratio = 1 - \
                    np.sum((segbox == 0) | (segbox == self.cen_obj['idx'] + 1)) / \
                    np.sum(np.ones_like(segbox))
                if (contam_ratio <= 0.2 and (~self.smaller_box)) or (
                        contam_ratio <= 0.2 and (self.smaller_box or self.bright)):
                    break
                else:
                    contam_ratio_list.append(contam_ratio)

            self.logger.info(
                '  - Wavelet modeling with the following hyperparameters:')
            print(f'  - Wavelet modeling with the following hyperparameters:')
            self.logger.info(
                f'    min_grad = {min_grad:.2f}, starlet_thresh = {self.starlet_thresh:.2f} (contam_ratio = {contam_ratio:.2f}), \n     monotonic = {self.monotonic}, variance = {self.variance:.5f}, scales = {self.scales}.')
            print(
                f'    min_grad = {min_grad:.2f}, starlet_thresh = {self.starlet_thresh:.2f} (contam_ratio = {contam_ratio:.2f}), \n     monotonic = {self.monotonic}, variance = {self.variance:.5f}, scales = {self.scales}.'
            )
            starlet_source.center = (
                np.array(starlet_source.bbox.shape) // 2 + starlet_source.bbox.origin)[1:]
            sources.append(starlet_source)

        elif self.method == 'spergel':
            from .measure import g1g2, flux_radius_array
            # We first initialize an ExtendedSource to guess the initial parameters
            new_source = scarlet.SingleExtendedSource(
                self.model_frame,
                (src['ra'], src['dec']),
                self.observation,
                satu_mask=self.data.masks,
                thresh=thresh,
                shifting=True,
                min_grad=min_grad)
            morph = new_source.morphology
            SED = np.array(new_source.spectrum * new_source.morphology.sum())
            g1, g2 = g1g2(np.array(morph))
            rhalf = flux_radius_array(np.array(morph), 0.45) * 1.5
            nu = np.array([0.5])

            david = scarlet.SpergelSource(
                self.model_frame,
                (src['ra'], src['dec']),
                nu, rhalf, np.array((g1[0], g2[0])),
                self.observation,
                SED=SED)
            sources.append(david)
            self.logger.info(
                f'  - Added Spergel profile with bbox = {david.bbox.shape}')
            print(f'  - Added Spergel profile with bbox = {david.bbox.shape}')

        return sources

    def _add_sources(self, K=2, min_grad=0.02,
                     thresh=0.1, shifting=True):
        '''
        Add all sources. Central source type depends on `self.method`.

        Parameters:
        -----------
        K (int): Number of central components if MultiExtendedSource is invoked. Typically K=2.
        min_grad (float): Minimum gradient of profile in initializing the central source.
        thresh (float): Threshold (# of std over background noise) in initializing the central source.
        variance (float): For central source, variance (actually std) below which non-monotnoicity is allowed.
        shifting (bool): Whether to allow the center of central object to shift or not.
        monotonic (bool): Whether to enforce monotonicity of the profile. Only works for wavelet source.
        scales (list): For these wavelet scales, enforce monotonicity constraint, positive constraint,
            and L0 penalty (controlled by starlet_thresh).
            For other scales, we only enforce monotonicity, but remove the positive constraint.
            Only works for wavelet source and if `monotonic` is True.
        bkg (bool): Whether to add a sky background source.
        '''
        sources = self._add_central_source(K=K, min_grad=min_grad,
                                           thresh=thresh, shifting=shifting,
                                           monotonic=self.monotonic,
                                           variance=self.variance, scales=self.scales)

        # Only model "real compact" sources
        if len(self.obj_cat_big) > 0 and len(self.obj_cat_cpct) > 0:
            # remove intersection between cpct and big objects
            # if an object is both cpct and big, we think it is big
            cpct_coor = SkyCoord(
                ra=np.array(self.obj_cat_cpct['ra']) * u.degree,
                dec=np.array(self.obj_cat_cpct['dec']) * u.degree)
            big = SkyCoord(ra=self.obj_cat_big['ra'] * u.degree,
                           dec=self.obj_cat_big['dec'] * u.degree)
            tempid, sep2d, _ = match_coordinates_sky(big, cpct_coor)
            cpct = self.obj_cat_cpct[np.setdiff1d(
                np.arange(len(self.obj_cat_cpct)), tempid[np.where(sep2d < 1 * u.arcsec)])]
        else:
            cpct = self.obj_cat_cpct

        if len(self.star_cat) > 0 and len(cpct) > 0:
            # remove intersection between cpct and stars
            # if an object is both cpct and star, we think it is star
            star = SkyCoord(
                ra=self.star_cat['ra'], dec=self.star_cat['dec'], unit='deg')
            cpct_coor = SkyCoord(
                ra=np.array(cpct['ra']) * u.degree,
                dec=np.array(cpct['dec']) * u.degree)
            tempid, sep2d, _ = match_coordinates_sky(star, cpct_coor)
            cpct = cpct[np.setdiff1d(np.arange(len(cpct)),
                                     tempid[np.where(sep2d < 1 * u.arcsec)])]

        if not (self.bright | self.smaller_box):
            # for bright galaxy, we don't include these compact sources into modeling,
            # due to the limited computation resources
            for k, src in enumerate(cpct):
                if src['fwhm_custom'] < 5:
                    new_source = scarlet.source.PointSource(
                        self.model_frame, (src['ra'], src['dec']), self.observation)
                elif src['fwhm_custom'] >= 5 and src['fwhm_custom'] < 10:
                    new_source = scarlet.source.CompactExtendedSource(
                        self.model_frame, (src['ra'], src['dec']), self.observation)
                else:
                    new_source = scarlet.source.SingleExtendedSource(
                        self.model_frame, (src['ra'],
                                           src['dec']), self.observation,
                        thresh=2, min_grad=0.2)
                sources.append(new_source)
        # IF GAIA stars are within the box: exclude it from the big_cat
        if len(self.obj_cat_big) > 0:
            if len(self.star_cat) > 0:
                star = SkyCoord(
                    ra=self.star_cat['ra'], dec=self.star_cat['dec'], unit='deg')
                tempid, sep2d, _ = match_coordinates_sky(big, star)
                # tempid, sep2d, _ = match_coordinates_sky(star, big)
                big_cat = self.obj_cat_big[np.setdiff1d(
                    np.arange(len(self.obj_cat_big)), np.where(sep2d < 1.5 * u.arcsec)[0])]
                # big_cat = obj_cat_big[np.setdiff1d(
                #     np.arange(len(obj_cat_big)), tempid[np.where(sep2d < 1 * u.arcsec)])]
            else:
                big_cat = self.obj_cat_big

            self._big_cat = big_cat

            for k, src in enumerate(big_cat):
                if src['fwhm_custom'] > 24:
                    new_source = scarlet.source.ExtendedSource(
                        self.model_frame, (src['ra'], src['dec']),
                        self.observation,
                        K=2, thresh=3, shifting=True, min_grad=0.2)
                else:
                    try:
                        new_source = scarlet.source.SingleExtendedSource(
                            self.model_frame, (src['ra'], src['dec']),
                            self.observation, satu_mask=self.data.masks,  # helps to get SED correct
                            thresh=3, shifting=False, min_grad=0.2)
                    except Exception as e:
                        self.logger.info(f'   ! Error: {e}')
                sources.append(new_source)

        # Add GAIA stars as Extended Sources
        if len(self.star_cat) > 0:
            for k, src in enumerate(self.star_cat):
                try:
                    if src['phot_g_mean_mag'] < 18:
                        new_source = scarlet.source.ExtendedSource(
                            self.model_frame, (src['ra'], src['dec']),
                            self.observation,
                            K=2, thresh=4, shifting=True, min_grad=0.4)
                    else:
                        new_source = scarlet.source.SingleExtendedSource(
                            self.model_frame, (src['ra'], src['dec']),
                            self.observation, satu_mask=self.data.masks,
                            thresh=2, shifting=False, min_grad=0.)
                except Exception as e:
                    self.logger.info(f'   ! Error: {e}')
                # only use SingleExtendedSource
                sources.append(new_source)

        # Add constant sky bkg
        if self.bkg == True:
            new_source = scarlet.ConstSkySource(
                self.model_frame,
                bbox=sources[0].bbox,
                observations=self.observation)
            new_source = scarlet.ConstWholeSkySource(
                self.model_frame,
                observations=self.observation)
            sources.append(new_source)
            print('    Added constant sky background')
            self.logger.info('    Added constant sky background')
        self._sources = sources
        self.blend = scarlet.Blend(self._sources, self.observation)

        print(f'    Total number of sources: {len(sources)}')
        self.logger.info(f'    Total number of sources: {len(sources)}')

    def _optimize(self):
        # Star fitting!
        start = time.time()
        fig = kz.display.display_scarlet_model(
            self.blend,
            minimum=-0.3,
            stretch=1,
            channels=self.data.channels,
            show_loss=False,
            show_mask=False,
            show_mark=True,
            scale_bar=False)
        plt.savefig(
            os.path.join(self.figure_dir, f'{self.prefix}-{self.index}-init-{self.method}.png'), dpi=70, bbox_inches='tight')
        if not self.show_figure:
            plt.close()

        [self.blend, self.best_logL, self.best_erel, self.best_epoch, self._blend] = _optimization(
            self.blend, bright=self.bright, logger=self.logger, bkg=self.bkg)
        with open(os.path.join(self.model_dir, f'{self.prefix}-{self.index}-trained-model-{self.method}.df'), 'wb') as fp:
            dill.dump(
                [self.blend, {'starlet_thresh': self.starlet_thresh,
                              'monotonic': self.monotonic,
                              'variance': self.variance,
                              'scales': self.scales,
                              'e_rel': self.best_erel,
                              'loss': self.best_logL}, None], fp)
            fp.close()

        end = time.time()
        self.logger.info(
            f'    Elapsed time for fitting: {(end - start):.2f} s')
        print(f'    Elapsed time for fitting: {(end - start):.2f} s')

    def _find_sed_ind(self):
        # In principle, Now we don't need to find which components compose a galaxy. The central Starlet is enough!
        if len(self.blend.sources) > 1:
            mag_mat = np.array(
                [-2.5 * np.log10(kz.measure.flux(src, self.observation)) + 27 for src in self.blend.sources])
            # g - r, g - i, g - z
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:]
            color_dist = np.linalg.norm(
                color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            sed_ind = np.where(color_dist < 0.1)[0]
            dist = np.array([
                np.linalg.norm(
                    src.center - self.blend.sources[0].center) * self.pixel_scale
                for src in np.array(self.blend.sources)[sed_ind]
            ])
            dist_flag = (
                dist < 3 * np.sqrt(self.cen_obj['a'] * self.cen_obj['b']) * self.pixel_scale)

            # maybe use segmap flag? i.e., include objects that are overlaped
            # with the target galaxy in the inital detection.

            point_flag = np.array([
                isinstance(src, scarlet.source.PointSource)
                for src in np.array(self.blend.sources)[sed_ind]
            ])  # we don't want point source

            near_cen_flag = [
                (self.segmap_ori == self.cen_obj['idx'] +
                 1)[int(src.center[0]), int(src.center[1])]  # src.center: [y, x]
                for src in np.array(self.blend.sources)[sed_ind]
            ]

            bkg_flag = np.array([isinstance(src, scarlet.source.ConstSkySource)
                                 for src in np.array(self.blend.sources)[sed_ind]])

            sed_ind = sed_ind[(~point_flag) & near_cen_flag &
                              dist_flag & (~bkg_flag)]

            if not 0 in sed_ind:
                # the central source must be included.
                sed_ind = np.array(list(set(sed_ind).union({0})))
        else:
            sed_ind = np.array([0])

        self.sed_ind = sed_ind
        self.logger.info(
            f'  - Components {sed_ind} are considered as the target galaxy.')
        print(
            f'  - Components {sed_ind} are considered as the target galaxy.')

    def _gen_final_mask(self):
        ############################################
        ################# Final mask ##################
        ############################################
        # Only mask bright stars!!!
        self.logger.info(
            '  - Masking stars and other sources that are modeled, to deal with leaky flux issue.')
        print('  - Masking stars and other sources that are modeled, to deal with leaky flux issue.')
        # Generate a VERY AGGRESSIVE mask, named "footprint"
        footprint = np.zeros_like(self.segmap_cpct, dtype=bool)
        # for ind in cpct['index']:  # mask ExtendedSources which are modeled
        #     footprint[segmap_cpct == ind + 1] = 1

        # footprint[segmap_cpct == cen_indx_highfreq + 1] = 0
        sed_ind_pix = np.array([item.center for item in np.array(
            self.blend.sources)[self.sed_ind]]).astype(int)  # the y and x of sed_ind objects
        # # if any objects in `sed_ind` is in `segmap_cpct`
        # sed_corr_indx = segmap_cpct[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
        # for ind in sed_corr_indx:
        #     footprint[segmap_cpct == ind] = 0

        # smooth_radius = 1.5
        # gaussian_threshold = 0.03
        # mask_conv = np.copy(footprint)
        # mask_conv[mask_conv > 0] = 1
        # mask_conv = convolve(mask_conv.astype(
        #     float), Gaussian2DKernel(smooth_radius))
        # footprint = (mask_conv >= gaussian_threshold)

        # Mask star within the box
        if len(self.star_cat) > 0:
            _, star_mask = kz.utils.gaia_star_mask(  # Generate GAIA mask only for stars outside of the Starlet box
                self.data.images.mean(axis=0),
                self.data.wcs,
                gaia_stars=self.star_cat,
                pixel_scale=self.pixel_scale,
                gaia_bright=19,
                mask_a=694.7,
                mask_b=3.8,
                factor_b=0.9,
                factor_f=1.1,
                tigress=self.tigress)
            footprint = footprint | star_mask

        # Mask big objects from `big_cat`
        if len(self.obj_cat_big) > 0 and hasattr(self, '_big_cat'):
            # Blow-up radius depends on the distance to target galaxy
            catalog_c = SkyCoord(
                self._big_cat['ra'], self._big_cat['dec'], unit='deg')
            dist = self.cen_obj['coord'].separation(catalog_c)
            if self.method == 'wavelet':
                near_flag = (
                    dist < 2 * self.cen_obj['a'] * self.pixel_scale * u.arcsec)
            else:
                near_flag = (
                    dist < 4 * self.cen_obj['a'] * self.pixel_scale * u.arcsec)

            footprint2 = np.zeros_like(self.segmap_big, dtype=bool)
            # mask ExtendedSources which are modeled
            for ind in self._big_cat[near_flag]['index']:
                footprint2[self.segmap_big == ind + 1] = 1

            # if any objects in `sed_ind` is in `segmap_big`
            sed_corr_indx = self.segmap_big[sed_ind_pix[:,
                                                        0], sed_ind_pix[:, 1]]
            for ind in sed_corr_indx:
                footprint2[self.segmap_big == ind] = 0
            footprint2[self.segmap_big == self.cen_obj['idx'] + 1] = 0

            smooth_radius = 1.5
            gaussian_threshold = 0.1
            mask_conv = np.copy(footprint2)
            mask_conv[mask_conv > 0] = 1
            mask_conv = convolve(mask_conv.astype(
                float), Gaussian2DKernel(smooth_radius))
            footprint2 = (mask_conv >= gaussian_threshold)

            footprint3 = np.zeros_like(self.segmap_big, dtype=bool)
            # mask ExtendedSources which are modeled
            for ind in self._big_cat[~near_flag]['index']:
                footprint3[self.segmap_big == ind + 1] = 1
            smooth_radius = 5
            gaussian_threshold = 0.01
            mask_conv = np.copy(footprint3)
            mask_conv[mask_conv > 0] = 1
            mask_conv = convolve(mask_conv.astype(
                float), Gaussian2DKernel(smooth_radius))
            footprint3 = (mask_conv >= gaussian_threshold)

            footprint2 += footprint3  # combine together

            #     # if any objects in `sed_ind` is in `segmap_big`
            # sed_corr_indx = segmap_big[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
            # for ind in sed_corr_indx:
            #     footprint2[segmap_big == ind] = 0
            # footprint2[segmap_big == cen_indx_big + 1] = 0

            # This is the mask for everything except target galaxy
            footprint = footprint + footprint2

        # Deal with non-monotonic flux
        if self.method == 'wavelet':  # and self.monotonic is True:
            src = self.blend.sources[0]
            img = self.observation.render(src.get_model())
            center = tuple(s // 2 for s in src.bbox.shape[1:])
            prox = prox_monotonic_mask(
                img[0], 1e-3,
                center=center,
                zero=0,
                center_radius=4,
                variance=1e-5,  # 1e-4,
                max_iter=10)
            smooth_radius = 3
            gaussian_threshold = 0.02
            mask_conv = np.copy(~prox[0])
            mask_conv = convolve(mask_conv.astype(
                float), Gaussian2DKernel(smooth_radius))
            mask_conv = (mask_conv > gaussian_threshold)
            monomask = np.zeros_like(footprint)
            monomask[src.bbox.origin[1]:src.bbox.origin[1] + src.bbox.shape[1],
                     src.bbox.origin[2]:src.bbox.origin[2] + src.bbox.shape[2]] = mask_conv
            monomask = monomask.astype(bool)

            footprint = footprint + monomask

        self.final_mask = footprint

        outdir = os.path.join(
            self.model_dir, f'{self.prefix}-{self.index}-trained-model-{self.method}.df')
        self.logger.info(
            f'  - Saving the results as {os.path.abspath(outdir)}')
        print(f'  - Saving the results as {os.path.abspath(outdir)}')
        with open(os.path.abspath(outdir), 'wb') as fp:
            dill.dump(
                [self.blend, {'starlet_thresh': self.starlet_thresh,
                              'e_rel': self.best_erel,
                              'epoch': self.best_epoch,
                              'loss': self.best_logL,
                              'sed_ind': self.sed_ind}, footprint], fp)
            fp.close()

    def _display_results(self, stretch=0.9):
        # Save fitting figure
        # zoomin_size: in arcsec, rounded to integer multiple of 30 arcsec
        zoomin_size = np.ceil(
            (self.blend.sources[0].bbox.shape[1] * self.pixel_scale * 3) / 30) * 20
        # cannot exceed the image size
        zoomin_size = min(
            zoomin_size, self.data.images.shape[1] * self.pixel_scale)

        fig = kz.display.display_scarlet_results_tigress(
            self.blend,
            self.final_mask,
            show_ind=self.sed_ind,
            zoomin_size=zoomin_size,
            minimum=-0.2,
            stretch=stretch,
            Q=1,
            channels=self.data.channels,
            show_loss=True,
            show_mask=False,
            show_mark=False,
            scale_bar=True,
            add_text=f'{self.prefix}-{self.index}',
            text_fontsize=20,)
        plt.savefig(
            os.path.join(self.figure_dir, f'{self.prefix}-{self.index}-zoomin-{self.method}.png'), dpi=55, bbox_inches='tight')
        if not self.show_figure:
            plt.close()

    def fit(self):
        print('  - Detect sources and make mask')
        self.logger.info('  - Detect sources and make mask')
        try:
            # First Gaia star search
            self._first_gaia_search()

            # Set the weights of saturated star centers to zero
            # In order to make the box size estimation more accurate.
            temp = np.copy(self.data.masks)
            for i in range(len(self.data.channels)):
                temp[i][~self.msk_star_ori.astype(bool)] = 0
                self.data.weights[i][temp[i].astype(bool)] = 0.0

            # Replace the vanilla detection with a convolved vanilla detection
            first_dblend_cont = 0.07 if max(
                self.data.images.shape) * self.pixel_scale > 200 else 0.006
            if self.method == 'wavelet':
                first_dblend_cont = 0.07 if max(
                    self.data.images.shape) * self.pixel_scale > 200 else 0.002
            self._first_detection(first_dblend_cont)

            self._estimate_box(self.cen_obj)
            self._mask_stars_outside_box()
            self._cpct_obj_detection()
            self._big_obj_detection()
            self._merge_catalogs()
            self._construct_obs_frames()
            if self.bright:
                self.variance = 0.05**2
                self.scales = [0, 1, 2, 3, 4, 5, 6]
                self.starlet_thresh = 0.5
                self.min_grad = 0.01

            self._add_sources(min_grad=self.min_grad, thresh=0.1)

            if self.show_figure:
                fig = kz.display.display_scarlet_sources(
                    self.data,
                    self._sources,
                    show_ind=None,
                    stretch=1,
                    Q=1,
                    minimum=-0.2,
                    show_mark=True,
                    scale_bar_length=10,
                    add_text=f'{self.prefix}-{self.index}')

            self._optimize()
            self._find_sed_ind()
            self._gen_final_mask()  # also save final results to file
            self._display_results()

            self.logger.info('Done! (♡˙︶˙♡)')
            self.logger.info('\n')

            return self.blend

        except Exception as e:
            raise ScarletFittingError(self.prefix, self.index, self.starlet_thresh, self.monotonic,
                                      self.data.channels, traceback.print_exc())


def fitting_obs_tigress(env_dict, lsbg, name='Seq', channels='griz',
                        method='vanilla',
                        min_grad=-0.02,
                        starlet_thresh=0.5, monotonic=True, bkg=True, scales=[0, 1, 2, 3, 4, 5], variance=0.07**2,
                        pixel_scale=HSC_pixel_scale, bright_thresh=17.5,
                        prefix='candy', model_dir='./Model', figure_dir='./Figure', log_dir='./log',
                        show_figure=False, global_logger=None, fail_logger=None):
    '''
    Run scarlet wavelet modeling on Tiger, modified on 01/04/2021.

    Parameters:
        env_dict (dict): dictionary indicating the file directories, such as
            `env_dict = {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/tigress/jiaxuanl/Data'}`
        lsbg (one row in `astropy.Table`): the galaxy to be modeled.
        name (str): the column name for the index of `lsbg`.
        channels (str): bandpasses to be used, such as 'grizy'.
        method (str): the method to be used for fitting. Either "vanilla" or "wavelet".
        starlet_thresh (float): this number controls how many high-frequency components are remained in the model.
            Larger value gives smoother modeling. Typically we use 0.5 to 1.
        pixel_scale (float): pixel scale of the input image, in arcsec / pixel. Default is for HSC.
        bright_thresh (float): magnitude threshold for bright galaxies, default is 17.0.
        prefix (str): the prefix for output files.
        model_dir (str): directory for output modeling files.
        figure_dir (str): directory for output figures.
        show_figure (bool): if True, show the figure displaying fitting results.
        logger (`logging.Logger`): a dedicated robot who writes down the log.
            If not provided, a Logger will be generated automatically.
        global_logger (`logging.Logger`): the logger used to pass the log within this function to outside.
        fail_logger (`logging.Logger`): the logger used to take notes for failed cases.

    Returns:
        blend (scarlet.blend object)

    '''
    __name__ = f"fitting_{method}_obs_tigress"

    from kuaizi.utils import padding_PSF
    from kuaizi.mock import Data

    index = lsbg[name]
    # whether this galaxy is a very bright one
    bright = (lsbg['mag_auto_i'] < bright_thresh)

    fitter = ScarletFitter(method=method,
                           tigress=True,
                           bright=bright,
                           min_grad=min_grad,
                           starlet_thresh=starlet_thresh,
                           monotonic=monotonic,
                           bkg=bkg,
                           scales=scales,
                           variance=variance,
                           log_dir=log_dir,
                           figure_dir=figure_dir,
                           model_dir=model_dir,
                           pixel_scale=pixel_scale,
                           prefix=prefix,
                           index=index,
                           show_figure=show_figure)

    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    # if not os.path.isdir(figure_dir):
    #     os.makedirs(figure_dir)

    fitter.logger.info(
        f'Running scarlet {method} modeling for `{lsbg["prefix"]}`')
    print(f'### Running scarlet {method} modeling for `{lsbg["prefix"]}`')

    if bright:
        fitter.logger.info(
            f"This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")
        print(
            f"    This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")

    fitter.logger.info(f'Working directory: {os.getcwd()}')
    print(f'    Working directory: {os.getcwd()}')

    kz.utils.set_env(**env_dict)
    kz.utils.set_matplotlib(style='default')

    try:
        #### Deal with possible exceptions on bandpasses ###
        assert isinstance(channels, str), 'Input channels must be a string!'
        if len(set(channels) & set('grizy')) == 0:
            raise ValueError('The input channels must be a subset of "grizy"!')

        overlap = [i for i, item in enumerate('grizy') if item in channels]

        #### Deal with possible exceptions on files (images, PSFs) ###
        file_exist_flag = np.all(lsbg['image_flag'][overlap]) & np.all(
            [os.path.isfile(f"{lsbg['prefix']}_{filt}.fits") for filt in channels])
        if not file_exist_flag:
            raise FileExistsError(
                f'The image files of `{lsbg["prefix"]}` in `{channels}` are not complete! Please check!')

        file_exist_flag = np.all(lsbg['psf_flag'][overlap]) & np.all(
            [os.path.isfile(f"{lsbg['prefix']}_{filt}_psf.fits") for filt in channels])
        default_exist_flag = np.all([os.path.isfile(
            f'/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/psf_{filt}.fits') for filt in channels])
        # This is the flag confirming the default PSFs exist.

        if not file_exist_flag:
            fitter.logger.info(
                f'The PSF files of `{lsbg["prefix"]}` in `{channels}` are not complete! Please check!')

            if default_exist_flag:
                fitter.logger.info(f'We use the default HSC PSFs instead.')
                psf_list = [fits.open(f'/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/psf_{filt}.fits')
                            for filt in channels]
            else:
                raise FileExistsError(
                    f'The PSF files of `{lsbg["prefix"]}` in `{channels}` are not complete! The default PSFs are also missing! Please check!')
        else:
            psf_list = [fits.open(f"{lsbg['prefix']}_{filt}_psf.fits")
                        for filt in channels]

        # Structure the data
        lsbg_coord = SkyCoord(ra=lsbg['ra'], dec=lsbg['dec'], unit='deg')
        cutout = [fits.open(f"{lsbg['prefix']}_{filt}.fits")
                  for filt in channels]

        images = np.array([hdu[1].data for hdu in cutout])
        # note: all bands share the same WCS here, but not necessarily true.
        w = wcs.WCS(cutout[0][1].header)
        weights = 1.0 / np.array([hdu[3].data for hdu in cutout])
        weights[np.isinf(weights)] = 0.0
        psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
        psfs = scarlet.ImagePSF(np.array(psf_pad))
        # saturation mask and interpolation mask from HSC S18A
        sat_mask = np.array([sum(unagi.mask.Mask(
            hdu[2].data, data_release='s18a').extract(['INTRP', 'SAT'])) for hdu in cutout])
        data = Data(images=images, weights=weights, masks=sat_mask,
                    wcs=w, psfs=psfs, channels=channels)

        fitter.load_data(data, lsbg_coord)

        # Collect free RAM
        del cutout, psf_list
        del images, w, weights, psf_pad, psfs
        gc.collect()

        blend = fitter.fit()

        if global_logger is not None:
            global_logger.info(
                f'{method} Task succeeded for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        gc.collect()

        return blend

    except Exception as e:
        fitter.logger.error(traceback.print_exc())
        print(traceback.print_exc())
        if bright:
            fitter.logger.error(
                f'{method} Task failed for BRIGHT galaxy `{lsbg["prefix"]}`')
        else:
            fitter.logger.error(f'{method} Task failed for `{lsbg["prefix"]}`')

        fitter.logger.info('\n')
        if fail_logger is not None:
            fail_logger.error(
                f'{method} Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        if global_logger is not None:
            global_logger.error(
                f'{method} Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        return
