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
from scarlet.source import StarletSource

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


def _optimization(blend, bright=False, logger=None):
    if bright:
        e_rel_list = [1e-4, 5e-4, 1e-5]  # otherwise it will take forever....
        n_iter = 100
    else:
        e_rel_list = [1e-4, 5e-4]  # , 5e-5, 1e-5
        n_iter = 100

    blend.fit(1, 1e-3)  # First iteration, just to get the inital loss

    best_model = blend
    best_logL = -blend.loss[-1]
    best_erel = 1e-3
    best_epoch = 1

    for i, e_rel in enumerate(e_rel_list):
        for k in range(n_iter // 10):
            blend.fit(10, e_rel)
            if -blend.loss[-1] > best_logL:
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

    return [best_model, best_logL, best_erel, best_epoch]


class ScarletFittingError(Exception):
    """Exception raised for errors when fitting a galaxy using scarlet.

    Attributes:
        prefix, index, coord, channels
        message -- explanation of the error
    """

    def __init__(self, prefix, index, starlet_thresh, channels, error):
        self.prefix = prefix
        self.index = index
        self.starlet_thresh = starlet_thresh
        self.channels = channels
        self.error = error
        super().__init__(self.error)

    def __str__(self):
        return f'{self.prefix}-{self.index} in `{self.channels}` bands with `starlet_thresh = {self.starlet_thresh}` -> {self.error}'


class ScarletFitter(object):
    def __init__(self, method='vanilla', tigress=True, bright=False, **kwargs) -> None:
        """Initialize a ScarletFitter object.

        Args:
            method (str, optional): method used in scarlet fitting, either 'vanilla' or 'wavelet'. 
                Defaults to 'vanilla'.
            tigress (bool, optional): whether to use Tigress data. Defaults to True.
            **kwargs: keyword arguments for ScarletFitter.
        """
        self.method = method
        self.tigress = tigress
        self.bright = bright

        self.log_dir = kwargs.get('log_dir', './log')
        self.figure_dir = kwargs.get('figure_dir', './Figure')
        self.model_dir = kwargs.get('model_dir', './Model')

        self.prefix = kwargs.get('prefix', 'test')
        self.index = kwargs.get('index', 0)
        self.pixel_scale = kwargs.get('pixel_scale', HSC_pixel_scale)
        self.zeropoint = kwargs.get('zeropoint', HSC_zeropoint)
        self.show_figure = kwargs.get('show_figure', False)
        self.starlet_thresh = kwargs.get('starlet_thresh', 0.5)

        self._set_logger()

    def load_data(self, data, coord):
        self.logger.info('    Load data')
        self.data = data
        self.coord = coord
        self.bright = False

    def _set_logger(self):
        from .utils import set_logger
        logger = set_logger(f'fitting_{self.method}', os.path.join(
            self.log_dir, f'{self.prefix}-{self.index}.log'))
        self.logger = logger

    def _first_gaia_search(self):
        print('    Query GAIA stars...')
        self.logger.info('    Query GAIA stars...')
        # Generate a mask for GAIA bright stars
        self.gaia_cat, self.msk_star_ori = kz.utils.gaia_star_mask(
            self.data.images.mean(axis=0),  # averaged image
            self.data.wcs,
            pixel_scale=self.pixel_scale,
            gaia_bright=19.5,
            mask_a=694.7,
            mask_b=3.8,
            factor_b=1.0,  # 0.7,
            factor_f=1.4,  # 1.0,
            tigress=self.tigress,
            logger=self.logger)
        self.n_stars = len(self.gaia_cat)

    def _first_detection(self, first_dblend_cont):
        obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
            [self.data],
            lvl=4,  # a.k.a., "sigma"
            mask=self.msk_star_ori,
            method='vanilla',
            convolve=True,
            conv_radius=2,
            match_gaia=False,
            show_fig=self.show_figure,
            visual_gaia=False,
            b=80,
            f=3,
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
        x, y, _ = sep.winpos(self.data.images.mean(
            axis=0), cen_obj['x'], cen_obj['y'], 6)
        ra, dec = self.data.wcs.wcs_pix2world(x, y, 0)
        cen_obj = dict(cen_obj)
        cen_obj['x'] = x
        cen_obj['y'] = y
        cen_obj['ra'] = ra
        cen_obj['dec'] = dec
        cen_obj['idx'] = cen_indx_ori
        cen_obj['coord'] = SkyCoord(cen_obj['ra'], cen_obj['dec'], unit='deg')
        #cen_obj_coord = SkyCoord(cen_obj['ra'], cen_obj['dec'], unit='deg')

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

        starlet_source = StarletSource(model_frame,
                                       (cen_obj['ra'], cen_obj['dec']),
                                       observation,
                                       thresh=0.01,
                                       min_grad=-0.01,  # the initial guess of box size is as large as possible
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

    def _mask_stars_outside_box(self):
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
                mask_a=694.7,
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

    def _big_obj_detection(self, lvl=4.0, b=32, f=3, deblend_cont=0.01):
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
        cen_indx_big = obj_cat[np.argsort(dist)[0]]['index']  # obj_cat_ori

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

        # STARLET_MASK!!! contains the mask for irrelavant objects (also very nearby objects),
        # as well as larger bright star mask
        # This is used to help getting the SED initialization correct.
        # When estimating the starlet box and SED, we need to mask out saturated pixels and nearby bright stars.
        self.starlet_mask = ((np.sum(self.observation.weights == 0, axis=0)
                              != 0) + self.msk_star_ori + (~((self.segmap_ori == 0) | (self.segmap_ori == self.cen_obj['idx'] + 1)))).astype(bool)

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
        #obj_cat_big.remove_rows(np.where(dist < 3 * u.arcsec)[0])
        self.obj_cat_big.remove_rows(np.where(
            dist < 2 * np.sqrt(self.cen_obj['a'] * self.cen_obj['b']) * self.pixel_scale * u.arcsec)[0])  # 2 times circularized effective radius

        # Remove objects in `obj_cat_big` that are already masked!
        inside_flag = [
            (self.data.weights[0] == 0)[item] for item in list(
                zip(self.obj_cat_big['y'].astype(int), self.obj_cat_big['x'].astype(int)))
        ]
        self.obj_cat_big.remove_rows(np.where(inside_flag)[0])

    def _add_sources_vanilla(self, K=2, min_grad=-0.1, thresh=0.01, shifting=True):
        sources = []

        # Add central Vanilla source
        src = self.cen_obj
        new_source = scarlet.source.ExtendedSource(
            self.model_frame, (src['ra'], src['dec']),
            self.observation,
            satu_mask=self.data.masks,
            K=K, thresh=thresh, shifting=shifting, min_grad=min_grad)
        sources.append(new_source)

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

        if not self.bright:
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
                if src['fwhm_custom'] > 22:
                    new_source = scarlet.source.ExtendedSource(
                        self.model_frame, (src['ra'], src['dec']),
                        self.observation,
                        K=2, thresh=2, shifting=True, min_grad=0.2)
                else:
                    try:
                        new_source = scarlet.source.SingleExtendedSource(
                            self.model_frame, (src['ra'], src['dec']),
                            self.observation, satu_mask=data.masks,  # helps to get SED correct
                            thresh=2, shifting=False, min_grad=0.2)
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
                            self.observation, satu_mask=data.masks,
                            thresh=2, shifting=False, min_grad=0.)
                except Exception as e:
                    self.logger.info(f'   ! Error: {e}')
                # only use SingleExtendedSource
                sources.append(new_source)

        self._sources = sources

        print(f'    Total number of sources: {len(sources)}')
        self.logger.info(f'    Total number of sources: {len(sources)}')

    def _optimize(self):
        # Star fitting!
        start = time.time()
        self.blend = scarlet.Blend(self._sources, self.observation)
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

        [self.blend, self.best_logL, self.best_erel, self.best_epoch] = _optimization(
            self.blend, bright=self.bright, logger=self.logger)
        with open(os.path.join(self.model_dir, f'{self.prefix}-{self.index}-trained-model-{self.method}.df'), 'wb') as fp:
            dill.dump(
                [self.blend, {'starlet_thresh': self.starlet_thresh,
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

            sed_ind = sed_ind[(~point_flag) & near_cen_flag & dist_flag]

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

    def _display_results(self):
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
            minimum=-0.3,
            stretch=1,
            Q=1,
            channels=self.data.channels,
            show_loss=True,
            show_mask=False,
            show_mark=False,
            scale_bar=True)
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
            self._first_detection(first_dblend_cont)

            self._estimate_box(self.cen_obj)
            self._mask_stars_outside_box()
            self._cpct_obj_detection()
            self._big_obj_detection()
            self._merge_catalogs()
            self._construct_obs_frames()
            self._add_sources_vanilla()

            if self.show_figure:
                fig = kz.display.display_scarlet_sources(
                    self.data,
                    self._sources,
                    show_ind=None,
                    stretch=1,
                    Q=1,
                    minimum=-0.3,
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
            raise ScarletFittingError(self.prefix, self.index, self.starlet_thresh,
                                      self.data.channels, traceback.print_exc())


def fitting_vanilla_obs_tigress(env_dict, lsbg, name='Seq', channels='griz',
                                starlet_thresh=0.5, pixel_scale=HSC_pixel_scale, bright_thresh=17.0,
                                prefix='candy', model_dir='./Model', figure_dir='./Figure', log_dir='./log',
                                show_figure=False, logger=None, global_logger=None, fail_logger=None):
    '''
    Run scarlet wavelet modeling on Tiger, modified on 09/13/2021. 

    Parameters:
        env_dict (dict): dictionary indicating the file directories, such as 
            `env_dict = {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/tigress/jiaxuanl/Data'}`
        lsbg (one row in `astropy.Table`): the galaxy to be modeled. 
        name (str): the column name for the index of `lsbg`.
        channels (str): bandpasses to be used, such as 'grizy'.
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
    __name__ = "fitting_vanilla_obs_tigress"

    from kuaizi.utils import padding_PSF
    from kuaizi.mock import Data
    import unagi  # for HSC saturation mask

    index = lsbg[name]
    # whether this galaxy is a very bright one
    bright = (lsbg['mag_auto_i'] < bright_thresh)

    fitter = ScarletFitter(method='vaniila',
                           tigress=True,
                           bright=bright,
                           starlet_thresh=starlet_thresh,
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

    logger.info(f'Running scarlet vanilla modeling for `{lsbg["prefix"]}`')
    print(f'### Running scarlet vanilla modeling for `{lsbg["prefix"]}`')

    if bright:
        logger.info(
            f"This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")
        print(
            f"    This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")

    logger.info(f'Working directory: {os.getcwd()}')
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
            logger.info(
                f'The PSF files of `{lsbg["prefix"]}` in `{channels}` are not complete! Please check!')

            if default_exist_flag:
                logger.info(f'We use the default HSC PSFs instead.')
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
                f'Vanilla Task succeeded for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        gc.collect()

        return blend

    except Exception as e:
        logger.error(traceback.print_exc())
        print(traceback.print_exc())
        if bright:
            logger.error(
                f'Vanilla Task failed for BRIGHT galaxy `{lsbg["prefix"]}`')
        else:
            logger.error(f'Vanilla Task failed for `{lsbg["prefix"]}`')

        logger.info('\n')
        if fail_logger is not None:
            fail_logger.error(
                f'Vanilla Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        if global_logger is not None:
            global_logger.error(
                f'Vanilla Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        return


def _fitting_wavelet(data, coord, pixel_scale=HSC_pixel_scale, starlet_thresh=0.5, bright=False,
                     prefix='mockgal', index=0, model_dir='./Model', figure_dir='./Figure',
                     show_figure=True, tigress=False, logger=None):
    '''
    This is a Python inner function for fitting galaxy using Starlet (wavelet) model, and apply a mask after fitting.

    Parameters:
        data (`kuaizi.mock.Data`): a Python class which contains information of a galaxy.
        coord (`astropy.coordinate.SkyCoord`): coordiante (in RA, Dec) of the galaxy.
        pixel_scale (float): pixel scale of the input image, in arcsec / pixel. Default is for HSC.
        starlet_thresh (float): this number controls how many high-frequency components are remained in the model.
            Larger value gives smoother modeling. Typically we use 0.5 to 1.
        bright (bool): whether treat this galaxy as a VERY BRIGHT GALAXY. This will omit compact sources. 
        prefix (str): the prefix for output files.
        index (int): the unique index of the galaxy, such as 214.
        model_dir (str): directory for output modeling files.
        figure_dir (str): directory for output figures. 
        show_figure (bool): if True, show the figure displaying fitting results.
        tigress (bool): if True, use the GAIA catalog on Tigress. Otherwise query GAIA catalog on internet.
            When running this function on Tiger computing nodes, internet connection is off, so you have to set `tigress=True`.
        logger (`logging.Logger`): a dedicated robot who writes down the log. 
            If not provided, a Logger will be generated automatically.

    Returns:
        blend

    '''
    __name__ = "_fitting_wavelet"

    if logger is None:
        from .utils import set_logger
        logger = set_logger(__name__, f'{prefix}-{index}.log')

    lsbg_coord = coord

    # 2 whitespaces before "-", i.e., 4 whitespaces before word
    print('  - Detect sources and make mask')
    logger.info('  - Detect sources and make mask')
    print('    Query GAIA stars...')
    logger.info('    Query GAIA stars...')
    gaia_cat, msk_star_ori = kz.utils.gaia_star_mask(  # Generate a mask for GAIA bright stars
        data.images.mean(axis=0),  # averaged image
        data.wcs,
        pixel_scale=pixel_scale,
        gaia_bright=19.5,
        mask_a=694.7,
        mask_b=3.8,
        factor_b=1.0,  # 0.7,
        factor_f=1.4,  # 1.0,
        tigress=tigress,
        logger=logger)

    # Set the weights of saturated star centers to zero
    # In order to make the box size estimation more accurate.
    temp = np.copy(data.masks)
    for i in range(len(data.channels)):
        temp[i][~msk_star_ori.astype(bool)] = 0
        data.weights[i][temp[i].astype(bool)] = 0.0

    # This vanilla detection with very low sigma finds out where is the central object and its footprint
    # if cutout is lareger than 200 arcsec => large galaxy, less aggressive deblend
    # if max(data.images.shape) * pixel_scale > 200:
    #     first_dblend_cont = 0.07
    # else:
    #     first_dblend_cont = 0.01

    # obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
    #     [data],
    #     lvl=2,  # a.k.a., "sigma"
    #     mask=msk_star_ori,
    #     method='vanilla',
    #     convolve=False,
    #     match_gaia=False,
    #     show_fig=show_figure,
    #     visual_gaia=False,
    #     b=80,  # 128
    #     f=3,
    #     pixel_scale=pixel_scale,
    #     minarea=20,
    #     deblend_nthresh=48,
    #     deblend_cont=first_dblend_cont,  # 0.01, 0.05, 0.07, I changed it to 0.1
    #     sky_subtract=True,
    #     logger=logger)

    # Replace the vanilla detection with a convolved vanilla detection
    if max(data.images.shape) * pixel_scale > 200:
        first_dblend_cont = 0.07
    else:
        first_dblend_cont = 0.02
    obj_cat_ori, segmap_ori, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=4,  # a.k.a., "sigma"
        mask=msk_star_ori,
        method='vanilla',
        convolve=True,
        conv_radius=2,
        match_gaia=False,
        show_fig=show_figure,
        visual_gaia=False,
        b=80,
        f=3,
        pixel_scale=pixel_scale,
        minarea=20,
        deblend_nthresh=48,
        deblend_cont=first_dblend_cont,  # 0.01, 0.05, 0.07, I changed it to 0.1
        sky_subtract=True,
        logger=logger)

    catalog_c = SkyCoord(obj_cat_ori['ra'], obj_cat_ori['dec'], unit='deg')
    dist = lsbg_coord.separation(catalog_c)
    # ori = original, i.e., first SEP run
    cen_indx_ori = obj_cat_ori[np.argsort(dist)[0]]['index']
    cen_obj = obj_cat_ori[cen_indx_ori]

    # Better position for cen_obj, THIS IS PROBLEMATIC!!!
    # x, y, _ = sep.winpos(data.images.mean(
    #     axis=0), cen_obj['x'], cen_obj['y'], 6)
    # ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    # cen_obj['x'] = x
    # cen_obj['y'] = y
    # cen_obj['ra'] = ra
    # cen_obj['dec'] = dec
    cen_obj_coord = SkyCoord(cen_obj['ra'], cen_obj['dec'], unit='deg')

    # We roughly guess the box size of the Starlet model
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(data.channels))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=data.wcs,
        psf=model_psf,
        channels=list(data.channels))
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psf=data.psfs,
        weights=data.weights,
        channels=list(data.channels))
    observation = observation.match(model_frame)

    ############################################
    ########### First box estimation ###########
    ############################################

    cen_obj = obj_cat_ori[cen_indx_ori]
    starlet_source = StarletSource(model_frame,
                                   (cen_obj['ra'], cen_obj['dec']),
                                   observation,
                                   thresh=0.01,
                                   min_grad=-0.05,  # the initial guess of box size is as large as possible
                                   starlet_thresh=5e-3)
    # If the initial guess of the box is way too large (but not bright galaxy), set min_grad = 0.1.
    # The box is way too large
    if starlet_source.bbox.shape[1] > 0.9 * data.images[0].shape[0] and (bright):
        # The box is way too large
        min_grad = 0.03
        smaller_box = True
    elif starlet_source.bbox.shape[1] > 0.9 * data.images[0].shape[0] and (~bright):
        # not bright but large box: something must be wrong! min_grad should be larger
        min_grad = 0.05
        smaller_box = True
    elif starlet_source.bbox.shape[1] > 0.6 * data.images[0].shape[0] and (bright):
        # If box is large and gal is bright
        min_grad = 0.02
        smaller_box = True
    elif starlet_source.bbox.shape[1] > 0.6 * data.images[0].shape[0] and (~bright):
        # If box is large and gal is not bright
        min_grad = 0.01
        smaller_box = True
    else:
        smaller_box = False

    if smaller_box:
        starlet_source = scarlet.StarletSource(model_frame,
                                               (cen_obj['ra'], cen_obj['dec']),
                                               observation,
                                               thresh=0.01,
                                               min_grad=min_grad,  # the initial guess of box size is as large as possible
                                               starlet_thresh=5e-3)

    starlet_extent = kz.display.get_extent(
        starlet_source.bbox)  # [x1, x2, y1, y2]

    # extra padding, to enlarge the box
    starlet_extent[0] -= 5
    starlet_extent[2] -= 5
    starlet_extent[1] += 5
    starlet_extent[3] += 5

    if show_figure:
        # Show the Starlet initial box
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = display_single(data.images.mean(axis=0), ax=ax)
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

    if gaia_cat is not None:
        # Find stars within the wavelet box, and mask them.
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
            factor_b=0.8,
            factor_f=0.6,
            tigress=tigress,
            logger=logger)
    else:
        star_cat = []
        msk_star = np.copy(msk_star_ori)

    ############################################
    ####### Source detection and masking #######
    ############################################

    # This step masks out high frequency sources by doing wavelet transformation
    obj_cat, segmap_highfreq, bg_rms = kz.detection.makeCatalog([data],
                                                                mask=msk_star,
                                                                lvl=2.,  # 2.5
                                                                method='wavelet',
                                                                high_freq_lvl=2,  # 3
                                                                wavelet_lvl=4,
                                                                match_gaia=False,
                                                                show_fig=show_figure,
                                                                visual_gaia=False,
                                                                b=24,
                                                                f=3,
                                                                pixel_scale=pixel_scale,
                                                                minarea=3,
                                                                deblend_nthresh=30,
                                                                deblend_cont=0.03,
                                                                sky_subtract=True,
                                                                logger=logger)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = cen_obj_coord.separation(catalog_c)
    cen_indx_highfreq = obj_cat[np.argsort(dist)[0]]['index']

    # Don't mask out objects that fall in the segmap of the central object and the Starlet box
    segmap = segmap_highfreq.copy()
    # overlap_flag is for objects which fall in the footprint
    # of central galaxy in the fist SEP detection
    overlap_flag = [(segmap_ori == (cen_indx_ori + 1))[item]
                    for item in list(zip(obj_cat['y'].astype(int), obj_cat['x'].astype(int)))]
    overlap_flag = np.array(overlap_flag)

    # box_flat is for objects which fall in the initial Starlet box
    box_flag = np.unique(
        segmap[starlet_extent[2]:starlet_extent[3], starlet_extent[0]:starlet_extent[1]]) - 1
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
    # This `seg_mask` only masks compact sources
    seg_mask = (mask_conv >= gaussian_threshold)

    # This step masks out bright and large contamination, which is not well-masked in previous step
    obj_cat, segmap_big, bg_rms = kz.detection.makeCatalog(
        [data],
        lvl=4.5,  # relative agressive threshold
        method='vanilla',
        match_gaia=False,
        show_fig=show_figure,
        visual_gaia=False,
        b=32,
        f=3,
        pixel_scale=pixel_scale,
        minarea=20,   # only want large things
        deblend_nthresh=30,
        deblend_cont=0.01,
        sky_subtract=True,
        logger=logger)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = cen_obj_coord.separation(catalog_c)
    cen_indx_big = obj_cat_ori[np.argsort(dist)[0]]['index']

    # mask out big objects that are NOT identified in the high_freq step
    segmap = segmap_big.copy()
    segbox = segmap[starlet_extent[2]:starlet_extent[3],
                    starlet_extent[0]:starlet_extent[1]]
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
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    # Set weights of masked pixels to zero
    for layer in data.weights:
        layer[msk_star.astype(bool)] = 0
        layer[seg_mask.astype(bool)] = 0
        layer[seg_mask_large.astype(bool)] = 0

    # Remove compact objects that are too close to the central
    # We don't want to shred the central galaxy
    catalog_c = SkyCoord(obj_cat_cpct['ra'], obj_cat_cpct['dec'], unit='deg')
    dist = cen_obj_coord.separation(catalog_c)
    obj_cat_cpct.remove_rows(np.where(dist < 3 * u.arcsec)[0])

    # Remove objects in `obj_cat_cpct` that are already masked!
    # (since our final mask is combined from three masks)
    inside_flag = [
        seg_mask_large[item] for item in list(
            zip(obj_cat_cpct['y'].astype(int), obj_cat_cpct['x'].astype(int)))
    ]
    obj_cat_cpct.remove_rows(np.where(inside_flag)[0])

    # Remove big objects that are toooo near to the target
    catalog_c = SkyCoord(obj_cat_big['ra'], obj_cat_big['dec'], unit='deg')
    dist = cen_obj_coord.separation(catalog_c)
    #obj_cat_big.remove_rows(np.where(dist < 3 * u.arcsec)[0])
    obj_cat_big.remove_rows(np.where(
        dist < 2 * np.sqrt(cen_obj['a'] * cen_obj['b']) * pixel_scale * u.arcsec)[0])  # 2 times circularized effective radius

    # Remove objects in `obj_cat_big` that are already masked!
    inside_flag = [
        (data.weights[0] == 0)[item] for item in list(
            zip(obj_cat_big['y'].astype(int), obj_cat_big['x'].astype(int)))
    ]
    obj_cat_big.remove_rows(np.where(inside_flag)[0])

    ############################################
    ####### Add sources and render scene #######
    ############################################

    # Construct `scarlet` frames and observation
    model_psf = scarlet.GaussianPSF(sigma=(0.8,) * len(data.channels))
    model_frame = scarlet.Frame(
        data.images.shape,
        wcs=data.wcs,
        psf=model_psf,
        channels=list(data.channels))
    observation = scarlet.Observation(
        data.images,
        wcs=data.wcs,
        psf=data.psfs,
        weights=data.weights,
        channels=list(data.channels))
    observation = observation.match(model_frame)

    # STARLET_MASK!!! contains the mask for irrelavant objects (also very nearby objects),
    # as well as larger bright star mask
    # This is used to help getting the SED initialization correct.
    # When estimating the starlet box and SED, we need to mask out saturated pixels and nearby bright stars.
    starlet_mask = ((np.sum(observation.weights == 0, axis=0)
                     != 0) + msk_star_ori + (~((segmap_ori == 0) | (segmap_ori == cen_indx_ori + 1)))).astype(bool)

    sources = []

    # Add central Starlet source
    src = obj_cat_ori[cen_indx_ori]
    # Find a better box, not too large, not too small
    if smaller_box:
        min_grad_range = np.arange(min_grad, 0.3, 0.05)
    else:
        min_grad_range = np.arange(-0.2, 0.3, 0.05)  # I changed -0.3 to -0.2

    # We calculate the ratio of contaminants' area over the box area
    # Then the box size is decided based on this ratio.
    contam_ratio_list = []
    for k, min_grad in enumerate(min_grad_range):
        if k == len(min_grad_range) - 1:
            # if min_grad reaches its maximum, and `contam_ratio` is still very large,
            # we choose the min_grad with the minimum `contam_ratio`
            min_grad = min_grad_range[np.argmin(contam_ratio_list)]

        starlet_source = StarletSource(
            model_frame,
            (src['ra'], src['dec']),
            observation,
            star_mask=starlet_mask,  # bright stars are masked when estimating morphology
            satu_mask=data.masks,  # saturated pixels are masked when estimating SED
            thresh=0.05,  # 0.01
            min_grad=min_grad,
            starlet_thresh=starlet_thresh)
        starlet_extent = kz.display.get_extent(starlet_source.bbox)
        segbox = segmap_ori[starlet_extent[2]:starlet_extent[3],
                            starlet_extent[0]:starlet_extent[1]]
        contam_ratio = 1 - \
            np.sum((segbox == 0) | (segbox == cen_indx_ori + 1)) / \
            np.sum(np.ones_like(segbox))
        if (contam_ratio <= 0.08 and (~smaller_box)) or (contam_ratio <= 0.10 and (smaller_box or bright)):
            break
        else:
            contam_ratio_list.append(contam_ratio)

    logger.info('  - Wavelet modeling with the following hyperparameters:')
    print(f'  - Wavelet modeling with the following hyperparameters:')
    logger.info(
        f'    min_grad = {min_grad:.2f}, starlet_thresh = {starlet_thresh:.2f} (contam_ratio = {contam_ratio:.2f}).')
    print(
        f'    min_grad = {min_grad:.2f}, starlet_thresh = {starlet_thresh:.2f} (contam_ratio = {contam_ratio:.2f}).'
    )

    starlet_source.center = (
        np.array(starlet_source.bbox.shape) // 2 + starlet_source.bbox.origin)[1:]
    sources.append(starlet_source)
    # Finish adding the starlet source

    # Only model "real compact" sources
    if len(obj_cat_big) > 0:
        # remove intersection between cpct and big objects
        # if an object is both cpct and big, we think it is big
        cpct_coor = SkyCoord(
            ra=np.array(obj_cat_cpct['ra']) * u.degree,
            dec=np.array(obj_cat_cpct['dec']) * u.degree)
        big = SkyCoord(ra=obj_cat_big['ra'] * u.degree,
                       dec=obj_cat_big['dec'] * u.degree)
        tempid, sep2d, _ = match_coordinates_sky(big, cpct_coor)
        cpct = obj_cat_cpct[np.setdiff1d(
            np.arange(len(obj_cat_cpct)), tempid[np.where(sep2d < 1 * u.arcsec)])]
    else:
        cpct = obj_cat_cpct

    if len(star_cat) > 0 and len(cpct) > 0:
        # remove intersection between cpct and stars
        # if an object is both cpct and star, we think it is star
        star = SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg')
        cpct_coor = SkyCoord(
            ra=np.array(cpct['ra']) * u.degree,
            dec=np.array(cpct['dec']) * u.degree)
        tempid, sep2d, _ = match_coordinates_sky(star, cpct_coor)
        cpct = cpct[np.setdiff1d(np.arange(len(cpct)),
                                 tempid[np.where(sep2d < 1 * u.arcsec)])]

    if not bright:
        # for bright galaxy, we don't include these compact sources into modeling,
        # due to the limited computation resources
        for k, src in enumerate(cpct):
            if src['fwhm_custom'] < 3:
                new_source = scarlet.source.PointSource(
                    model_frame, (src['ra'], src['dec']), observation)
            elif src['fwhm_custom'] >= 3 and src['fwhm_custom'] < 5:
                new_source = scarlet.source.CompactExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation)
            else:
                new_source = scarlet.source.SingleExtendedSource(
                    model_frame, (src['ra'], src['dec']), observation, thresh=2, min_grad=0.2)
            sources.append(new_source)

    # IF GAIA stars are within the box: exclude it from the big_cat
    if len(obj_cat_big) > 0:
        if len(star_cat) > 0:
            star = SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg')
            tempid, sep2d, _ = match_coordinates_sky(big, star)
            # tempid, sep2d, _ = match_coordinates_sky(star, big)
            big_cat = obj_cat_big[np.setdiff1d(
                np.arange(len(obj_cat_big)), np.where(sep2d < 1.5 * u.arcsec)[0])]
            # big_cat = obj_cat_big[np.setdiff1d(
            #     np.arange(len(obj_cat_big)), tempid[np.where(sep2d < 1 * u.arcsec)])]
        else:
            big_cat = obj_cat_big

        for k, src in enumerate(big_cat):
            if src['fwhm_custom'] > 22:
                new_source = scarlet.source.ExtendedSource(
                    model_frame, (src['ra'], src['dec']),
                    observation,
                    K=2, thresh=2, shifting=True, min_grad=0.2)
            else:
                try:
                    new_source = scarlet.source.SingleExtendedSource(
                        model_frame, (src['ra'], src['dec']),
                        observation, satu_mask=data.masks,  # helps to get SED correct
                        thresh=2, shifting=False, min_grad=0.2)
                except Exception as e:
                    logger.info(f'   ! Error: {e}')
            sources.append(new_source)

    if len(star_cat) > 0:
        for k, src in enumerate(star_cat):
            try:
                if src['phot_g_mean_mag'] < 18:
                    new_source = scarlet.source.ExtendedSource(
                        model_frame, (src['ra'], src['dec']),
                        observation,
                        K=2, thresh=4, shifting=True, min_grad=0.4)
                else:
                    new_source = scarlet.source.SingleExtendedSource(
                        model_frame, (src['ra'], src['dec']),
                        observation, satu_mask=data.masks,
                        thresh=2, shifting=False, min_grad=0.)
            except Exception as e:
                logger.info(f'   ! Error: {e}')
            # only use SingleExtendedSource
            sources.append(new_source)

    print(f'    Total number of sources: {len(sources)}')
    logger.info(f'    Total number of sources: {len(sources)}')

    # Visualize our data and mask and source
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # fig = kz.display.display_scarlet_sources(
    #     data,
    #     sources,
    #     show_ind=None,
    #     stretch=1,
    #     Q=1,
    #     minimum=-0.3,
    #     show_mark=True,
    #     scale_bar_length=10,
    #     add_text=f'{prefix}-{index}')
    # plt.savefig(
    #     os.path.join(figure_dir, f'{prefix}-{index}-src-wavelet.png'), dpi=70, bbox_inches='tight')
    # if not show_figure:
    #     plt.close()

    # Star fitting!
    start = time.time()
    blend = scarlet.Blend(sources, observation)
    fig = kz.display.display_scarlet_model(
        blend,
        minimum=-0.3,
        stretch=1,
        channels=data.channels,
        show_loss=False,
        show_mask=False,
        show_mark=True,
        scale_bar=False)
    plt.savefig(
        os.path.join(figure_dir, f'{prefix}-{index}-init-wavelet.png'), dpi=70, bbox_inches='tight')
    if not show_figure:
        plt.close()

    ############################################
    ################# Fitting ##################
    ############################################
    try:
        if bright:
            e_rel_list = [5e-4, 1e-5]  # otherwise it will take forever....
            n_iter = 100
        else:
            e_rel_list = [5e-4, 1e-5, 5e-5, 1e-6]
            n_iter = 150

        blend.fit(n_iter, 1e-4)

        with open(os.path.join(model_dir, f'{prefix}-{index}-trained-model-wavelet.df'), 'wb') as fp:
            dill.dump([blend, {'starlet_thresh': starlet_thresh,
                               'e_rel': 1e-4, 'loss': blend.loss[-1]}, None], fp)
            fp.close()
        last_loss = blend.loss[-1]
        logger.info(
            f'    Optimizaiton: Succeed for e_rel = 1e-04 with {len(blend.loss)} iterations! Try higher accuracy!')
        print(
            f'    Optimizaiton: Succeed for e_rel = 1e-04 with {len(blend.loss)} iterations! Try higher accuracy!')

        for i, e_rel in enumerate(e_rel_list):
            blend.fit(150, e_rel)
            if len(blend.loss) > 50:  # must have more than 50 iterations
                recent_loss = np.mean(blend.loss[-10:])
                min_loss = np.min(blend.loss[:-10])
                if recent_loss < min_loss:
                    logger.info(
                        f'    Optimizaiton: Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    print(
                        f'    Optimizaiton: Succeed for e_rel = {e_rel} with {len(blend.loss)} iterations! Try higher accuracy!')
                    with open(os.path.join(model_dir, f'{prefix}-{index}-trained-model-wavelet.df'), 'wb') as fp:
                        dill.dump(
                            [blend, {'starlet_thresh': starlet_thresh, 'e_rel': e_rel, 'loss': blend.loss[-1]}, None], fp)
                        fp.close()
                elif abs((recent_loss - min_loss) / min_loss) < 0.02:
                    if recent_loss < last_loss:  # better than the saved model
                        logger.info(
                            f'    Optimizaiton: I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        print(
                            f'    Optimizaiton: I am okay with relative loss difference = {abs((recent_loss - min_loss) / min_loss)}. Fitting stopped.')
                        with open(os.path.join(model_dir, f'{prefix}-{index}-trained-model-wavelet.df'), 'wb') as fp:
                            dill.dump(
                                [blend, {'starlet_thresh': starlet_thresh, 'e_rel': e_rel, 'loss': blend.loss[-1]}, None], fp)
                            fp.close()
                        break
                else:
                    logger.info(
                        f'  ! Optimizaiton: Cannot achieve a global optimization with e_rel = {e_rel}.')
                    print(
                        f'  ! Optimizaiton: Cannot achieve a global optimization with e_rel = {e_rel}.')
            else:
                continue
        if len(blend.loss) < 50:
            logger.warning(
                '  ! Might be poor fitting! Iterations less than 50.')
            print('  ! Might be poor fitting! Iterations less than 50.')
        logger.info("  - After {1} iterations, logL = {2:.2f}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        print("  - After {1} iterations, logL = {2:.2f}".format(
            e_rel, len(blend.loss), -blend.loss[-1]))
        end = time.time()
        logger.info(f'    Elapsed time for fitting: {(end - start):.2f} s')
        print(f'    Elapsed time for fitting: {(end - start):.2f} s')

        # In principle, Now we don't need to find which components compose a galaxy. The central Starlet is enough!
        if len(blend.sources) > 1:
            mag_mat = np.array(
                [-2.5 * np.log10(kz.measure.flux(src, observation)) + 27 for src in sources])
            # g - r, g - i, g - z
            color_mat = (- mag_mat + mag_mat[:, 0][:, np.newaxis])[:, 1:]
            color_dist = np.linalg.norm(
                color_mat - color_mat[0], axis=1) / np.linalg.norm(color_mat[0])
            sed_ind = np.where(color_dist < 0.1)[0]
            dist = np.array([
                np.linalg.norm(
                    src.center - blend.sources[0].center) * HSC_pixel_scale
                for src in np.array(blend.sources)[sed_ind]
            ])
            dist_flag = (
                dist < 3 * np.sqrt(cen_obj['a'] * cen_obj['b']) * HSC_pixel_scale)

            # maybe use segmap flag? i.e., include objects that are overlaped
            # with the target galaxy in the inital detection.

            point_flag = np.array([
                isinstance(src, scarlet.source.PointSource)
                for src in np.array(blend.sources)[sed_ind]
            ])  # we don't want point source

            near_cen_flag = [
                (segmap_ori == cen_indx_ori +
                 1)[int(src.center[0]), int(src.center[1])]  # src.center: [y, x]
                for src in np.array(blend.sources)[sed_ind]
            ]

            sed_ind = sed_ind[(~point_flag) & near_cen_flag & dist_flag]

            if not 0 in sed_ind:
                # the central source must be included.
                sed_ind = np.array(list(set(sed_ind).union({0})))
        else:
            sed_ind = np.array([0])
        logger.info(
            f'  - Components {sed_ind} are considered as the target galaxy.')
        print(
            f'  - Components {sed_ind} are considered as the target galaxy.')

        ############################################
        ################# Final mask ##################
        ############################################
        # Only mask bright stars!!!
        logger.info(
            '  - Masking stars and other sources that are modeled, to deal with leaky flux issue.')
        print('  - Masking stars and other sources that are modeled, to deal with leaky flux issue.')
        # Generate a VERY AGGRESSIVE mask, named "footprint"
        footprint = np.zeros_like(segmap_highfreq, dtype=bool)
        # for ind in cpct['index']:  # mask ExtendedSources which are modeled
        #     footprint[segmap_highfreq == ind + 1] = 1

        # footprint[segmap_highfreq == cen_indx_highfreq + 1] = 0
        sed_ind_pix = np.array([item.center for item in np.array(
            sources)[sed_ind]]).astype(int)  # the y and x of sed_ind objects
        # # if any objects in `sed_ind` is in `segmap_highfreq`
        # sed_corr_indx = segmap_highfreq[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
        # for ind in sed_corr_indx:
        #     footprint[segmap_highfreq == ind] = 0

        # smooth_radius = 1.5
        # gaussian_threshold = 0.03
        # mask_conv = np.copy(footprint)
        # mask_conv[mask_conv > 0] = 1
        # mask_conv = convolve(mask_conv.astype(
        #     float), Gaussian2DKernel(smooth_radius))
        # footprint = (mask_conv >= gaussian_threshold)

        # Mask star within the box
        if len(star_cat) > 0:
            _, star_mask = kz.utils.gaia_star_mask(  # Generate GAIA mask only for stars outside of the Starlet box
                data.images.mean(axis=0),
                data.wcs,
                gaia_stars=star_cat,
                pixel_scale=pixel_scale,
                gaia_bright=19,
                mask_a=694.7,
                mask_b=3.8,
                factor_b=0.9,
                factor_f=1.1,
                tigress=tigress)
            footprint = footprint | star_mask

        # Mask big objects from `big_cat`
        if len(obj_cat_big) > 0:
            # Blow-up radius depends on the distance to target galaxy
            catalog_c = SkyCoord(big_cat['ra'], big_cat['dec'], unit='deg')
            dist = cen_obj_coord.separation(catalog_c)
            near_flag = (dist < 4 * cen_obj['a'] * HSC_pixel_scale * u.arcsec)

            footprint2 = np.zeros_like(segmap_big, dtype=bool)
            # mask ExtendedSources which are modeled
            for ind in big_cat[near_flag]['index']:
                footprint2[segmap_big == ind + 1] = 1

            # if any objects in `sed_ind` is in `segmap_big`
            sed_corr_indx = segmap_big[sed_ind_pix[:, 0], sed_ind_pix[:, 1]]
            for ind in sed_corr_indx:
                footprint2[segmap_big == ind] = 0
            footprint2[segmap_big == cen_indx_big + 1] = 0

            smooth_radius = 1.5
            gaussian_threshold = 0.1
            mask_conv = np.copy(footprint2)
            mask_conv[mask_conv > 0] = 1
            mask_conv = convolve(mask_conv.astype(
                float), Gaussian2DKernel(smooth_radius))
            footprint2 = (mask_conv >= gaussian_threshold)

            footprint3 = np.zeros_like(segmap_big, dtype=bool)
            # mask ExtendedSources which are modeled
            for ind in big_cat[~near_flag]['index']:
                footprint3[segmap_big == ind + 1] = 1
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

        outdir = os.path.join(
            model_dir, f'{prefix}-{index}-trained-model-wavelet.df')
        logger.info(f'  - Saving the results as {os.path.abspath(outdir)}')
        print(f'  - Saving the results as {os.path.abspath(outdir)}')
        with open(os.path.abspath(outdir), 'wb') as fp:
            dill.dump(
                [blend, {'starlet_thresh': starlet_thresh, 'e_rel': e_rel, 'loss': blend.loss[-1], 'sed_ind': sed_ind}, footprint], fp)
            fp.close()

        # Save fitting figure
        # zoomin_size: in arcsec, rounded to integer multiple of 30 arcsec
        zoomin_size = np.ceil(
            (sources[0].bbox.shape[1] * pixel_scale * 3) / 30) * 30
        # cannot exceed the image size
        zoomin_size = min(zoomin_size, data.images.shape[1] * pixel_scale)

        fig = kz.display.display_scarlet_results_tigress(
            blend,
            footprint,
            show_ind=sed_ind,
            zoomin_size=zoomin_size,
            minimum=-0.3,
            stretch=1,
            Q=1,
            channels=data.channels,
            show_loss=True,
            show_mask=False,
            show_mark=False,
            scale_bar=True)
        plt.savefig(
            os.path.join(figure_dir, f'{prefix}-{index}-zoomin-wavelet.png'), dpi=55, bbox_inches='tight')
        if not show_figure:
            plt.close()

        logger.info('Done! (♡˙︶˙♡)')
        logger.info('\n')
        # fig = kz.display.display_scarlet_model(
        #     blend,
        #     minimum=-0.3,
        #     stretch=1,
        #     channels=data.channels,
        #     show_loss=True,
        #     show_mask=False,
        #     show_mark=False,
        #     scale_bar=False)
        # plt.savefig(
        #     os.path.join(figure_dir, f'{prefix}-{index}-fitting-wavelet.png'), bbox_inches='tight')
        # if not show_figure:
        #     plt.close()

        # # Save zoomin figure (non-agressively-masked, target galaxy only)
        # fig = kz.display.display_scarlet_model(
        #     blend,
        #     show_ind=sed_ind,
        #     zoomin_size=50,
        #     minimum=-0.3,
        #     stretch=1,
        #     channels=data.channels,
        #     show_loss=True,
        #     show_mask=False,
        #     show_mark=False,
        #     scale_bar=False)
        # plt.savefig(
        #     os.path.join(figure_dir, f'{prefix}-{index}-zoomin-wavelet.png'), bbox_inches='tight')
        # if not show_figure:
        #     plt.close()

        # # Save zoomin figure (aggressively-masked, target galaxy only)
        # new_weights = data.weights.copy()
        # for layer in new_weights:
        #     layer[footprint.astype(bool)] = 0
        # observation2 = scarlet.Observation(
        #     data.images,
        #     wcs=data.wcs,
        #     psf=data.psfs,
        #     weights=new_weights,
        #     channels=list(data.channels))
        # observation2 = observation2.match(model_frame)
        # blend2 = scarlet.Blend(sources, observation2)

        # fig = kz.display.display_scarlet_model(
        #     blend2,
        #     show_ind=sed_ind,
        #     zoomin_size=50,
        #     minimum=-0.3,
        #     stretch=1,
        #     channels=data.channels,
        #     show_loss=False,
        #     show_mask=True,
        #     show_mark=False,
        #     scale_bar=False)
        # plt.savefig(
        #     os.path.join(figure_dir, f'{prefix}-{index}-zoomin-mask-wavelet.png'), bbox_inches='tight')
        # if not show_figure:
        #     plt.close()

        # # Save high-freq-removed figure
        # ## remove high-frequency features from the Starlet objects
        # for src in np.array(blend2.sources)[sed_ind]:
        #     if isinstance(src, StarletSource):
        #         # Cutout a patch of original image
        #         y_cen, x_cen = np.array(src.bbox.shape)[1:] // 2 + np.array(src.bbox.origin)[1:]
        #         size = np.array(src.bbox.shape)[1:] // 2
        #         img_ = observation.data[:, y_cen - size[0]:y_cen + size[0] + 1, x_cen - size[1]:x_cen + size[1] + 1]

        #         morph = src.children[1]
        #         stlt = Starlet(morph.get_model(), direct=True)
        #         c = stlt.coefficients
        #         c[:, :2, :, :] = 0 # Remove high-frequency features
        #         new_morph = copy.deepcopy(morph)
        #         new_src = copy.deepcopy(src)
        #         new_morph.__init__(morph.frame, img_, coeffs=c, bbox=morph.bbox)
        #         src.children[1] = new_morph

        # fig = kz.display.display_scarlet_model(
        #     blend2,
        #     show_ind=sed_ind,
        #     zoomin_size=50,
        #     minimum=-0.3,
        #     stretch=1,
        #     channels='griz',
        #     show_loss=False,
        #     show_mask=True,
        #     show_mark=False,
        #     scale_bar=False)
        # plt.savefig(
        #     os.path.join(figure_dir, f'{prefix}-{index:04d}-zoomin-blur-wavelet.png'), bbox_inches='tight')
        # if not show_figure:
        #     plt.close()

        return blend
    except Exception as e:
        raise ScarletFittingError(
            prefix, index, starlet_thresh, data.channels, e)


def fitting_wavelet_observation(lsbg, hsc_dr, cutout_halfsize=1.0, starlet_thresh=0.5, prefix='LSBG', pixel_scale=HSC_pixel_scale,
                                zp=HSC_zeropoint, model_dir='./Models', figure_dir='./Figure', show_figure=False):

    from kuaizi.utils import padding_PSF
    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    # kz.utils.set_matplotlib(usetex=False, fontsize=15)

    index = lsbg['Seq']
    lsbg_coord = SkyCoord(ra=lsbg['RAJ2000'], dec=lsbg['DEJ2000'], unit='deg')

    img_dir = './Images/'
    psf_dir = './PSFs/'

    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(psf_dir):
        os.makedirs(psf_dir)

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
        prefix=f'LSBG_{index:04d}_img')  # {prefix}

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
    weights = 1.0 / np.array([hdu[3].data for hdu in cutout])
    weights[np.isinf(weights)] = 0.0
    psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
    psfs = scarlet.ImagePSF(np.array(psf_pad))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    blend = _fitting_wavelet(
        data, lsbg_coord, starlet_thresh=starlet_thresh, prefix=prefix, index=index, pixel_scale=pixel_scale,
        model_dir=model_dir, figure_dir=figure_dir, show_figure=show_figure)
    return blend


def fitting_wavelet_obs_tigress(env_dict, lsbg, name='Seq', channels='grizy',
                                starlet_thresh=0.5, pixel_scale=HSC_pixel_scale, bright_thresh=17.0,
                                prefix='candy', model_dir='./Model', figure_dir='./Figure', show_figure=False,
                                logger=None, global_logger=None, fail_logger=None):
    '''
    Run scarlet wavelet modeling on Tiger, modified on 09/13/2021. 

    Parameters:
        env_dict (dict): dictionary indicating the file directories, such as 
            `env_dict = {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/tigress/jiaxuanl/Data'}`
        lsbg (one row in `astropy.Table`): the galaxy to be modeled. 
        name (str): the column name for the index of `lsbg`.
        channels (str): bandpasses to be used, such as 'grizy'.
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
    __name__ = "fitting_wavelet_obs_tigress"

    from kuaizi.utils import padding_PSF
    from kuaizi.mock import Data
    import unagi  # for HSC saturation mask

    index = lsbg[name]
    # whether this galaxy is a very bright one
    bright = (lsbg['mag_auto_i'] < bright_thresh)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if logger is None:
        from .utils import set_logger
        logger = set_logger(__name__, os.path.join(
            model_dir, f'{prefix}-{index}.log'), level='info')

    logger.info(f'Running scarlet wavelet modeling for `{lsbg["prefix"]}`')
    print(f'### Running scarlet wavelet modeling for `{lsbg["prefix"]}`')

    if bright:
        logger.info(
            f"This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")
        print(
            f"    This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")

    logger.info(f'Working directory: {os.getcwd()}')
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
            logger.info(
                f'The PSF files of `{lsbg["prefix"]}` in `{channels}` are not complete! Please check!')

            if default_exist_flag:
                logger.info(f'We use the default HSC PSFs instead.')
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

        # Collect free RAM
        del cutout, psf_list
        del images, w, weights, psf_pad, psfs
        gc.collect()

        blend = _fitting_wavelet(
            data, lsbg_coord, starlet_thresh=starlet_thresh,
            prefix=prefix, bright=bright, index=index, pixel_scale=pixel_scale,
            model_dir=model_dir, figure_dir=figure_dir, show_figure=show_figure,
            tigress=True, logger=logger)

        if global_logger is not None:
            global_logger.info(
                f'Task succeeded for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        gc.collect()

        return blend

    except Exception as e:
        logger.error(e)
        print(e)
        if bright:
            logger.error(f'Task failed for BRIGHT galaxy `{lsbg["prefix"]}`')
        else:
            logger.error(f'Task failed for `{lsbg["prefix"]}`')

        logger.info('\n')
        if fail_logger is not None:
            fail_logger.error(
                f'Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        if global_logger is not None:
            global_logger.error(
                f'Task failed for `{lsbg["prefix"]}` in `{channels}` with `starlet_thresh = {starlet_thresh}`')

        return


def fitting_wavelet_mockgal(index=0, starlet_thresh=0.5, prefix='MockLSBG', pixel_scale=HSC_pixel_scale,
                            zp=HSC_zeropoint, model_dir='./Models/MockGalModel', output_dir='./Models/',
                            figure_dir='./Figure', show_figure=False):

    kz.utils.set_env(project='HSC', name='HSC_LSBG')
    index = index

    from kuaizi.mock import MockGal
    mgal = MockGal.read(os.path.join(
        model_dir, f'{prefix}-{index:04d}.pkl'))
    print('Loading', os.path.join(
        model_dir, f'{prefix}-{index:04d}.pkl'))
    channels = mgal.channels
    channels_list = list(channels)
    filters = channels_list
    lsbg_coord = SkyCoord(
        ra=mgal.model.info['ra'], dec=mgal.model.info['dec'], unit='deg')

    # Reconstructure data
    images = mgal.mock.images
    w = mgal.mock.wcs
    weights = 1 / mgal.mock.variances
    weights[np.isinf(weights)] = 0.0
    psfs = scarlet.ImagePSF(np.array(mgal.mock.psfs))
    data = Data(images=images, weights=weights,
                wcs=w, psfs=psfs, channels=channels)

    blend = _fitting_wavelet(
        data, lsbg_coord, starlet_thresh=starlet_thresh, prefix=prefix, index=index, pixel_scale=pixel_scale,
        zp=zp, model_dir=output_dir, figure_dir=figure_dir, show_figure=show_figure)
    return blend
