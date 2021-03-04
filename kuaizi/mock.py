from __future__ import division, print_function

import copy
import os
import pickle, dill
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table
from astropy.units import Quantity
from matplotlib.patches import Ellipse as mpl_ellip

from . import HSC_pixel_scale, HSC_zeropoint
from .display import ORG, SEG_CMAP, display_single

hsc_sky = {'g': 0.010, 'r': 0.014, 'i': 0.016,
           'z': 0.022, 'y': 0.046}  # muJy/arcsec^2
# https://github.com/dr-guangtou/hsc_massive/blob/master/notebooks/selection/s18a_wide_sky_objects.ipynb

# Class to provide compact input of instrument data and metadata


class Data:
    """ This is a rudimentary class to set the necessary information for a scarlet run.

    While it is possible for scarlet to run without wcs or psf,
    it is strongly recommended not to, which is why these entry are not optional.
    """

    def __init__(self, images, variances=None, masks=None, channels=None, wcs=None, weights=None, psfs=None, info=None):
        self.images = images
        self.variances = variances
        # weights and variances don't necessarily need to be the same.
        self.weights = weights
        self.masks = masks
        self.channels = channels
        self.wcs = wcs
        self.psfs = psfs
        self.info = info
        #self.pixel_scale = np.sqrt(np.abs(np.linalg.det(self.wcs.pixel_scale_matrix))) * 3600

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    @property
    def variances(self):
        return self._variances

    @variances.setter
    def variances(self, variances):
        self._variances = variances

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, masks):
        self._masks = masks

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, info):
        self._info = info


class MockGal:
    """
    This is a class for "simple" (can be expressed by GalSim) mock galaxy.

    bkg_image
    bkg_variance
    bkg_mask

    image
    model
    variance

    info

    """

    def __init__(self, bkg):
        self.bkg = bkg
        self.channels = bkg.channels

    def __del__(self):
        print('Mock Galaxy deleted.')

    @property
    def bkg(self):
        return self._bkg

    @bkg.setter
    def bkg(self, bkg):
        self._bkg = bkg
        self.channels = bkg.channels

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def set_mock(self):
        # Set the `mock` object
        mock_img = self.model.images + self.bkg.images
        mock_var = mock_variance(
            self.model.images, self.bkg.images, self.bkg.variances)
        self._mock = Data(images=mock_img, variances=mock_var,
                          masks=self.bkg.masks, channels=self.channels,
                          wcs=self.bkg.wcs, weights=None, psfs=self.bkg.psfs)

    @property
    def mock(self):
        return self._mock

    def gen_mock_lsbg(self, galaxy, zp=HSC_zeropoint, pixel_scale=HSC_pixel_scale, verbose=True):
        '''
        Generate mock low surface brightness galaxies. 
        '''
        import galsim
        from galsim import Angle, Image, InterpolatedImage, degrees
        from galsim.fitswcs import AstropyWCS
        from galsim.interpolant import Lanczos
        big_fft_params = galsim.GSParams(maximum_fft_size=20000)

        if not isinstance(galaxy['comp'], list):
            galaxy['comp'] = list(galaxy['comp'])

        if len(galaxy['comp']) == 1:
            galaxy['flux_fraction'] = [1.0]

        # print some information
        if verbose:
            print('# Generating mock galaxy.')
            print('    - Total components: ', len(galaxy['comp']))
            print('    - Types: ',
                  [c['model'].__name__ for c in galaxy['comp']])
            print('    - Flux fraction: ', galaxy['flux_fraction'])

        # Empty canvas
        field = np.empty_like(self.bkg.images[0])
        model_images = np.empty_like(self.bkg.images)

        # Calculate RA, DEC of the mock galaxy
        y_cen = self.bkg.images.shape[2] / 2
        x_cen = self.bkg.images.shape[1] / 2
        galaxy['ra'], galaxy['dec'] = self.bkg.wcs.wcs_pix2world(
            x_cen, y_cen, 0)

        # Calculate flux based on i-band mag and SED
        i_band_loc = np.argwhere(np.array(list(self.channels)) == 'i')[
            0][0]  # location of i-band in `channels`
        seds = np.array([c['sed'] for c in galaxy['comp']])
        # Normalize SED w.r.t i-band
        seds /= seds[:, i_band_loc][:, np.newaxis]
        tot_sed = np.sum(
            seds * np.array(galaxy['flux_fraction'])[:, np.newaxis], axis=0)
        for i, band in enumerate(self.channels):
            galaxy[f'{band}mag'] = -2.5 * np.log10(tot_sed[i]) + galaxy['imag']

        if verbose:
            print(f'    - Magnitude in {self.channels}: ',
                  [round(galaxy[f'{band}mag'], 1) for band in self.channels])

        #### Star generating mock galaxy in each band ####
        for i, band in enumerate(self.channels):  # griz
            # Random number seed
            # This random number seed should be fixed across bands!!!
            rng = galsim.BaseDeviate(23333)

            # Sky background level
            sky_SB = 29  # mag/arcsec^2
            sky_level = 10**((zp - sky_SB) / 2.5)  # counts / arcsec^2

            # Define the PSF
            interp_psf = InterpolatedImage(Image(self.bkg.psfs[i] / self.bkg.psfs[i].sum(), dtype=float),
                                           scale=pixel_scale,
                                           x_interpolant=Lanczos(3))

            # Total flux for all components
            tot_flux = 10**((zp - galaxy[f'{band}mag']) / 2.5)

            gal_list = []
            for k, comp in enumerate(galaxy['comp']):
                # Define the galaxy
                gal = comp['model'](**comp['model_params'],
                                    gsparams=big_fft_params)

                gal_shape = galsim.Shear(
                    **comp['shear_params'])  # Shear the galaxy
                gal = gal.shear(gal_shape)

                if 'shift' in comp.keys():  # Shift the center
                    gal = gal.shift(comp['shift'])

                # Add star forming knots
                if 'n_knots' in comp.keys() and comp['n_knots'] > 0:
                    if not 'knots_frac' in comp.keys():
                        raise KeyError(
                            '`knots_frac` must be provided to generate star forming knots!')
                    else:
                        if 'knots_sed' in comp.keys():
                            knot_frac = comp['knots_frac'] * \
                                (comp['knots_sed'] /
                                 np.sum(comp['knots_sed']))[i]
                        else:
                            knot_frac = comp['knots_frac'] * 0.25  # flat SED
                        knots = galsim.RandomKnots(comp['n_knots'],
                                                   half_light_radius=comp['model_params']['half_light_radius'],
                                                   flux=knot_frac,
                                                   rng=rng)
                    gal = galsim.Add([gal, knots])

                gal = gal.withFlux(
                    tot_flux * galaxy['flux_fraction'][k])  # Get Flux
                gal_list.append(gal)

            # Adding all components together
            gal = galsim.Add(gal_list)

            # Convolve galaxy with PSF
            final = galsim.Convolve([gal, interp_psf])

            # Draw the image with a particular pixel scale.
            gal_image = final.drawImage(
                scale=pixel_scale, nx=field.shape[1], ny=field.shape[0])

            # Add noise
            sky_sigma = hsc_sky[f'{band}'] / 3.631 * \
                10**((zp - 22.5) / 2.5) * pixel_scale**2
            noise = galsim.GaussianNoise(rng, sigma=sky_sigma)
            # gal_image.addNoise(noise)

            # Generate mock image
            model_img = gal_image.array
            model_images[i] = model_img

        # Generate variance map
        mock_model = Data(images=model_images, variances=None,
                          masks=None, channels=self.channels,
                          wcs=None, weights=None, psfs=self.bkg.psfs, info=galaxy)

        # Finished!!!
        self.model = mock_model  # model only has `images`, `channels`, `psfs`, and `info`!
        self.set_mock()  # mock has other things, including modified variances.

    def from_scarlet(self, scarlet_model_dir, pixel_scale=HSC_pixel_scale):
        from galsim import Image, InterpolatedImage
        from galsim.interpolant import Lanczos

        with open(scarlet_model_dir, 'rb') as f:
            blend, info, mask = dill.load(f)
            f.close()
        assert ''.join(blend.observations[0].channels) == self.channels

        # Crop the mask
        new_weights = blend.observations[0].weights
        x1, y1 = blend.sources[0].bbox.origin[1:]
        x2 = x1 + blend.sources[0].bbox.shape[1:][0]
        y2 = y1 + blend.sources[0].bbox.shape[1:][1]
        mask = mask.astype(bool)[x1:x2, y1:y2]
        mask += np.sum((new_weights[:, x1:x2, y1:y2] == 0), axis=0).astype(bool)
        

        mockgal_img = blend.sources[0].get_model() * np.repeat(~mask[np.newaxis, :, :], len(self.channels), axis=0)
        gal_image = np.empty_like(self.bkg.images)
        for i in range(len(mockgal_img)):
            mockgal = InterpolatedImage(
                Image(mockgal_img[i], dtype=float),
                scale=pixel_scale,
                x_interpolant=Lanczos(3))
            gal_image[i] = mockgal.drawImage(
                scale=pixel_scale, nx=self.bkg.images.shape[2], ny=self.bkg.images.shape[1]).array

        # Generate variance map
        ra, dec = self.bkg.wcs.wcs_pix2world(self.bkg.images.shape[2] / 2, self.bkg.images.shape[1] / 2, 0)
        info = {'ra': ra, 'dec': dec, 
                'scarlet_model': scarlet_model_dir,
                'model_type': 'scarlet_lsbg_wvlt_0.5'}
        mock_model = Data(images=gal_image, variances=None,
                          masks=None, channels=self.channels,
                          wcs=None, weights=None, psfs=self.bkg.psfs, 
                          info=info)

        # Finished!!!
        self.model = mock_model  # model only has `images`, `channels`, `psfs`, and `info`!
        self.set_mock()  # mock has other things, including modified variances.

    def display(self, zoomin_size=None, ax=None, stretch=1, Q=0.1, minimum=-0.2, pixel_scale=0.168, scale_bar=True,
                scale_bar_length=20.0, scale_bar_fontsize=15, scale_bar_y_offset=0.3, scale_bar_color='w',
                scale_bar_loc='left', add_text=None, usetex=False, text_fontsize=30, text_y_offset=0.80, text_color='w'):
        '''
        Display the background image, mock galaxy model, and mock image.
        Inherited from `kuaizi.display.display_scarlet_model`.

        Arguments:
            zoomin_size (float, in arcsec): the size of shown image, if not showing in full size
            ax (matplotlib.axes object): input axes object
            show_loss (bool): whether displaying the loss curve
            show_mask (bool): whether displaying the mask encoded in `data.weights'
            show_ind (list): if not None, only objects with these indices are shown in the figure
            stretch, Q, minimum (float): parameters for displaying image, see https://pmelchior.github.io/scarlet/tutorials/display.html
            channels (str): names of the bands in `observation`
            show_mark (bool): whether plot the indices of sources in the figure
            pixel_scale (float): default is 0.168 arcsec/pixel

        Returns: 
            ax: if input `ax` is provided
            fig: if no `ax` is provided as input

        '''
        import scarlet

        if ax is None:
            fig = plt.figure(figsize=(18, 6))
            ax = [fig.add_subplot(1, 3, n + 1) for n in range(3)]

        if zoomin_size is not None:
            x_cen = self.bkg.images.shape[2] // 2
            y_cen = self.bkg.images.shape[1] // 2
            size = int(zoomin_size / pixel_scale / 2)  # half-size
            # Image
            images = self.bkg.images[:, y_cen - size:y_cen +
                                     size + 1, x_cen - size:x_cen + size + 1]
            # Model
            model = self.model.images[:, y_cen - size:y_cen +
                                      size + 1, x_cen - size:x_cen + size + 1]
        else:
            # Image
            images = self.bkg.images
            # Model
            model = self.model.images

        # Create RGB images
        norm = scarlet.display.AsinhMapping(
            minimum=minimum, stretch=stretch, Q=Q)

        img_rgb = scarlet.display.img_to_rgb(images, norm=norm)
        ax[0].imshow(img_rgb)
        ax[0].set_title("Background")
        img_rgb = scarlet.display.img_to_rgb(images + model, norm=norm)
        ax[1].imshow(img_rgb)
        ax[1].set_title("Mock Image")
        img_rgb = scarlet.display.img_to_rgb(model, norm=norm)
        ax[2].imshow(img_rgb)
        ax[2].set_title("Model")

        (img_size_x, img_size_y) = images[0].shape
        if scale_bar:
            if scale_bar_loc == 'left':
                scale_bar_x_0 = int(img_size_x * 0.04)
                scale_bar_x_1 = int(img_size_x * 0.04 +
                                    (scale_bar_length / pixel_scale))
            else:
                scale_bar_x_0 = int(img_size_x * 0.95 -
                                    (scale_bar_length / pixel_scale))
                scale_bar_x_1 = int(img_size_x * 0.95)
            scale_bar_y = int(img_size_y * 0.10)
            scale_bar_text_x = (scale_bar_x_0 + scale_bar_x_1) / 2
            scale_bar_text_y = (scale_bar_y * scale_bar_y_offset)

            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
            scale_bar_text_size = scale_bar_fontsize

            ax[0].plot(
                [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
                linewidth=3,
                c=scale_bar_color,
                alpha=1.0)
            ax[0].text(
                scale_bar_text_x,
                scale_bar_text_y,
                scale_bar_text,
                fontsize=scale_bar_text_size,
                horizontalalignment='center',
                color=scale_bar_color)

        if add_text is not None:
            text_x_0 = int(img_size_x * 0.08)
            text_y_0 = int(img_size_y * text_y_offset)
            if usetex:
                ax[0].text(
                    text_x_0, text_y_0, r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)
            else:
                ax[0].text(text_x_0, text_y_0, add_text,
                           fontsize=text_fontsize, color=text_color)

        from matplotlib.ticker import MaxNLocator, NullFormatter
        for axx in ax:
            axx.yaxis.set_major_locator(MaxNLocator(5))
            axx.xaxis.set_major_locator(MaxNLocator(5))

        if ax is None:
            return fig
        return ax

    #### IO related ####

    def write(self, filename, format='pkl', overwrite=False):
        if format == 'pkl':
            if overwrite is False and os.path.isfile(filename):
                print('File already exists. No changes are made.')
            else:
                with open(filename, 'wb') as fp:
                    pickle.dump(self, fp)
                    fp.close()
        else:
            raise ValueError(
                'Other formats are not supported yet. Please use `pkl`.')

    @classmethod
    def read(cls, filename, format='pkl'):
        if format == 'pkl':
            with open(filename, "rb") as fp:
                gal = pickle.load(fp)
                fp.close()
            return gal
        else:
            raise ValueError(
                'Other formats are not supported yet. Please use `pkl`.')
            return


# Determine the coefficient when converting image flux to `sigma map` for HSC
def calc_sigma_coeff(images, variance_map):
    '''
    This function empirically calculate the coefficient used to convert `image flux` to `sigma map`.
    This only works for single band now. 

    `sigma map = coeff * sqrt(image_flux)`, where `sigma map = sqrt(variance map)`. 

    Parameters:
        images (numpy 2-D array or 3-D array for multiple bands): image array. Band is along the first axis.
        variance_map (numpy 2-D array or 3-D array for multiple bands): variance array, 
            i.e., the third layer of HSC cutout (`hdu[3].data`).

    Return:
        A_best (numpy array): coeff in each band
    '''
    from astropy.convolution import Gaussian2DKernel, convolve

    from kuaizi.utils import extract_obj

    if len(images.shape) == 2:  # single band
        images = images[np.newaxis, :, :]
        variance_map = variance_map[np.newaxis, :, :]

    sigma_map = np.sqrt(variance_map)
    # Generate a mask, which only includes bright objects
    obj_cat, segmap = extract_obj(images.mean(
        axis=0), b=32, f=2, sigma=3, show_fig=False, verbose=False)
    mask = segmap > 0
    mask = convolve(mask, Gaussian2DKernel(2)) > 0.2
    # Calculate the loss between `sigma_map` and `images`. `A` is a coefficient.
    sigma_map -= np.median(sigma_map, axis=(1, 2)
                           )[:, np.newaxis, np.newaxis]  # remove a background
    # len(images) is the number of bands
    mask = np.repeat(mask[np.newaxis, :, :], len(images), axis=0)

    A_set = np.linspace(0, 0.01, 200)
    loss = np.zeros([len(images), len(A_set)])
    for i, A in enumerate(A_set):
        temp = np.sqrt(images)
        temp[np.isnan(temp)] = 0.0
        diff = (sigma_map - A * temp)[mask].reshape(len(images), -1)
        # diff = diff[~np.isnan(diff)]
        loss[:, i] = np.linalg.norm(diff, axis=1) / np.sum(mask[0])
    A_best = A_set[np.argmin(loss, axis=1)]

    return A_best


def mock_variance(mock_img, images, variance):
    '''
    Generate mock variance map based on given mock image. 
    `Band` is always along the first axis of the array.
    E.g., `images.shape = (4, 300, 300)` corresponds to images in `g, r, i, z` bands.

    Parameters:
        mock_img (numpy array): array for the image of mock galaxy. 
        images (numpy array): array for the original image (onto which the mock galaxy is placed).
        variance (numpy array): array for the original variance map (`hdu[3].data` for HSC).

    Returns:
        mock_variance (numpy array): array for the mock variance.

    '''
    A = calc_sigma_coeff(images, variance)
    sigma_map = np.sqrt(variance)
    temp = np.sqrt(mock_img)
    temp[np.isnan(temp)] = 0.0
    mock_sigma = sigma_map + A[:, np.newaxis, np.newaxis] * temp
    mock_variance = mock_sigma ** 2
    return mock_variance


"""

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

def mock_variance(mock_img, images, variance):
    '''
    Generate mock variance map based on given mock image. 
    `Band` is always along the first axis of the array.
    E.g., `images.shape = (4, 300, 300)` corresponds to images in `g, r, i, z` bands.

    Parameters:
        mock_img (numpy array): array for the image of mock galaxy. 
        images (numpy array): array for the original image (onto which the mock galaxy is placed).
        variance (numpy array): array for the original variance map (`hdu[3].data` for HSC).

    Returns:
        mock_variance (numpy array): array for the mock variance.

    '''
    A = calc_sigma_coeff(images, variance)
    sigma_map = np.sqrt(variance)
    mock_sigma = sigma_map + A[:, np.newaxis, np.newaxis] * mock_img
    mock_variance = mock_sigma ** 2
    return mock_variance
"""
