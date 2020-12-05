from __future__ import division, print_function
import os, sys
import sep
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
from contextlib import contextmanager

import pickle

from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from .display import display_single, SEG_CMAP, ORG

# Class to provide compact input of instrument data and metadata
class Data:
    """ This is a rudimentary class to set the necessary information for a scarlet run.

    While it is possible for scarlet to run without wcs or psf,
    it is strongly recommended not to, which is why these entry are not optional.
    """
    def __init__(self, images, variances, masks, channels, wcs, psfs=None, info=None):
        self.images = images
        self.variances = variances
        self.masks = masks
        self.channels = channels
        self.wcs = wcs
        self.psfs = psfs
        self.info = info

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
    This is a class for mock galaxy.
    
    bkg_image
    bkg_variance
    bkg_mask

    image
    model
    variance

    info
    
    """
    def __init__(self, bkg, model):
        self.bkg = bkg
        self.model = model
    
    @property
    def bkg(self):
        return self._bkg
    @bkg.setter
    def bkg(self, bkg):
        self._bkg = bkg

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model


    def set_mock(self):
        # Set the `mock` object
        mock_img = self.model.images + self.bkg.images
        mock_var = mock_variance(self.model.images, self.bkg.images, self.bkg.variances)
        self._mock = Data(mock_img, mock_var, self.bkg.masks, self.bkg.channels, self.bkg.wcs, self.bkg.psfs)
    
    @property
    def mock(self):
        return self._mock



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
            raise ValueError('Other formats are not supported yet. Please use `pkl`.')

    @classmethod
    def read(cls, filename, format='pkl'):
        if format == 'pkl':
            with open(filename, "rb") as fp:
                gal = pickle.load(fp)
                fp.close()
            return gal
        else:
            raise ValueError('Other formats are not supported yet. Please use `pkl`.')
        return

    


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

def gen_mock_lsbg():
    '''
    Generate mock low surface brightness galaxies. 

    '''