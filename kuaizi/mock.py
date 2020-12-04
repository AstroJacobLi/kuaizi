from __future__ import division, print_function
import os, sys
import sep
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as mpl_ellip
from contextlib import contextmanager

from astropy.io import fits
from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from .display import display_single, SEG_CMAP, ORG

def mock_variance(mock_images, images, variance):
    '''
    Generate mock variance map based on given mock image. 
    `Band` is always along the first axis of the array.
    E.g., `images.shape = (4, 300, 300)` corresponds to images in `g, r, i, z` bands.

    Parameters:
        mock_images (numpy array): array for mock image. 
        images (numpy array): array for the original image (onto which the mock galaxy is placed).
        variance (numpy array): array for the original variance map (`hdu[3].data` for HSC).

    Returns:
        mock_variance (numpy array): array for the mock variance.

    '''
    from .utils import calc_sigma_coeff
    A = calc_sigma_coeff(images, variance)
    sigma_map = np.sqrt(variance)
    mock_sigma = sigma_map + A[:, np.newaxis, np.newaxis] * mock_images
    mock_variance = mock_sigma ** 2
    return mock_variance

def gen_mock_lsbg():
    '''
    Generate mock low surface brightness galaxies. 

    '''