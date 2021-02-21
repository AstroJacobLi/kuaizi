import os
import copy
import numpy as np
from astropy import wcs
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, Column
from tqdm import tqdm
import urllib

DECaLS_pixel_scale = 0.262

class TqdmUpTo(tqdm):
    """
    Provides ``update_to(n)`` which uses ``tqdm.update(delta_n)``.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b (int, optional): Number of blocks transferred so far [default: 1].
        bsize (int, optional): Size of each block (in tqdm units) [default: 1].
        tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.

        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

################# DECaLS download related ##################
def download_decals_cutout(ra, dec, size, band, layer='dr8-south', pixel_unit=False, 
                    output_dir='./', output_name='DECaLS_img', overwrite=True):
    '''
    Download DECaLS small image cutout of a given image. Maximum size is 3000 * 3000 pix.
    
    Parameters:
        ra (float): RA (degrees)
        dec (float): DEC (degrees)
        size (float): image size in pixel or arcsec. If pixel_unit = True, it's in pixel.
        band (string): such as 'r' or 'g'
        layer (string): data release of DECaLS. If your object is too north, try 'dr8-north'. 
            For details, please check http://legacysurvey.org/dr8/description/.
        pixel_unit (bool): If true, size will be in pixel unit.
        output_dir (str): directory of output files.
        output_name (str): prefix of output images. The suffix `.fits` will be appended automatically. 
        overwrite (bool): overwrite files or not.

    Return:
        None
    '''

    if pixel_unit is False:
        s = size / DECaLS_pixel_scale
    else:
        s = size
    
    URL = 'http://legacysurvey.org/viewer/fits-cutout?ra={0}&dec={1}&pixscale={2}&layer={3}&size={4:.0f}&bands={5}'.format(ra, dec, DECaLS_pixel_scale, layer, s, band)
    filename = output_name + '_' + band + '.fits'
    if not os.path.isfile(filename):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ') 
    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ')                            
    elif os.path.isfile(filename) and not overwrite:
        print('!!!The image "' + output_dir + filename + '" already exists!!!')
    return

# Generate DECaLS tractor url, given bricknames
def download_decals_tractor_catalog(bricknames, layer='dr8', output_dir='./', overwrite=True, return_table=True):
    '''
    Generate DECaLS tractor url, given bricknames. Work for python 2 and 3.
    '''
    if not isinstance(bricknames, list):
        if not isinstance(bricknames, np.ndarray):
            bricknames = [bricknames]
        else:
            bricknames = list(bricknames.astype(str))
    for brick in bricknames:
        URL = f'http://portal.nersc.gov/project/cosmo/data/legacysurvey/{layer}/south/tractor/{brick[:3]}/tractor-{brick}.fits'
        filename = os.path.join(output_dir, f'tractor-{brick}.fits')
        if not os.path.isfile(filename):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
                urllib.request.urlretrieve(URL, filename=filename,
                                        reporthook=t.update_to, data=None)
            print('# Downloading ' + filename + ' finished! ') 
        elif os.path.isfile(filename) and overwrite:
            os.remove(filename)
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
                urllib.request.urlretrieve(URL, filename=filename,
                                        reporthook=t.update_to, data=None)
            print('# Downloading ' + filename + ' finished! ')                            
        elif os.path.isfile(filename) and not overwrite:
            print('!!!The file "' + filename + '" already exists!!!')
    if return_table:
        from astropy.table import Table, vstack
        return vstack([Table.read(os.path.join(output_dir, f'tractor-{brick}.fits')) for brick in bricknames])