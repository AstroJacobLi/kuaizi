import sep
import numpy as np
import scarlet
from scarlet.wavelet import mad_wavelet, Starlet

# Class to provide compact input of instrument data and metadata
class Data:
    """ This is a rudimentary class to set the necessary information for a scarlet run.

    While it is possible for scarlet to run without wcs or psf,
    it is strongly recommended not to, which is why these entry are not optional.
    """
    def __init__(self,images, wcs, psfs, channels):
        self._images = images
        self.wcs = wcs
        self.psfs = psfs
        self.channels = channels

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

def interpolate(data_lr, data_hr):
    ''' Interpolate low resolution data to high resolution

    Parameters
    ----------
    data_lr: Data
        low resolution Data
    data_hr: Data
        high resolution Data

    Result
    ------
    interp: numpy array
        the images in data_lr interpolated to the grid of data_hr
    '''
    frame_lr = scarlet.Frame(data_lr.images.shape, wcs = data_lr.wcs, channels = data_lr.channels)
    frame_hr = scarlet.Frame(data_hr.images.shape, wcs = data_hr.wcs, channels = data_hr.channels)

    coord_lr0 = (np.arange(data_lr.images.shape[1]), np.arange(data_lr.images.shape[1]))
    coord_hr = (np.arange(data_hr.images.shape[1]), np.arange(data_hr.images.shape[1]))
    coord_lr = scarlet.resampling.convert_coordinates(coord_lr0, frame_lr, frame_hr)

    interp = []
    for image in data_lr.images:
        interp.append(scarlet.interpolation.sinc_interp(image[None, :, :], coord_hr, coord_lr, angle=None)[0].T)
    return np.array(interp)


def makeCatalog(datas, lvl=3, wave=True):
    ''' Creates a detection catalog by combining low and high resolution data

    This function is used for detection before running scarlet.
    It is particularly useful for stellar crowded fields and for detecting high frequency features.

    Parameters
    ----------
    datas: array
        array of Data objects
    lvl: int
        detection lvl
    wave: Bool
        set to True to use wavelet decomposition of images before combination

    Returns
    -------
    catalog: sextractor catalog
        catalog of detected sources
    bg_rms: array
        background level for each data set
    '''
    if len(datas) == 1:
        hr_images = datas[0].images / np.sum(datas[0].images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(hr_images, axis=0)
    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalisation of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)
    if np.size(detect_image.shape) == 3:
        if wave:
            # Wavelet detection in the first three levels
            wave_detect = Starlet(detect_image.mean(axis=0), lvl=4).coefficients
            wave_detect[:, -1, :, :] = 0
            detect = Starlet(coefficients=wave_detect).image
        else:
            # Direct detection
            detect = detect_image.mean(axis=0)
    else:
        if wave:
            wave_detect = Starlet(detect_image).coefficients
            detect = wave_detect[0][0]
        else:
            detect = detect_image

    bkg = sep.Background(detect)
    catalog = sep.extract(detect, lvl, err=bkg.globalrms)

    if len(datas) ==1:
        bg_rms = mad_wavelet(datas[0].images)
    else:
        bg_rms = []
        for data in datas:
            bg_rms.append(mad_wavelet(data.images))

    return catalog, bg_rms
