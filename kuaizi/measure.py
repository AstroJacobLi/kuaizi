import numpy as np
import scarlet
import matplotlib.pyplot as plt
import sep
import os
import copy
import gc
import statmorph

import kuaizi as kz
from kuaizi.display import display_single
from scarlet.source import StarletSource
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.convolution import convolve, Gaussian2DKernel
import astropy.units as u
from astropy.io import fits
import astropy.wcs as wcs

import statmorph

#############################################
# Use Statmorph to measure galaxy morphology#
#############################################

# Measurements should be done before convolving any
# real PSF! So, don't render the scene!


def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return tuple(
        np.array(np.unravel_index(np.argmax(model), model.shape)) +
        np.array(component.bbox.origin)
    )


def flux(components, observation):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    tot_flux = 0
    if not isinstance(components, list):
        components = [components]

    mask = (observation.weights == 0)
    blend = scarlet.Blend(components, observation)
    model = blend.get_model()

    tot_flux = (model * ~mask).sum(axis=(1, 2))

    return tot_flux


def SED(component):
    """Determine SED of single component

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    return np.sum([*component.parameters[::2]], axis=0)


def centroid(components, observation=None):
    """Determine centroid of (multiple) components

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Components to analyze

    Returns
    -------
        y, x
    """
    if not isinstance(components, list):
        components = [components]

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid


def winpos(components, observation=None):
    """Calculate more accurate object centroids using ‘windowed’ algorithm.
    https://sep.readthedocs.io/en/v1.0.x/api/sep.winpos.html

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Components to analyze

    Returns
    -------
        y, x: winpos in each channel
    """
    if not isinstance(components, list):
        components = [components]

    # Determine the centroid, averaged through channels
    _, y_cen, x_cen = centroid(components, observation=observation)

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask

    R50 = flux_radius(components, observation, frac=0.5)
    sig = 2. / 2.35 * R50  # R50 is half-light radius for each channel

    depth = model.shape[0]

    x_ = []
    y_ = []
    if depth > 1:
        for i in range(depth):
            xwin, ywin, flag = sep.winpos(model[i], x_cen, y_cen, sig[i])
            x_.append(xwin)
            y_.append(ywin)

    return np.array(y_), np.array(x_)


'''
def cen_peak(component):
    """Determine position of the pixel with maximum intensity of a model

    TODO: expand to multiple components

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    peak_set = []
    for i in range(len(model)):
        peak_set.append(np.mean(np.where(model[i] == np.max(model[i])), axis=1))
    peak_set = np.array(peak_set)
    return peak_set + component.bbox.origin[1:]
'''


def flux_radius(components, observation=None, frac=0.5, weight_order=0):
    """
    Determine the radius R (in pixels, along semi-major axis), 
    the flux within R has a fraction of `frac` over the total flux.

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation: 

    frac: float
        fraction of lights within this R.

    """
    from scipy.interpolate import interp1d, UnivariateSpline

    if not isinstance(components, list):
        components = [components]

    # Determine the centroid, averaged through channels
    _, y_cen, x_cen = centroid(components, observation=observation)
    s = shape(components, observation, show_fig=False,
              weight_order=weight_order)
    q = s['q']
    theta = np.deg2rad(s['pa'])

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask

    total_flux = model.sum(axis=(1, 2))

    depth = model.shape[0]
    r_frac = []

    # sep.sum_ellipse is very slow! Try to improve!
    if depth > 1:
        for i in range(depth):
            r_max = max(model.shape)
            r_ = np.linspace(0, r_max, 500)
            flux_ = sep.sum_ellipse(
                model[i], [x_cen], [y_cen], 1, 1 * q[i], theta[i], r=r_)[0]
            flux_ /= total_flux[i]
            func = UnivariateSpline(r_, flux_ - frac, s=0)
            r_frac.append(func.roots()[0])
    else:  # might be buggy
        r_max = max(model.shape)
        r_ = np.linspace(0, r_max, 500)
        flux_ = sep.sum_ellipse(
            model[0], [x_cen], [y_cen], 1, 1 * q[0], theta[0], r=r_)[0]
        flux_ /= total_flux[0]
        func = UnivariateSpline(r_, flux_ - frac, s=0)
        r_frac.append(func.roots()[0])

    return np.array(r_frac)


def kron_radius(components, observation=None, weight_order=0):
    """
    Determine the Kron Radius 

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation

    """
    if not isinstance(components, list):
        components = [components]

    # Determine the centroid, averaged through channels
    _, y_cen, x_cen = centroid(components, observation=observation)
    s = shape(components, observation, show_fig=False,
              weight_order=weight_order)
    q = s['q']
    theta = np.deg2rad(s['pa'])

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask

    depth = model.shape[0]
    kron = []

    if depth > 1:
        for i in range(depth):
            r_max = max(model.shape)
            r = sep.kron_radius(model[i], x_cen, y_cen,
                                1, 1 * q[i], theta[i], r_max)[0]
            kron.append(r)

    return np.array(kron)


def raw_moment(data, i_order, j_order, weight):
    n_depth, n_row, n_col = data.shape
    y, x = np.mgrid[:n_row, :n_col]
    if weight is None:
        data = data * x**i_order * y**j_order
    else:
        data = data * weight * x**i_order * y**j_order
    return np.sum(data, axis=(1, 2))


def shape(components, observation=None, show_fig=False, weight_order=0):
    """Determine b/a ratio `q` and position angle `pa` of model by calculating its second moments.

    TODO: add weight function

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    weight_order: W(x, y) = I(x, y) ** (weight_order)

    """
    if not isinstance(components, list):
        components = [components]

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask

    if weight_order < 0:
        raise ValueError(
            'Weight order cannot be negative, this will introduce infinity!')
    elif weight_order == 0:
        weight = None
    else:
        weight = model ** weight_order

    # zeroth-order moment: total flux
    w00 = raw_moment(model, 0, 0, weight)

    # first-order moment: centroid
    w10 = raw_moment(model, 1, 0, weight)
    w01 = raw_moment(model, 0, 1, weight)
    x_c = w10 / w00
    y_c = w01 / w00

    # second-order moment: b/a ratio and position angle
    m11 = raw_moment(model, 1, 1, weight) / w00 - x_c * y_c
    m20 = raw_moment(model, 2, 0, weight) / w00 - x_c**2
    m02 = raw_moment(model, 0, 2, weight) / w00 - y_c**2
    cov = np.array([m20, m11, m11, m02]).T.reshape(-1, 2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # q = b/a
    q = np.sqrt(np.min(eigvals, axis=1) / np.max(eigvals, axis=1))

    # position angle PA: between the major axis and the east (positive x-axis)
    major_axis = eigvecs[np.arange(
        len(eigvecs)), np.argmax(eigvals, axis=1), :]
    sign = np.sign(major_axis[:, 1])  # sign of y-component
    pa = np.rad2deg(np.arccos(np.dot(major_axis, [1, 0])))
    pa = np.array([x - 180 if abs(x) > 90 else x for x in pa])
    pa *= sign

    if show_fig:
        fig, ax = plt.subplots()
        norm = scarlet.display.AsinhMapping(minimum=-0.1, stretch=0.5, Q=1)
        ax.imshow(scarlet.display.img_to_rgb(model, norm=norm))

        def make_lines(eigvals, eigvecs, mean, i):
            """Make lines a length of 2 stddev."""
            std = np.sqrt(eigvals[i])
            vec = 1.5 * std * eigvecs[:, i] / np.hypot(*eigvecs[:, i])
            x, y = np.vstack((mean - vec, mean, mean + vec)).T
            return x, y

        mean = np.array([x_c[0], y_c[0]])
        ax.plot(*make_lines(eigvals[0], eigvecs[0],
                            mean, 0), marker='o', color='blue', alpha=0.4)
        ax.plot(*make_lines(eigvals[0], eigvecs[0],
                            mean, -1), marker='o', color='red', alpha=0.4)

        mean = np.array([x_c[2], y_c[2]])
        ax.plot(*make_lines(eigvals[2], eigvecs[2],
                            mean, 0), marker='o', color='blue', alpha=0.4)
        ax.plot(*make_lines(eigvals[2], eigvecs[2],
                            mean, -1), marker='o', color='red', alpha=0.4)

        ax.axis('image')
        plt.show()

    return {'q': q, 'pa': pa}


def mu_central(components, observation=None, method='centroid', zeropoint=27.0, pixel_scale=0.168, weight_order=0):
    """
    Determine the central surface brightness, by calculating the average of 9 pixels around the centroid

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation

    method: 'centroid' or 'winpos'

    """
    if not isinstance(components, list):
        components = [components]
    if method == 'winpos':
        y_cen, x_cen = winpos(components, observation=observation)
        y_cen = y_cen.mean()
        x_cen = x_cen.mean()
    else:
        # Determine the centroid, averaged through channels
        _, y_cen, x_cen = centroid(components, observation=observation)

    blend = scarlet.Blend(components, observation)
    model = blend.get_model()
    mask = (observation.weights == 0)
    model = model * ~mask

    depth = model.shape[0]
    mu_cen = []

    if depth > 1:
        for i in range(depth):
            img = model[i]
            mu = img[int(y_cen) - 1:int(y_cen) + 2,
                     int(x_cen) - 1:int(x_cen) + 2].mean()
            mu_cen.append(mu)
    mu_cen = -2.5 * np.log10(np.array(mu_cen) / (pixel_scale**2)) + zeropoint
    return mu_cen


def makeMeasurement(components, observation, aggr_mask=None, makesegmap=True, sigma=2,
                    zeropoint=27.0, pixel_scale=0.168,
                    out_prefix=None, show_fig=True, **kwargs):
    """
    Measure the structural parameters of the galaxy, after modeling with scarlet.

    Parameters:
        components (list of Scarlet sources): a list of sources that will be blended with observation
        observation (scarlet.observation): observation data
        aggr_mask (2-D binary array): the aggressive mask, which is generated during modeling
        sigma (float): the threshold when generating a segmentation map for `statmorph`
        zeropoint (float): photometric zeropoint of the input data
        pixel_scale (float): the angular size of pixel, default is 0.168 (for HSC)
        out_prefix (str): the prefix for each key in output dictionary
        show_fig (bool): if True, a default `statmorph` figure will be shown

    Returns:
        measure_dict (dict): a dictionary containing all measurements
    """
    min_cutout_size = max([comp.bbox.shape[1] for comp in components])
    # Multi-components enabled
    _blend = scarlet.Blend(components, observation)
    lower_left = np.min([np.array(comp.bbox.origin)
                        for comp in components], axis=0)
    upper_right = np.max([np.array(comp.bbox.origin) +
                         np.array(comp.bbox.shape) for comp in components], axis=0)
    bbox = scarlet.Box(upper_right - lower_left, origin=lower_left)
    bbox.center = np.array(bbox.origin) + np.array(bbox.shape) // 2
    bbox.shape = tuple(int(i * 1.5) for i in bbox.shape)
    bbox.origin = tuple(bbox.center[i] - bbox.shape[i] // 2 for i in range(3))

    models = _blend.get_model()  # PSF-free model
    models = observation.render(models)  # PSF-convoled model
    models = models[:, bbox.origin[1]:bbox.origin[1] + bbox.shape[1],
                    bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]

    data = observation.data
    weights = observation.weights
    psfs = observation.psf.get_model()
    if aggr_mask is None:
        mask = (weights.sum(axis=0) == 0)
    else:
        mask = aggr_mask | (weights.sum(axis=0) == 0)

    data = data[:, bbox.origin[1]:bbox.origin[1] + bbox.shape[1],
                bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    data = np.ascontiguousarray(data)
    mask = mask[bbox.origin[1]:bbox.origin[1] + bbox.shape[1],
                bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    mask = np.ascontiguousarray(mask)
    weights = weights[:, bbox.origin[1]:bbox.origin[1] +
                      bbox.shape[1], bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    weights = np.ascontiguousarray(weights)

    # Flux and magnitude in each band
    measure_dict = {}
    measure_dict['flux'] = flux(components, observation)
    # normalized against g-band
    SED = (measure_dict['flux'] / measure_dict['flux'][0])
    measure_dict['mag'] = -2.5 * np.log10(measure_dict['flux']) + zeropoint

    # We take the model and weight map in g-band. Such that we can use the SED to get
    # surface brightness in other bands.
    # A sky background is estimated on the original image,
    # and we run `sep` to generate a 1-sigma segmentation map.
    # Then we run `statmorph` using that segmap

    filt = 0
    img = models[filt]
    data_avg = np.average(data, weights=np.sqrt(
        weights.sum(axis=(1, 2))), axis=0)

    if makesegmap:
        bkg = sep.Background(data_avg, bh=12, bw=12, mask=mask)
        _, segmap = sep.extract(img - bkg.globalback, sigma, err=bkg.globalrms, minarea=1,
                                deblend_cont=1,
                                mask=mask, segmentation_map=True)

        # Only select relevant detections.
        cen_ind = [segmap[int(comp.center[0] - bbox.origin[1]),
                          int(comp.center[1] - bbox.origin[2])] for comp in components]
        segmap[~np.add.reduce(
            [segmap == ind for ind in cen_ind]).astype(bool)] = 0
        segmap = (segmap > 0)
        segmap = convolve(segmap, Gaussian2DKernel(4)) > 0.01

        img[~segmap] = np.nan
    else:
        segmap = np.ones_like(img)

    source_morphs = statmorph.source_morphology(
        img, segmap, weightmap=np.sqrt(weights[filt]),
        n_sigma_outlier=15, min_cutout_size=min_cutout_size, cutout_extent=2,
        mask=mask, psf=psfs[filt])
    morph = source_morphs[0]

    measure_dict['xc_centroid'] = morph.xc_centroid
    measure_dict['yc_centroid'] = morph.yc_centroid
    measure_dict['xc_peak'] = morph.xc_peak
    measure_dict['yc_peak'] = morph.yc_peak
    measure_dict['ellipticity_centroid'] = morph.ellipticity_centroid
    measure_dict['elongation_centroid'] = morph.elongation_centroid
    measure_dict['orientation_centroid'] = morph.orientation_centroid
    measure_dict['xc_asymmetry'] = morph.xc_asymmetry
    measure_dict['yc_asymmetry'] = morph.yc_asymmetry
    measure_dict['ellipticity_asymmetry'] = morph.ellipticity_asymmetry
    measure_dict['elongation_asymmetry'] = morph.elongation_asymmetry
    measure_dict['orientation_asymmetry'] = morph.orientation_asymmetry
    measure_dict['rpetro_circ'] = morph.rpetro_circ
    measure_dict['rpetro_ellip'] = morph.rpetro_ellip
    measure_dict['rhalf_circ'] = morph.rhalf_circ
    measure_dict['rhalf_ellip'] = morph.rhalf_ellip
    measure_dict['r20'] = morph.r20
    measure_dict['r50'] = morph.r50
    measure_dict['r80'] = morph.r80
    measure_dict['SB_0_circ'] = -2.5 * np.log10(morph.SB_0_circ * SED / (
        pixel_scale**2)) + zeropoint   # in mag per arcsec2
    measure_dict['SB_0_ellip'] = -2.5 * np.log10(
        morph.SB_0_ellip * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['SB_eff_circ'] = -2.5 * np.log10(
        morph.SB_eff_circ * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['SB_eff_ellip'] = -2.5 * np.log10(
        morph.SB_eff_ellip * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['Gini'] = morph.gini
    measure_dict['M20'] = morph.m20
    measure_dict['F(G, M20)'] = morph.gini_m20_bulge
    measure_dict['S(G, M20)'] = morph.gini_m20_merger
    measure_dict['sn_per_pixel'] = morph.sn_per_pixel
    measure_dict['C'] = morph.concentration
    measure_dict['A'] = morph.asymmetry
    measure_dict['A_outer'] = morph.outer_asymmetry
    measure_dict['A_shape'] = morph.shape_asymmetry
    measure_dict['S'] = morph.smoothness
    measure_dict['sersic_amplitude'] = morph.sersic_amplitude
    measure_dict['sersic_rhalf'] = morph.sersic_rhalf
    measure_dict['sersic_n'] = morph.sersic_n
    measure_dict['sersic_xc'] = morph.sersic_xc
    measure_dict['sersic_yc'] = morph.sersic_yc
    measure_dict['sersic_ellip'] = morph.sersic_ellip
    measure_dict['sersic_theta'] = morph.sersic_theta
    measure_dict['sky_mean'] = morph.sky_mean
    measure_dict['sky_median'] = morph.sky_median
    measure_dict['sky_sigma'] = morph.sky_sigma
    measure_dict['flag'] = morph.flag
    measure_dict['flag_sersic'] = morph.flag_sersic

    if show_fig:
        from statmorph.utils.image_diagnostics import make_figure
        fig = make_figure(morph, **kwargs)

    measure_dict_new = {}
    if out_prefix is not None:
        for key in measure_dict.keys():
            measure_dict_new['_'.join([out_prefix, key])] = measure_dict[key]
        measure_dict = measure_dict_new

    return measure_dict, morph


def _write_to_row(row, measurement):
    '''
    Write the output of `makeMeasurement` to a Row of astropy.table.Table.

    Parameters:
        row (astropy.table.Row): one row of the table.
        measurement (dict): the output dictionary of `kuaizi.measure.makeMeasurement`

    Returns:
        row (astropy.table.Row)
    '''
    row['flux'] = measurement['flux']
    row['mag'] = measurement['mag']
    row['SB_0'] = measurement['SB_0_circ']
    row['SB_eff_circ'] = measurement['SB_eff_circ']
    row['SB_eff_ellip'] = measurement['SB_eff_ellip']

    row['xc_cen'] = measurement['xc_centroid']
    row['yc_cen'] = measurement['yc_centroid']
    row['xc_sym'] = measurement['xc_asymmetry']
    row['yc_sym'] = measurement['yc_asymmetry']
    row['ell_cen'] = measurement['ellipticity_centroid']
    row['ell_sym'] = measurement['ellipticity_asymmetry']
    row['PA_cen'] = measurement['orientation_centroid']
    row['PA_sym'] = measurement['orientation_asymmetry']

    row['rhalf_circ'] = measurement['rhalf_circ']
    row['rhalf_ellip'] = measurement['rhalf_ellip']
    row['r20'] = measurement['r20']
    row['r50'] = measurement['r50']
    row['r80'] = measurement['r80']
    row['Gini'] = measurement['Gini']
    row['M20'] = measurement['M20']
    row['F(G,M20)'] = measurement['F(G, M20)']
    row['S(G,M20)'] = measurement['S(G, M20)']

    row['C'] = measurement['C']
    row['A'] = measurement['A']
    row['A_outer'] = measurement['A_outer']
    row['A_shape'] = measurement['A_shape']
    row['S'] = measurement['S']
    row['sersic_n'] = measurement['sersic_n']
    row['sersic_rhalf'] = measurement['sersic_rhalf']
    row['sersic_ell'] = measurement['sersic_ellip']
    row['sersic_PA'] = measurement['sersic_theta']
    row['sersic_xc'] = measurement['sersic_xc']
    row['sersic_yc'] = measurement['sersic_yc']
    row['sersic_amp'] = measurement['sersic_amplitude']
    row['flag'] = measurement['flag']
    row['flag_sersic'] = measurement['flag_sersic']

    return row


def Sersic_fitting(components, observation=None, file_dir='./Models/', prefix='LSBG', index=0,
                   zeropoint=27.0, pixel_scale=0.168, save_fig=True):
    '''
    Fit a single Sersic model to the rendered model. Using `pymfit` by Johnny Greco https://github.com/johnnygreco/pymfit

    '''
    from .utils import save_to_fits
    from .display import display_pymfit_model
    import pymfit

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    if observation is None:
        raise ValueError('Please provide `Observation`.')

    if isinstance(components, scarlet.Component):
        # Single component
        model = components.get_model()
    else:
        # Multiple components
        blend = scarlet.Blend(components, observation)
        model = blend.get_model()

    # First, we measure the basic properties of the galaxy to get initial values for fitting
    measure_dict = makeMeasurement(components, observation, frac=0.5,
                                   zeropoint=zeropoint, pixel_scale=pixel_scale, weight_order=0)

    # Then, we save the scarlet model and the inverse-variance map into a FITS file
    img_fn = os.path.join(file_dir, f'{prefix}-{index:04d}-scarlet-model.fits')
    invvar_fn = os.path.join(
        file_dir, f'{prefix}-{index:04d}-scarlet-invvar.fits')

    _ = save_to_fits(model.mean(axis=0), img_fn)

    invvar = []  # Calculate inv-variance map for the scene
    if isinstance(components, scarlet.Component):
        # Single component
        src = components
        invvar.append(1 / np.sum(list(map(lambda a: a.std.data **
                                          2, src.parameters[1::2])), axis=0))  # boxed inv-variance
        _ = save_to_fits(invvar[0], invvar_fn)
    else:
        # Multiple components
        for src in components:
            invvar.append(1 / np.sum(list(map(lambda a: 1 / a.std.data **
                                              2, src.parameters[1::2])), axis=0))  # boxed inv-variance
        slices = tuple((src._model_frame_slices[1:], src._model_slices[1:])
                       for src in components)
        # Inv-variance map in the full scene
        full_invvar = np.zeros(
            blend.model_frame.shape[1:], dtype=blend.model_frame.dtype)
        # the inv-variance of background is 0???
        full_invvar = scarlet.blend._add_models(
            *invvar, full_model=full_invvar, slices=slices)
        _ = save_to_fits(full_invvar, invvar_fn)

    # Fit a Sersic model using `pymfit`

    # Initial params that are different from defaults.
    # syntax is {parameter: [value, low, high]}
    pa_init = - np.sign(measure_dict['pa'].mean()) * \
        (90 - abs(measure_dict['pa'].mean()))
    init_params = dict(PA=[pa_init, -90, 90],
                       n=[1.0, 0.01, 5.0],)
    # e=[1 - measure_dict['q'].mean(), 0, 1])
    # create a config dictionary
    config = pymfit.sersic_config(init_params, img_shape=img_fn)

    # run imfit
    # note that the image file is a multi-extension cube, which explains the '[#]' additions
    # also note that this config will be written to config_fn. if you already have a
    # config file written, then use config=None (default) and skip the above step.
    sersic = pymfit.run(img_fn, config_fn=os.path.join(file_dir, 'config.txt'),
                        mask_fn=None, config=config, var_fn=invvar_fn,
                        out_fn=os.path.join(file_dir, 'best-fit.dat'),
                        weights=True)
    if isinstance(components, scarlet.Component):
        # Single component:
        sersic['X0_scene'] = sersic['X0'] + components.bbox.origin[2]
        sersic['Y0_scene'] = sersic['Y0'] + components.bbox.origin[1]

    w = observation.model_frame.wcs
    ra, dec = w.wcs_pix2world(sersic['X0_scene'], sersic['Y0_scene'], 0)
    sersic['RA0'] = float(ra)
    sersic['DEC0'] = float(dec)

    sersic_model = pymfit.Sersic(sersic)

    if save_fig:
        if isinstance(components, scarlet.Component):
            # Single component
            blend = scarlet.Blend([components], observation)
        display_pymfit_model(blend, sersic_model.params, figsize=(30, 6), cmap='Greys_r',
                             colorbar=True, fontsize=17, show=False,
                             save_fn=os.path.join('./Figures/', f'{prefix}-{index:04d}-Sersic.png'))

    # Make a dictionary containing non-parametric measurements and Sersci fitting results
    measurement = copy.deepcopy(measure_dict)
    for key in sersic_model.params.keys():
        measurement['_'.join(['sersic', key])] = sersic_model.params[key]
    measurement['sersic_SB0'] = sersic_model.mu_0
    measurement['sersic_SBe'] = sersic_model.mu_e
    measurement['sersic_mag'] = sersic_model.m_tot

    return sersic_model, measurement


# Sersic constant
def bn(n):
    temp = 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2) + 131 / (
        1148175 * n**3) - 2194697 / (30690717750 * n**4)
    return temp


def delta_bn(n):
    temp = 2 - 4 / (405 * n**2) - 2 * 46 / (25515 * n**3) - 3 * 131 / (
        1148175 * n**4) + 4 * 2194697 / (30690717750 * n**5)
    return temp


def cal_mu0(n, Re, mag):
    '''
    Re should be circularized: a * np.sqrt(b/a)!
    https://ui.adsabs.harvard.edu/abs/10.1071/AS05001
    '''
    from scipy.special import gamma, polygamma
    mu = mag + 5 * np.log10(Re) + 2.5 * np.log10(gamma(2 * n + 1)
                                                 * np.pi) - 5 * n * np.log10(bn(n))
    return mu


def cal_mue(n, Re, mag):
    '''
    Re should be circularized: a * np.sqrt(b/a)!
    https://ui.adsabs.harvard.edu/abs/10.1071/AS05001
    '''
    from scipy.special import gamma, polygamma
    mu = mag + 5 * np.log10(Re) + 2.5 * np.log10(gamma(2 * n + 1)
                                                 * np.pi) + 2.5 * bn(n) / np.log(10) - 5 * n * np.log10(bn(n))
    return mu


def _measure_image(data, coord, index, pixel_scale=0.168, zeropoint=27.0, bright=False,
                   show_figure=False, show_figure_statmorph=True, figure_dir='./Figures/',
                   tigress=True, logger=None, **kwargs):
    if logger is None:
        from .utils import set_logger
        logger = set_logger(__name__, f'measure_img.log')

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
    x, y, _ = sep.winpos(data.images.mean(
        axis=0), cen_obj['x'], cen_obj['y'], 6)
    ra, dec = data.wcs.wcs_pix2world(x, y, 0)
    cen_obj['x'] = x
    cen_obj['y'] = y
    cen_obj['ra'] = ra
    cen_obj['dec'] = dec
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
                                   min_grad=-0.01,  # the initial guess of box size is as large as possible
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
                                                                logger=None)

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

    # # `box_flag` is for objects which fall in the initial Starlet box
    # box_flag = np.unique(
    #     segmap[starlet_extent[2]:starlet_extent[3], starlet_extent[0]:starlet_extent[1]]) - 1
    # if len(box_flag) > 0:
    #     box_flag = np.delete(np.sort(box_flag), 0)
    #     overlap_flag[box_flag] = True
    # if len(overlap_flag) > 0:
    #     obj_cat_cpct = obj_cat[overlap_flag] # obj_cat_cpct is the catalog for compact sources

    # Remove the source from `obj_cat_cpct` if it is the central galaxy
    # if dist[cen_indx_highfreq] < 1 * u.arcsec:
    #     obj_cat_cpct.remove_rows(
    #         np.where(obj_cat_cpct['index'] == cen_indx_highfreq)[0])

    overlap_flag |= (dist < 1 * u.arcsec)

    for ind in np.where(overlap_flag)[0]:
        segmap[segmap == ind + 1] = 0

    smooth_radius = 3
    gaussian_threshold = 0.01
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
        show_fig=False,
        visual_gaia=False,
        b=64,
        f=3,
        pixel_scale=pixel_scale,
        minarea=20,   # only want large things
        deblend_nthresh=36,
        deblend_cont=0.01,
        sky_subtract=True,
        logger=None)

    catalog_c = SkyCoord(obj_cat['ra'], obj_cat['dec'], unit='deg')
    dist = cen_obj_coord.separation(catalog_c)
    cen_indx_big = obj_cat_ori[np.argsort(dist)[0]]['index']

    # mask out big objects that are NOT identified in the high_freq step
    segmap = segmap_big.copy()
    # segbox = segmap[starlet_extent[2]:starlet_extent[3], starlet_extent[0]:starlet_extent[1]]
    # box_flag = np.unique(segbox) - 1
    # if len(box_flag) > 0:
    #     box_flag = np.delete(np.sort(box_flag), 0)
    #     for ind in box_flag:
    #         if np.sum(segbox == ind + 1) / np.sum(segmap == ind + 1) > 0.5: segmap[segmap == ind + 1] = 0
    #     box_flag = np.delete(box_flag, np.where(box_flag == cen_indx_big)[
    #         0])  # dont include the central galaxy
    #     obj_cat_big = obj_cat[box_flag]
    # else:
    #     obj_cat_big = obj_cat
    segmap[segmap == cen_indx_big + 1] = 0
    for ind in np.where(dist < 2 * u.arcsec)[0]:
        segmap[segmap == ind + 1] = 0
    # `obj_cat_big` is catalog of the big/high SNR objects in the image

    smooth_radius = 5
    gaussian_threshold = 0.01
    mask_conv = np.copy(segmap)
    mask_conv[mask_conv > 0] = 1
    mask_conv = convolve(mask_conv.astype(
        float), Gaussian2DKernel(smooth_radius))
    # This `seg_mask_large` masks large bright sources
    seg_mask_large = (mask_conv >= gaussian_threshold)

    mask = (seg_mask_large + seg_mask + msk_star.astype(bool))

    # Crop the image for measurement
    components = [starlet_source]
    min_cutout_size = max([comp.bbox.shape[1] for comp in components])

    lower_left = np.min([np.array(comp.bbox.origin)
                        for comp in components], axis=0)
    upper_right = np.max([np.array(comp.bbox.origin) +
                         np.array(comp.bbox.shape) for comp in components], axis=0)
    bbox = scarlet.Box(upper_right - lower_left, origin=lower_left)
    bbox.center = np.array(bbox.origin) + np.array(bbox.shape) // 2
    bbox.shape = tuple(int(i * 1.5) for i in bbox.shape)
    bbox.origin = tuple(bbox.center[i] - bbox.shape[i] // 2 for i in range(3))

    images = data.images
    weights = data.weights
    psfs = data.psfs.get_model()

    images = images[:, bbox.origin[1]:bbox.origin[1] + bbox.shape[1],
                    bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    images = np.ascontiguousarray(images)
    mask = mask[bbox.origin[1]:bbox.origin[1] + bbox.shape[1],
                bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    mask = np.ascontiguousarray(mask)
    weights = weights[:, bbox.origin[1]:bbox.origin[1] +
                      bbox.shape[1], bbox.origin[2]:bbox.origin[2] + bbox.shape[2]]
    weights = np.ascontiguousarray(weights)

    ############ Very rough photometry ##############
    measure_dict = {}
    measure_dict['flux'] = (images * (~mask.astype(bool))).sum(axis=(1, 2))
    # normalized against g-band
    SED = (measure_dict['flux'] / measure_dict['flux'][0])
    measure_dict['mag'] = -2.5 * np.log10(measure_dict['flux']) + zeropoint

    # We take the model and weight map in g-band. Such that we can use the SED to get
    # surface brightness in other bands.
    # A sky background is estimated on the original image,
    # and we run `sep` to generate a 1-sigma segmentation map.
    # Then we run `statmorph` using that segmap
    sigma = 2
    filt = 2
    img = np.average(images, weights=weights.sum(axis=(1, 2)),
                     axis=0)  # This is the average image
    # img = images[filt]#models[filt]
    bkg = sep.Background(img, bh=12, bw=12, mask=mask)
    _, segmap = sep.extract(img, sigma, err=bkg.globalrms, minarea=1,
                            deblend_cont=1,
                            mask=None, segmentation_map=True)

    # mask = np.copy(segmap)

    # cen_ind = [segmap[int(comp.center[0] - comp.bbox.origin[1]),
    #                   int(comp.center[1] - comp.bbox.origin[2])] for comp in components]
    cen_ind = [segmap[int(bbox.center[1] - bbox.origin[1]),
                      int(bbox.center[2] - bbox.origin[2])] for comp in components]
    segmap[~np.add.reduce([segmap == ind for ind in cen_ind]).astype(bool)] = 0

    segmap = (segmap > 0)
    segmap = convolve(segmap, Gaussian2DKernel(4)) > 0.01

    source_morphs = statmorph.source_morphology(
        img, segmap, weightmap=np.sqrt(weights[filt]),
        n_sigma_outlier=10, min_cutout_size=min_cutout_size,
        cutout_extent=2.,
        skybox_size=24,
        # petro_extent_cas=1,
        # petro_fraction_gini=0.1,
        mask=mask,
        psf=psfs[filt], **kwargs)
    morph = source_morphs[0]

    measure_dict['xc_centroid'] = morph.xc_centroid
    measure_dict['yc_centroid'] = morph.yc_centroid
    measure_dict['xc_peak'] = morph.xc_peak
    measure_dict['yc_peak'] = morph.yc_peak
    measure_dict['ellipticity_centroid'] = morph.ellipticity_centroid
    measure_dict['elongation_centroid'] = morph.elongation_centroid
    measure_dict['orientation_centroid'] = morph.orientation_centroid
    measure_dict['xc_asymmetry'] = morph.xc_asymmetry
    measure_dict['yc_asymmetry'] = morph.yc_asymmetry
    measure_dict['ellipticity_asymmetry'] = morph.ellipticity_asymmetry
    measure_dict['elongation_asymmetry'] = morph.elongation_asymmetry
    measure_dict['orientation_asymmetry'] = morph.orientation_asymmetry
    measure_dict['rpetro_circ'] = morph.rpetro_circ
    measure_dict['rpetro_ellip'] = morph.rpetro_ellip
    measure_dict['rhalf_circ'] = morph.rhalf_circ
    measure_dict['rhalf_ellip'] = morph.rhalf_ellip
    measure_dict['r20'] = morph.r20
    measure_dict['r50'] = morph.r50
    measure_dict['r80'] = morph.r80
    measure_dict['SB_0_circ'] = -2.5 * np.log10(morph.SB_0_circ * SED / (
        pixel_scale**2)) + zeropoint   # in mag per arcsec2
    measure_dict['SB_0_ellip'] = -2.5 * np.log10(
        morph.SB_0_ellip * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['SB_eff_circ'] = -2.5 * np.log10(
        morph.SB_eff_circ * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['SB_eff_ellip'] = -2.5 * np.log10(
        morph.SB_eff_ellip * SED / (pixel_scale**2)) + zeropoint  # in mag per arcsec2
    measure_dict['Gini'] = morph.gini
    measure_dict['M20'] = morph.m20
    measure_dict['F(G, M20)'] = morph.gini_m20_bulge
    measure_dict['S(G, M20)'] = morph.gini_m20_merger
    measure_dict['sn_per_pixel'] = morph.sn_per_pixel
    measure_dict['C'] = morph.concentration
    measure_dict['A'] = morph.asymmetry
    measure_dict['A_outer'] = morph.outer_asymmetry
    measure_dict['A_shape'] = morph.shape_asymmetry
    measure_dict['S'] = morph.smoothness
    measure_dict['sersic_amplitude'] = morph.sersic_amplitude
    measure_dict['sersic_rhalf'] = morph.sersic_rhalf
    measure_dict['sersic_n'] = morph.sersic_n
    measure_dict['sersic_xc'] = morph.sersic_xc
    measure_dict['sersic_yc'] = morph.sersic_yc
    measure_dict['sersic_ellip'] = morph.sersic_ellip
    measure_dict['sersic_theta'] = morph.sersic_theta
    measure_dict['sky_mean'] = morph.sky_mean
    measure_dict['sky_median'] = morph.sky_median
    measure_dict['sky_sigma'] = morph.sky_sigma
    measure_dict['flag'] = morph.flag
    measure_dict['flag_sersic'] = morph.flag_sersic

    from statmorph.utils.image_diagnostics import make_figure
    fig = make_figure(morph, **kwargs)
    plt.savefig(os.path.join(figure_dir, 'statmorph_' +
                str(index) + '.png'), dpi=100, bbox_inches='tight')
    if show_figure_statmorph:
        plt.show()
    else:
        plt.close()
    # measure_dict_new = {}
    # if out_prefix is not None:
    #     for key in measure_dict.keys():
    #         measure_dict_new['_'.join([out_prefix, key])] = measure_dict[key]
    #     measure_dict = measure_dict_new

    return measure_dict, morph


def measure_image_tigress(env_dict, lsbg, name='viz-id', channels='griz',
                          pixel_scale=0.168, bright_thresh=17.0,
                          prefix='nsa', output_dir='./Measure', figure_dir='./Figure', show_figure_statmorph=False,
                          logger=None, global_logger=None, fail_logger=None):
    __name__ = "fitting_vanilla_obs_tigress"

    from kuaizi.utils import padding_PSF
    from kuaizi.mock import Data
    import unagi  # for HSC saturation mask

    index = lsbg[name]
    # whether this galaxy is a very bright one
    bright = (lsbg['mag_auto_i'] < bright_thresh)

    if logger is None:
        from .utils import set_logger
        logger = set_logger(__name__, os.path.join(
            output_dir, f'{prefix}-{index}.log'), level='info')

    if bright:
        logger.info(
            f"This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")
        print(
            f"    This galaxy is very bright, with i-mag = {lsbg['mag_auto_i']:.2f}")

    logger.info(f'Working directory: {os.getcwd()}')
    print(f'    Working directory: {os.getcwd()}')

    kz.utils.set_env(**env_dict)
    # kz.utils.set_matplotlib(style='default')

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

        measurement, morph = _measure_image(data, lsbg_coord, index, bright=bright,
                                            pixel_scale=pixel_scale,
                                            zeropoint=27.0, tigress=True,
                                            show_figure=False, show_figure_statmorph=show_figure_statmorph,
                                            figure_dir=figure_dir,
                                            logger=logger)

        if global_logger is not None:
            global_logger.info(
                f'Image measurement Task succeeded for `{lsbg["prefix"]}` in `{channels}`.')

        gc.collect()

        return measurement

    except Exception as e:
        logger.error(e)
        print(e)
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

        return measurement
