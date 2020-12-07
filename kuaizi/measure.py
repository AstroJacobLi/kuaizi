import numpy as np
import scarlet
import matplotlib.pyplot as plt
import sep
import os
import copy

# Measurements should be done before convolving any real PSF! So, don't render the scene!

def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return tuple(
        np.array(np.unravel_index(np.argmax(model), model.shape)) + np.array(component.bbox.origin)
    )


def flux(components):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    tot_flux = 0
    if not isinstance(components, list):
        components = [components]
    for comp in components:
        model = comp.get_model()
        tot_flux += model.sum(axis=(1, 2))
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
    
    if isinstance(components, scarlet.Component):
        # Single component
        model = components.get_model()
        indices = np.indices(model.shape)
        centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
        return centroid + components.bbox.origin
    else:
        if observation is None:
            raise ValueError('Please provide `Observation`.')
            return
        else:
            # Multiple components
            blend = scarlet.Blend(components, observation)
            model = blend.get_model()
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

    _, y_cen, x_cen = centroid(components, observation=observation) # Determine the centroid, averaged through channels
    blend = scarlet.Blend(components, observation) # Render model image
    model = blend.get_model()

    R50 = R_frac(components, observation, frac=0.5)
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

def R_frac(components, observation=None, frac=0.5, weight_order=0):
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

    _, y_cen, x_cen = centroid(components, observation=observation) # Determine the centroid, averaged through channels
    s = shape(components, observation, show_fig=False, weight_order=weight_order)
    q = s['q']
    theta = np.deg2rad(s['pa'])

    blend = scarlet.Blend(components, observation) # Render model image
    model = blend.get_model()
    total_flux = model.sum(axis=(1, 2))

    depth = model.shape[0]
    r_frac = []

    if depth > 1:
        for i in range(depth):
            r_max = max(model.shape)
            r_ = np.linspace(0, r_max, 500)
            flux_ = sep.sum_ellipse(model[i], x_cen, y_cen, 1, 1 * q[i], theta[i], r=r_)[0]
            flux_ /= total_flux[i]
            func = UnivariateSpline(r_, flux_ - frac, s=0)
            r_frac.append(func.roots()[0])
    else: # might be buggy
        r_max = max(model.shape)
        r_ = np.linspace(0, r_max, 500)
        flux_ = sep.sum_ellipse(model[0], x_cen, y_cen, 1, 1 * q[0], theta[0], r=r_)[0]
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

    _, y_cen, x_cen = centroid(components, observation=observation) # Determine the centroid, averaged through channels
    s = shape(components, observation, show_fig=False, weight_order=weight_order)
    q = s['q']
    theta = np.deg2rad(s['pa'])

    blend = scarlet.Blend(components, observation) # Render model image
    model = blend.get_model()

    depth = model.shape[0]
    kron = []

    if depth > 1:
        for i in range(depth):
            r_max = max(model.shape)
            r = sep.kron_radius(model[i], x_cen, y_cen, 1, 1 * q[i], theta[i], r_max)[0]
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
    if isinstance(components, scarlet.Component):
        # Single component
        model = components.get_model()
    else:
        if observation is None:
            raise ValueError('Please provide `Observation`.')
            return
        else:
            # Multiple components
            blend = scarlet.Blend(components, observation)
            model = blend.get_model()

    if weight_order < 0:
        raise ValueError('Weight order cannot be negative, this will introduce infinity!') 
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
    major_axis = eigvecs[np.arange(len(eigvecs)), np.argmax(eigvals, axis=1), :]
    sign = np.sign(major_axis[:, 1]) # sign of y-component
    pa = np.rad2deg(np.arccos(np.dot(major_axis, [1, 0])))
    pa = np.array([x - 180 if abs(x) > 90 else x for x in pa])
    pa *= sign

    if show_fig:
        fig, ax = plt.subplots()
        norm = scarlet.display.AsinhMapping(minimum=0, stretch=0.5, Q=1)
        ax.imshow(scarlet.display.img_to_rgb(model, norm=norm))

        def make_lines(eigvals, eigvecs, mean, i):
            """Make lines a length of 2 stddev."""
            std = np.sqrt(eigvals[i])
            vec = 1 * std * eigvecs[:, i] / np.hypot(*eigvecs[:, i])
            x, y = np.vstack((mean - vec, mean, mean + vec)).T
            return x, y
        
        mean = np.array([x_c[0], y_c[0]])
        ax.plot(*make_lines(eigvals[0], eigvecs[0], mean, 0), marker='o', color='blue', alpha=0.4)
        ax.plot(*make_lines(eigvals[0], eigvecs[0], mean, -1), marker='o', color='red', alpha=0.4)

        mean = np.array([x_c[2], y_c[2]])
        ax.plot(*make_lines(eigvals[2], eigvecs[2], mean, 0), marker='o', color='blue', alpha=0.4)
        ax.plot(*make_lines(eigvals[2], eigvecs[2], mean, -1), marker='o', color='red', alpha=0.4)

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
        _, y_cen, x_cen = centroid(components, observation=observation) # Determine the centroid, averaged through channels
    
    blend = scarlet.Blend(components, observation) # Render model image
    model = blend.get_model()

    depth = model.shape[0]
    mu_cen = []

    if depth > 1:
        for i in range(depth):
            img = model[i]
            mu = img[int(y_cen) - 1:int(y_cen) + 2, int(x_cen) - 1:int(x_cen) + 2].mean()
            mu_cen.append(mu)
    mu_cen = -2.5 * np.log10(np.array(mu_cen) / (pixel_scale**2)) + zeropoint
    return mu_cen


def makeMeasurement(components, observation, frac=0.5, zeropoint=27.0, pixel_scale=0.168, weight_order=0):
    measure_dict = {}
    _cen = centroid(components, observation)
    measure_dict['x_cen'] = _cen[2]
    measure_dict['y_cen'] = _cen[1]
    w = observation.frame.wcs
    ra, dec = w.wcs_pix2world(measure_dict['x_cen'], measure_dict['y_cen'], 0)
    measure_dict['ra_cen'] = float(ra)
    measure_dict['dec_cen'] = float(dec)

    _cen = winpos(components, observation)
    measure_dict['x_cen_winpos'] = _cen[1].mean()
    measure_dict['y_cen_winpos'] = _cen[0].mean()
    ra, dec = w.wcs_pix2world(measure_dict['x_cen_winpos'], measure_dict['y_cen_winpos'], 0)
    measure_dict['ra_cen_winpos'] = float(ra)
    measure_dict['dec_cen_winpos'] = float(dec)

    measure_dict['flux'] = flux(components)
    measure_dict['mag'] = -2.5 * np.log10(measure_dict['flux']) + zeropoint
    measure_dict['R50'] = R_frac(components, observation, frac=frac) * pixel_scale
    _shape = shape(components, observation, weight_order=weight_order)
    measure_dict['q'] = _shape['q']
    measure_dict['pa'] = _shape['pa']
    measure_dict['SB0'] = mu_central(components, observation, method='centroid', 
                                                zeropoint=zeropoint, pixel_scale=pixel_scale)
    measure_dict['SB0_winpos'] = mu_central(components, observation, method='winpos', 
                                                zeropoint=zeropoint, pixel_scale=pixel_scale)

    return measure_dict


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

    ## First, we measure the basic properties of the galaxy to get initial values for fitting
    measure_dict = makeMeasurement(components, observation, frac=0.5, zeropoint=zeropoint, pixel_scale=pixel_scale, weight_order=0)

    ## Then, we save the scarlet model and the inverse-variance map into a FITS file
    img_fn = os.path.join(file_dir, f'{prefix}-{index:04d}-scarlet-model.fits')
    invvar_fn = os.path.join(file_dir, f'{prefix}-{index:04d}-scarlet-invvar.fits')

    _ = save_to_fits(model.mean(axis=0), img_fn)

    invvar = []  # Calculate inv-variance map for the scene
    if isinstance(components, scarlet.Component):
        # Single component
        src = components
        invvar.append(1 / np.sum(list(map(lambda a: a.std.data ** 2, src.parameters[1::2])), axis=0)) # boxed inv-variance
        _ = save_to_fits(invvar[0], invvar_fn)
    else:
        # Multiple components
        for src in components:
            invvar.append(1 / np.sum(list(map(lambda a: 1 / a.std.data ** 2, src.parameters[1::2])), axis=0)) # boxed inv-variance
        slices = tuple((src._model_frame_slices[1:], src._model_slices[1:])
                        for src in components)
        full_invvar = np.zeros(blend.frame.shape[1:], dtype=blend.frame.dtype)  # Inv-variance map in the full scene
        full_invvar = scarlet.blend._add_models(*invvar, full_model=full_invvar, slices=slices)  # the inv-variance of background is 0???
        _ = save_to_fits(full_invvar, invvar_fn)
    

    ## Fit a Sersic model using `pymfit`

    # Initial params that are different from defaults.
    # syntax is {parameter: [value, low, high]}
    pa_init = - np.sign(measure_dict['pa'].mean()) * (90 - abs(measure_dict['pa'].mean()))
    init_params = dict(PA=[pa_init, -90, 90],
                    n=[1.0, 0.01, 5.0],) 
                    #e=[1 - measure_dict['q'].mean(), 0, 1]) 
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

    w = observation.frame.wcs
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
                             save_fn=os.path.join('./Figures/', f'{prefix}-{index:04d}-Sersic.png'));

    ## Make a dictionary containing non-parametric measurements and Sersci fitting results
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
    mu = mag + 5 * np.log10(Re) + 2.5 * np.log10(gamma(2 * n + 1) * np.pi) - 5 * n * np.log10(bn(n))
    return mu

def cal_mue(n, Re, mag):
    '''
    Re should be circularized: a * np.sqrt(b/a)!
    https://ui.adsabs.harvard.edu/abs/10.1071/AS05001
    '''
    from scipy.special import gamma, polygamma
    mu = mag + 5 * np.log10(Re) + 2.5 * np.log10(gamma(2 * n + 1) * np.pi) + 2.5 * bn(n) / np.log(10) - 5 * n * np.log10(bn(n))
    return mu