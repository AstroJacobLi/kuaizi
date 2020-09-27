import numpy as np
import scarlet
import matplotlib.pyplot as plt

def max_pixel(component):
    """Determine pixel with maximum value

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return tuple(
        np.unravel_index(np.argmax(model), model.shape) + component.bbox.origin
    )


def flux(components):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    tot_flux = 0
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
    import sep
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
    
    return np.array(x_), np.array(y_)
    
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
    Determine the radius R (in pixels), the flux within R has a fraction of `frac` over the total flux.

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation: 

    frac: float
        fraction of lights within this R.

    """
    import sep
    from scipy.interpolate import interp1d, UnivariateSpline

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
    import sep

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
        norm = scarlet.display.AsinhMapping(minimum=0, stretch=1, Q=1)
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

def mu_central(components, observation=None, zeropoint=27.0, pixel_scale=0.168, weight_order=0):
    """
    Determine the central surface brightness, by calculating the average of 9 pixels around the centroid

    Parameters
    ----------
    components: a list of `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    observation

    """
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