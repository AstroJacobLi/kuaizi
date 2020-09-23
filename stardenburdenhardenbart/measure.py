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


def flux(component):
    """Determine flux in every channel

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    return model.sum(axis=(1, 2))

def SED(component):
    """Determine SED of component

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    return np.sum([*component.parameters[::2]], axis=0)

def centroid(component):
    """Determine centroid of model

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """
    model = component.get_model()
    indices = np.indices(model.shape)
    centroid = np.array([np.sum(ind * model) for ind in indices]) / model.sum()
    return centroid + component.bbox.origin


def cen_peak(component):
    """Determine position of the pixel with maximum intensity of a model

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


def shape(component, show_fig=False, weight=None):
    """Determine b/a ratio `q` and position angle `pa` of model by calculating its second moments.

    TODO: add weight function

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    weight: 2-D numpy array, W(x, y)
    """

    def raw_moment(data, i_order, j_order, weight):
        n_depth, n_row, n_col = data.shape
        y, x = np.mgrid[:n_row, :n_col]
        if weight is None:
            data = data * x**i_order * y**j_order
        else:
            data = data * weight * x**i_order * y**j_order
        return np.sum(data, axis=(1, 2))
    
    model = component.get_model()
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
        norm = scarlet.display.AsinhMapping(minimum=0, stretch=0.1, Q=1)
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
        ax.axis('image')
        plt.show()
    return {'q': q, 'pa': pa}
