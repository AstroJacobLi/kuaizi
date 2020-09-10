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


def shape(component, show_fig=False):
    """Determine b/a ratio `q` and position angle `pa` of model by calculating its second moments.

    Parameters
    ----------
    component: `scarlet.Component` or `scarlet.ComponentTree`
        Component to analyze
    """

    def raw_moment(data, i_order, j_order):
        n_depth, n_row, n_col = data.shape
        y, x = np.mgrid[:n_row, :n_col]
        data = data * x**i_order * y**j_order
        return np.sum(data, axis=(1, 2))
    
    model = component.get_model()
    data_sum = np.sum(model, axis=(1, 2))

    # first-order moment: centroid
    w10 = raw_moment(model, 1, 0)
    w01 = raw_moment(model, 0, 1)
    x_mean = w10 / data_sum
    y_mean = w01 / data_sum

    # second-order moment: b/a ratio and position angle
    m11 = (raw_moment(model, 1, 1) - x_mean * w01 - y_mean * w10 + x_mean * y_mean * data_sum) / data_sum
    m20 = (raw_moment(model, 2, 0) - 2 * x_mean * w10 + data_sum * x_mean**2) / data_sum
    m02 = (raw_moment(model, 0, 2) - 2 * y_mean * w01 + data_sum * y_mean**2) / data_sum
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
        mean = np.array([x_mean[0], y_mean[0]])
        ax.plot(*make_lines(eigvals[0], eigvecs[0], mean, 0), marker='o', color='white', alpha=0.5)
        ax.plot(*make_lines(eigvals[0], eigvecs[0], mean, -1), marker='o', color='red', alpha=0.5)
        ax.axis('image')
        plt.show()
    return {'q': q, 'pa': pa}
