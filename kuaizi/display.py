from __future__ import division, print_function

import os
import copy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy import wcs
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import (AsymmetricPercentileInterval,
                                   ZScaleInterval, make_lupton_rgb)
from matplotlib import colors, rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import (AutoMinorLocator, FormatStrFormatter,
                               MaxNLocator, NullFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.sequential import (Blues_9, Greys_9, OrRd_9,
                                               Purples_9, YlGn_9)


def random_cmap(ncolors=256, background_color='white'):
    """Random color maps, from ``kungpao`` https://github.com/dr-guangtou/kungpao.

    Generate a matplotlib colormap consisting of random (muted) colors.
    A random colormap is very useful for plotting segmentation images.

    Parameters
        ncolors : int, optional
            The number of colors in the colormap.  The default is 256.
        random_state : int or ``~numpy.random.RandomState``, optional
            The pseudo-random number generator state used for random
            sampling.  Separate function calls with the same
            ``random_state`` will generate the same colormap.

    Returns
        cmap : `matplotlib.colors.Colormap`
            The matplotlib colormap with random colors.

    Notes
        Based on: colormaps.py in photutils

    """
    prng = np.random.mtrand._rand

    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)

    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    if background_color is not None:
        if background_color not in colors.cnames:
            raise ValueError('"{0}" is not a valid background color '
                             'name'.format(background_color))
        rgb[0] = colors.hex2color(colors.cnames[background_color])

    return colors.ListedColormap(rgb)


# About the Colormaps
IMG_CMAP = copy.copy(matplotlib.cm.get_cmap("viridis"))
IMG_CMAP.set_bad(color='black')
SEG_CMAP = random_cmap(ncolors=512, background_color=u'white')
SEG_CMAP.set_bad(color='white')
SEG_CMAP.set_under(color='white')

BLK = Greys_9.mpl_colormap
ORG = OrRd_9.mpl_colormap
BLU = Blues_9.mpl_colormap
GRN = YlGn_9.mpl_colormap
PUR = Purples_9.mpl_colormap


def _display_single(img,
                    pixel_scale=0.168,
                    physical_scale=None,
                    xsize=8,
                    ysize=8,
                    ax=None,
                    stretch='arcsinh',
                    scale='zscale',
                    scale_manual=None,
                    contrast=0.25,
                    no_negative=False,
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                    cmap=IMG_CMAP,
                    scale_bar=True,
                    scale_bar_length=5.0,
                    scale_bar_fontsize=20,
                    scale_bar_y_offset=0.5,
                    scale_bar_color='w',
                    scale_bar_loc='left',
                    color_bar=False,
                    color_bar_loc=1,
                    color_bar_width='75%',
                    color_bar_height='5%',
                    color_bar_fontsize=18,
                    color_bar_color='w',
                    add_text=None,
                    text_fontsize=30,
                    text_y_offset=0.80,
                    text_color='w'):

    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    if scale_manual is not None:
        assert len(scale_manual) == 2, '# length of manual scale must be two!'
        zmin, zmax = scale_manual

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
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
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        ax1.text(text_x_0, text_y_0,
                 r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig, zmin, zmax
    return ax1, zmin, zmax


def display_single(img,
                   pixel_scale=0.168,
                   physical_scale=None,
                   xsize=8,
                   ysize=8,
                   ax=None,
                   stretch='arcsinh',
                   scale='zscale',
                   contrast=0.25,
                   no_negative=False,
                   lower_percentile=1.0,
                   upper_percentile=99.0,
                   cmap=IMG_CMAP,
                   norm=None,
                   scale_bar=True,
                   scale_bar_length=5.0,
                   scale_bar_fontsize=20,
                   scale_bar_y_offset=0.5,
                   scale_bar_color='w',
                   scale_bar_loc='left',
                   color_bar=False,
                   color_bar_loc=1,
                   color_bar_width='75%',
                   color_bar_height='5%',
                   color_bar_fontsize=18,
                   color_bar_color='w',
                   add_text=None,
                   usetex=True,
                   text_fontsize=30,
                   text_y_offset=0.80,
                   text_color='w'):
    """
    Display single image using ``arcsinh`` stretching, "zscale" scaling and ``viridis`` colormap as default.
    This function is from ``kungpao`` https://github.com/dr-guangtou/kungpao.

    Parameters:
        img (numpy 2-D array): The image array.
        pixel_scale (float): The pixel size, in unit of "arcsec/pixel".
        physical_scale (bool): Whether show the image in physical scale.
        xsize (int): Width of the image, default = 8.
        ysize (int): Height of the image, default = 8.
        ax (``matplotlib.pyplot.axes`` object): The user could provide axes on which the figure will be drawn.
        stretch (str): Stretching schemes. Options are "arcsinh", "log", "log10" and "linear".
        scale (str): Scaling schemes. Options are "zscale" and "percentile".
        contrast (float): Contrast of figure.
        no_negative (bool): If true, all negative pixels will be set to zero.
        lower_percentile (float): Lower percentile, if using ``scale="percentile"``.
        upper_percentile (float): Upper percentile, if using ``scale="percentile"``.
        cmap (str): Colormap.
        scale_bar (bool): Whether show scale bar or not.
        scale_bar_length (float): The length of scale bar.
        scale_bar_y_offset (float): Offset of scale bar on y-axis.
        scale_bar_fontsize (float): Fontsize of scale bar ticks.
        scale_bar_color (str): Color of scale bar.
        scale_bar_loc (str): Scale bar position, options are "left" and "right".
        color_bar (bool): Whether show colorbar or not.
        add_text (str): The text you want to add to the figure.
        usetex (bool): whether render the text in LaTeX.
        text_fontsize (float): Fontsize of text.
        text_y_offset (float): Offset of text on y-axis.
        text_color (str): Color of text.

    Returns:
        ax: If the input ``ax`` is not ``None``.

    """

    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    # Stretch option
    if stretch.strip() == 'arcsinh':
        img_scale = np.arcsinh(img)
    elif stretch.strip() == 'log':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log(img)
    elif stretch.strip() == 'log10':
        if no_negative:
            img[img <= 0.0] = 1.0E-10
        img_scale = np.log10(img)
    elif stretch.strip() == 'linear':
        img_scale = img
    else:
        raise Exception("# Wrong stretch option.")

    # Scale option
    if scale.strip() == 'zscale':
        try:
            zmin, zmax = ZScaleInterval(
                contrast=contrast).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    elif scale.strip() == 'percentile':
        try:
            zmin, zmax = AsymmetricPercentileInterval(
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile).get_limits(img_scale)
        except IndexError:
            # TODO: Deal with problematic image
            zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = np.nanmin(img_scale), np.nanmax(img_scale)

    show = ax1.imshow(img_scale, origin='lower', cmap=cmap, norm=norm,
                      vmin=zmin, vmax=zmax)

    # Hide ticks and tick labels
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax1.axis('off')

    # Put scale bar on the image
    (img_size_x, img_size_y) = img.shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
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
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        if usetex:
            ax.text(text_x_0, text_y_0,
                    r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)
        else:
            ax.text(text_x_0, text_y_0, add_text,
                    fontsize=text_fontsize, color=text_color)

    # Put a color bar on the image
    if color_bar:
        ax_cbar = inset_axes(ax1,
                             width=color_bar_width,
                             height=color_bar_height,
                             loc=color_bar_loc)
        if ax is None:
            cbar = plt.colorbar(show, ax=ax1, cax=ax_cbar,
                                orientation='horizontal')
        else:
            cbar = plt.colorbar(show, ax=ax, cax=ax_cbar,
                                orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color=color_bar_color)
        cbar.ax.yaxis.set_tick_params(color=color_bar_color)
        cbar.outline.set_edgecolor(color_bar_color)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color=color_bar_color, fontsize=color_bar_fontsize)

    if ax is None:
        return fig
    return ax1


def display_multiple(data_array, text=None, ax=None, scale_bar=True, **kwargs):
    """
    Display multiple images together using the same strecth and scale.

    Parameters:
        data_array (list): A list containing images which are numpy 2-D arrays.
        text (str): A list containing strings which you want to add to each panel.
        ax (list): The user could provide a list of axes on which the figure will be drawn.
        **kwargs: other arguments in ``display_single``.

    Returns:
        axes: If the input ``ax`` is not ``None``.

    """
    if ax is None:
        fig, axes = plt.subplots(
            1, len(data_array), figsize=(len(data_array) * 4, 8))
    else:
        axes = ax

    if text is None:
        _, zmin, zmax = _display_single(
            data_array[0], ax=axes[0], scale_bar=scale_bar, **kwargs)
    else:
        _, zmin, zmax = _display_single(
            data_array[0], add_text=text[0], ax=axes[0], scale_bar=scale_bar, **kwargs)
    for i in range(1, len(data_array)):
        if text is None:
            _display_single(data_array[i], ax=axes[i], scale_manual=[
                            zmin, zmax], scale_bar=False, **kwargs)
        else:
            _display_single(data_array[i], add_text=text[i], ax=axes[i], scale_manual=[
                            zmin, zmax], scale_bar=False, **kwargs)

    plt.subplots_adjust(wspace=0.0)
    if ax is None:
        return fig
    else:
        return axes


def display_HSC_cutout_rgb(images, ax=None, half_width=None):
    import scarlet
    from scarlet.display import AsinhMapping

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Crop
    if half_width is not None:
        if half_width * 2 < min(images.shape[1], images.shape[2]):
            cen = (images.shape[1] // 2, images.shape[2] // 2)
            images = images[:, cen[0] - half_width:cen[0] +
                            half_width, cen[1] - half_width:cen[1] + half_width]

    # Norm color
    f_c = np.array([1.9, 1.2, 1., 0.85])
    _images = images * f_c[:, np.newaxis, np.newaxis]

    # Display
    norm = AsinhMapping(minimum=-0.15, stretch=1.2, Q=3)

    img_rgb = scarlet.display.img_to_rgb(_images, norm=norm)
    ax.imshow(img_rgb, origin='lower')
    ax.axis('off')

    if ax is None:
        return fig
    return ax


def display_merian_cutout_rgb(images, filters=list('griz') + ['N708'],
                              ax=None, half_width=None,
                              minimum=-0.15,
                              stretch=1.2, Q=3,
                              color_norm=None
                              ):
    import scarlet
    from scarlet.display import AsinhMapping

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Crop
    if half_width is not None:
        if half_width * 2 < min(images.shape[1], images.shape[2]):
            cen = (images.shape[1] // 2, images.shape[2] // 2)
            images = images[:, cen[0] - half_width:cen[0] +
                            half_width, cen[1] - half_width:cen[1] + half_width]

    # Norm color
    if color_norm is None:
        color_norm = {'g': 1.9, 'r': 1.2, 'i': 1.0,
                      'z': 0.85, 'y': 0.5, 'N708': 0.8}

    f_c = np.array([color_norm[filt] for filt in filters])
    _images = images * f_c[:, np.newaxis, np.newaxis]

    # Display
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)

    img_rgb = scarlet.display.img_to_rgb(_images, norm=norm)

    ax.imshow(img_rgb, origin='lower')
    ax.axis('off')

    if ax is None:
        return fig
    return ax


def display_rgb(images,
                mask=None,
                ax=None,
                stretch=2,
                Q=1,
                minimum=-0.2,
                pixel_scale=0.168,
                scale_bar=True,
                physical_scale=None,
                scale_bar_length=5.0,
                scale_bar_fontsize=20,
                scale_bar_y_offset=0.5,
                scale_bar_color='w',
                scale_bar_loc='left',
                add_text=None,
                usetex=True,
                text_fontsize=30,
                text_y_offset=0.80,
                text_color='w',
                xsize=8.0,
                ysize=8.0,
                hide_ticks=False):
    """
    Display multi-band image in RGB using ``arcsinh`` stretching.

    Parameters:
        images (numpy 3-D array): the images array, dimension follows (Number of filters, Height, Width). 
        pixel_scale (float): The pixel size, in unit of "arcsec/pixel".
        xsize (int): Width of the image, default = 8.
        ysize (int): Height of the image, default = 8.
        ax (``matplotlib.pyplot.axes`` object): The user could provide axes on which the figure will be drawn.
        stretch (float): 
        Q (float): 
        scale_bar (bool): Whether show scale bar or not.
        physical_scale (bool): comoving physical scale, in "kpc/arcsec".
        scale_bar_length (float): The length of scale bar.
        scale_bar_y_offset (float): Offset of scale bar on y-axis.
        scale_bar_fontsize (float): Fontsize of scale bar ticks.
        scale_bar_color (str): Color of scale bar.
        scale_bar_loc (str): Scale bar position, options are "left" and "right".
        add_text (str): The text you want to add to the figure.
        usetex (bool): whether render the text in LaTeX.
        text_fontsize (float): Fontsize of text.
        text_y_offset (float): Offset of text on y-axis.
        text_color (str): Color of text.

    Returns:
        ax: If the input ``ax`` is not ``None``.

    """
    import scarlet
    from scarlet.display import img_to_rgb, AsinhMapping

    if ax is None:
        fig = plt.figure(figsize=(xsize, ysize))
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    f_c = np.array([1.9, 1.2, 1., 0.85])
    channel_map = scarlet.display.channels_to_rgb(len(images)) * f_c

    img_rgb = img_to_rgb(images, norm=AsinhMapping(
        minimum=minimum, stretch=stretch, Q=Q), channel_map=channel_map)

    show = ax1.imshow(img_rgb, origin='lower')

    if mask is not None:
        plt.imshow(mask.astype(float), origin='lower',
                   alpha=0.1, cmap='Greys_r')

    # Put scale bar on the image
    (img_size_x, img_size_y) = images[0].shape
    if physical_scale is not None:
        pixel_scale *= physical_scale
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
        if physical_scale is not None:
            if scale_bar_length > 1000:
                scale_bar_text = r'$%d\ \mathrm{Mpc}$' % int(
                    scale_bar_length / 1000)
            else:
                scale_bar_text = r'$%d\ \mathrm{kpc}$' % int(scale_bar_length)
        else:
            if scale_bar_length < 60:
                scale_bar_text = r'$%d^{\prime\prime}$' % int(scale_bar_length)
            elif 60 < scale_bar_length < 3600:
                scale_bar_text = r'$%d^{\prime}$' % int(scale_bar_length / 60)
            else:
                scale_bar_text = r'$%d^{\circ}$' % int(scale_bar_length / 3600)
        scale_bar_text_size = scale_bar_fontsize

        ax1.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax1.text(
            scale_bar_text_x,
            scale_bar_text_y,
            scale_bar_text,
            fontsize=scale_bar_text_size,
            horizontalalignment='center',
            color=scale_bar_color)
    if add_text is not None:
        text_x_0 = int(img_size_x*0.08)
        text_y_0 = int(img_size_y*text_y_offset)
        if usetex:
            ax.text(text_x_0, text_y_0,
                    r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)
        else:
            ax.text(text_x_0, text_y_0, add_text,
                    fontsize=text_fontsize, color=text_color)

    if hide_ticks:
        # Hide ticks and tick labels
        ax1.tick_params(
            labelbottom=False,
            labelleft=False,
            axis=u'both',
            which=u'both',
            length=0)
        ax1.axis('off')

    if ax is None:
        return fig
    return ax1


def draw_circles(img, catalog, colnames=['x', 'y'], header=None, ax=None, circle_size=30,
                 pixel_scale=0.168, color='r', **kwargs):
    """
    Draw circles on an image according to a catalogue.

    Parameters:
        img (numpy 2-D array): Image itself.
        catalog (``astropy.table.Table`` object): A catalog which contains positions.
        colnames (list): List of string, indicating which columns correspond to positions.
            It can also be "ra" and "dec", but then "header" is needed.
        header: Header file of a FITS image containing WCS information, typically ``astropy.io.fits.header`` object.
        ax (``matplotlib.pyplot.axes`` object): The user could provide axes on which the figure will be drawn.
        circle_size (float): Radius of circle, in pixel.
        pixel_scale (float): Pixel size, in arcsec/pixel. Needed for correct scale bar.
        color (str): Color of circles.
        **kwargs: other arguments of ``display_single``.

    Returns:
        ax: If the input ``ax`` is not ``None``.

    """
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0,
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax

    # ax1.yaxis.set_major_formatter(NullFormatter())
    # ax1.xaxis.set_major_formatter(NullFormatter())
    ax1.axis('off')

    from matplotlib.patches import Ellipse, Rectangle
    if np.any([item.lower() == 'ra' for item in colnames]):
        if header is None:
            raise ValueError(
                '# Header containing WCS must be provided to convert sky coordinates into image coordinates.')
            return
        else:
            w = wcs.WCS(header)
            x, y = w.wcs_world2pix(Table(catalog)[colnames[0]].data.data,
                                   Table(catalog)[colnames[1]].data.data, 0)
    else:
        x, y = catalog[colnames[0]], catalog[colnames[1]]
    display_single(img, ax=ax1, pixel_scale=pixel_scale, **kwargs)
    for i in range(len(catalog)):
        e = Ellipse(xy=(x[i], y[i]),
                    height=circle_size,
                    width=circle_size,
                    angle=0)
        e.set_facecolor('none')
        e.set_edgecolor(color)
        e.set_alpha(0.7)
        e.set_linewidth(1.3)
        ax1.add_artist(e)
    if ax is not None:
        return ax


def draw_rectangles(img, catalog, colnames=['x', 'y'], header=None, ax=None, rectangle_size=[30, 30],
                    pixel_scale=0.168, color='r', **kwargs):
    """
    Draw rectangles on an image according to a catalogue.

    Parameters:
        img (numpy 2-D array): Image itself.
        catalog (``astropy.table.Table`` object): A catalog which contains positions.
        colnames (list): List of string, indicating which columns correspond to positions.
            It can also be "ra" and "dec", but then "header" is needed.
        header: Header file of a FITS image containing WCS information, typically ``astropy.io.fits.header`` object.
        ax (``matplotlib.pyplot.axes`` object): The user could provide axes on which the figure will be drawn.
        rectangle_size (list of floats): Size of rectangles, in pixel.
        pixel_scale (float): Pixel size, in arcsec/pixel. Needed for correct scale bar.
        color (str): Color of rectangles.
        **kwargs: other arguments of ``display_single``.

    Returns:
        ax: If the input ``ax`` is not ``None``.

    """
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(left=0.0, right=1.0,
                            bottom=0.0, top=1.0,
                            wspace=0.00, hspace=0.00)
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.00)
        ax1 = fig.add_subplot(gs[0])
    else:
        ax1 = ax

    # ax1.yaxis.set_major_formatter(NullFormatter())
    # ax1.xaxis.set_major_formatter(NullFormatter())
    # ax1.axis('off')

    from matplotlib.patches import Rectangle
    if np.any([item.lower() == 'ra' for item in colnames]):
        if header is None:
            raise ValueError(
                '# Header containing WCS must be provided to convert sky coordinates into image coordinates.')
            return
        else:
            w = wcs.WCS(header)
            x, y = w.wcs_world2pix(Table(catalog)[colnames[0]].data.data,
                                   Table(catalog)[colnames[1]].data.data, 0)
    else:
        x, y = catalog[colnames[0]], catalog[colnames[1]]
    display_single(img, ax=ax1, pixel_scale=pixel_scale, **kwargs)
    for i in range(len(catalog)):
        e = Rectangle(xy=(x[i] - rectangle_size[0] // 2,
                          y[i] - rectangle_size[1] // 2),
                      height=rectangle_size[0],
                      width=rectangle_size[1],
                      angle=0)
        e.set_facecolor('none')
        e.set_edgecolor(color)
        e.set_alpha(0.7)
        e.set_linewidth(1.3)
        ax1.add_artist(e)
    if ax is not None:
        return ax


def display_scarlet_sources(data, sources, ax=None, show_mask=True, show_ind=None,
                            stretch=2, Q=1, minimum=0.0, show_mark=True, pixel_scale=0.168, scale_bar=True,
                            scale_bar_length=5.0, scale_bar_fontsize=20, scale_bar_y_offset=0.5, scale_bar_color='w',
                            scale_bar_loc='left', add_text=None, usetex=False, text_fontsize=30, text_y_offset=0.80, text_color='w'):
    '''
    Display the scene, including image, mask, and sources.

    Arguments:
        data (kuazi.detection.Data): input `Data` object
        sources (list): a list containing `scarlet` sources
        ax (matplotlib.axes object): input axes object
        show_mask (bool): whether displaying the mask encoded in `data.weights'
        show_ind (list): if not None, only objects with these indices are shown in the figure
        stretch, Q, minimum (float): parameters for displaying image, see https://pmelchior.github.io/scarlet/tutorials/display.html
        show_mark (bool): whether plot the indices of sources in the figure
        pixel_scale (float): default is 0.168 arcsec/pixel.

    Returns:
        ax: if input `ax` is provided
        fig: if no `ax` is provided as input

    '''
    import scarlet
    from scarlet.display import AsinhMapping

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if show_ind is not None:
        sources = np.array(sources)[show_ind]
    # Image
    images = data.images
    # Weights
    weights = data.weights
    mask = (np.sum(data.weights == 0, axis=0) != 0)
    # Display
    f_c = np.array([1.9, 1.2, 1., 0.85])
    channel_map = scarlet.display.channels_to_rgb(len(images)) * f_c
    norm = AsinhMapping(minimum=-0.2, stretch=stretch, Q=Q)
    img_rgb = scarlet.display.img_to_rgb(
        images, norm=norm, channel_map=channel_map)
    plt.imshow(img_rgb)

    if show_mask:
        plt.imshow(mask.astype(float), origin='lower',
                   alpha=0.1, cmap='Greys_r')

    ax.tick_params(
        labelbottom=False,
        labelleft=False,
        axis=u'both',
        which=u'both',
        length=0)
    # ax.axis('off')

    if show_mark:
        # Mark all of the sources from the detection cataog
        for k, src in enumerate(sources):
            if isinstance(src, scarlet.source.PointSource):
                color = 'white'
            elif isinstance(src, scarlet.source.CompactExtendedSource):
                color = 'yellow'
            elif isinstance(src, scarlet.source.SingleExtendedSource):
                color = 'red'
            elif isinstance(src, scarlet.source.MultiExtendedSource):
                color = 'cyan'
            elif isinstance(src, scarlet.source.StarletSource):
                color = 'lime'
            else:
                color = 'gray'
            if hasattr(src, "center"):
                y, x = src.center
                plt.text(x, y, str(k), color=color)
                plt.text(x, y, '+', color=color,
                         horizontalalignment='center', verticalalignment='center')
            else:
                raise ValueError('Wrong!')

    (img_size_x, img_size_y) = data.images[0].shape
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

        ax.plot(
            [scale_bar_x_0, scale_bar_x_1], [scale_bar_y, scale_bar_y],
            linewidth=3,
            c=scale_bar_color,
            alpha=1.0)
        ax.text(
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
            ax.text(text_x_0, text_y_0,
                    r'$\mathrm{'+add_text+'}$', fontsize=text_fontsize, color=text_color)
        else:
            ax.text(text_x_0, text_y_0, add_text,
                    fontsize=text_fontsize, color=text_color)

    if ax is None:
        return fig
    return ax


def display_scarlet_model(blend, zoomin_size=None, ax=None, show_loss=False, show_mask=False, show_gray_mask=True,
                          show_ind=None, add_boxes=True,
                          stretch=2, Q=1, minimum=0.0, channels='grizy', show_mark=True, pixel_scale=0.168, scale_bar=True,
                          scale_bar_length=5.0, scale_bar_fontsize=20, scale_bar_y_offset=0.5, scale_bar_color='w',
                          scale_bar_loc='left', add_text=None, usetex=False, text_fontsize=30, text_y_offset=0.80, text_color='w'):
    '''
    Display the scene, including image, mask, and the models.

    Arguments:
        blend (scarlet.blend): the blend of observation and sources
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
    from scarlet.display import AsinhMapping

    if ax is None:
        if show_loss:
            fig = plt.figure(figsize=(24, 6))
            ax = [fig.add_subplot(1, 4, n + 1) for n in range(4)]
        else:
            fig = plt.figure(figsize=(18, 6))
            ax = [fig.add_subplot(1, 3, n + 1) for n in range(3)]

    # Observation
    observation = blend.observations[0]
    loss = blend.loss

    # Sometimes we only want to show a few sources
    if show_ind is not None:
        sources = np.copy(blend.sources)
        gal_sources = np.array(sources)[show_ind]
        blend = scarlet.Blend(gal_sources, observation)

    if zoomin_size is not None:
        y_cen, x_cen = blend.sources[0].center.astype(int)
        _, y_img_size, x_img_size = observation.data.shape
        # x_cen = observation.model_frame.shape[2] // 2
        # y_cen = observation.model_frame.shape[1] // 2
        size = int(zoomin_size / pixel_scale / 2)  # half-size
        # half-size should not exceed the image half-size
        size = min(size, y_cen, y_img_size - y_cen, x_cen, x_img_size - x_cen)

        # Image
        images = observation.data[:, y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1]
        # Weights
        weights = observation.weights[:, y_cen -
                                      size:y_cen + size + 1, x_cen - size:x_cen + size + 1]
        # Compute model
        model = blend.get_model()[:, y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1]
        # this model is under `model_frame`, i.e. under the modest PSF
    else:
        # Image
        images = observation.data
        # Weights
        weights = observation.weights
        # Compute model
        model = blend.get_model()
        # this model is under `model_frame`, i.e. under the modest PSF

    # Render it in the observed frame
    model_ = observation.render(model)
    # Mask
    mask = (np.sum(weights == 0, axis=0) != 0)
    # Compute residual
    residual = images - model_

    # Create RGB images
    f_c = np.array([1.9, 1.2, 1., 0.85])
    channel_map = scarlet.display.channels_to_rgb(len(channels)) * f_c

    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
    img_rgb = scarlet.display.img_to_rgb(
        images, norm=norm, channel_map=channel_map)
    model_rgb = scarlet.display.img_to_rgb(
        model_, norm=norm, channel_map=channel_map)
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q/2)
    residual_rgb = scarlet.display.img_to_rgb(
        residual, norm=norm, channel_map=channel_map)
    vmax = np.max(np.abs(residual_rgb))

    # Show the data, model, and residual
    if show_mask:
        ax[0].imshow(img_rgb * (~np.tile(mask.T, (3, 1, 1))).T)
        ax[0].set_title("Data")
        ax[1].imshow(model_rgb * (~np.tile(mask.T, (3, 1, 1))).T)
        ax[1].set_title("Model")
        ax[2].imshow(residual_rgb * (~np.tile(mask.T, (3, 1, 1))).T,
                     vmin=-vmax, vmax=vmax, cmap='seismic')
        ax[2].set_title("Residual")
    elif show_gray_mask:
        ax[0].imshow(img_rgb)
        ax[0].set_title("Data")
        ax[1].imshow(model_rgb)
        ax[1].set_title("Model")
        ax[2].imshow(residual_rgb, vmin=-vmax, vmax=vmax, cmap='seismic')
        ax[2].set_title("Residual")
        ax[0].imshow(mask.astype(float), origin='lower',
                     alpha=0.1, cmap='Greys_r')
        ax[1].imshow(mask.astype(float), origin='lower',
                     alpha=0.1, cmap='Greys_r')
        ax[2].imshow(mask.astype(float), origin='lower',
                     alpha=0.1, cmap='Greys_r')
    else:
        ax[0].imshow(img_rgb)
        ax[0].set_title("Data")
        ax[1].imshow(model_rgb)
        ax[1].set_title("Model")
        ax[2].imshow(residual_rgb, vmin=-vmax, vmax=vmax, cmap='seismic')
        ax[2].set_title("Residual")

    if show_mark:
        for k, src in enumerate(blend.sources):
            if isinstance(src, scarlet.source.PointSource):
                color = 'white'
            elif isinstance(src, scarlet.source.CompactExtendedSource):
                color = 'yellow'
            elif isinstance(src, scarlet.source.SingleExtendedSource):
                color = 'red'
            elif isinstance(src, scarlet.source.MultiExtendedSource):
                color = 'cyan'
            elif isinstance(src, scarlet.source.StarletSource):
                color = 'lime'
            else:
                color = 'gray'
            if hasattr(src, "center"):
                y, x = src.center
                if zoomin_size is not None:
                    y = y - y_cen + size
                    x = x - x_cen + size
                ax[0].text(x, y, k, color=color)
                ax[0].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')
                ax[1].text(x, y, k, color=color)
                ax[1].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')
                ax[2].text(x, y, k, color=color)
                ax[2].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')

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

    if show_loss:
        ax[3].plot(-np.array(loss))
        ax[3].set_xlabel('Iteration', labelpad=-40)
        # ax[3].set_ylabel('log-Likelihood')
        ax[3].set_title("log-Likelihood")
        xlim, ylim = ax[3].axes.get_xlim(), ax[3].axes.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        ax[3].set_aspect((xrange / yrange), adjustable='box')
        ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if add_boxes:
        from matplotlib.patches import Rectangle
        for k, src in enumerate(blend.sources):
            box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

            if zoomin_size is not None:
                extent = get_extent(src.bbox, [x_cen - size, y_cen - size])
            else:
                extent = get_extent(src.bbox)
            # print(extent)
            rect = Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                **box_kwargs
            )
            xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
            ax[0].add_patch(rect)
            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)

    from matplotlib.ticker import MaxNLocator, NullFormatter
    for axx in ax:
        axx.yaxis.set_major_locator(MaxNLocator(5))
        axx.xaxis.set_major_locator(MaxNLocator(5))

    # Show the size of PSF. FWHM is 1 arcsec.
    from matplotlib.patches import Circle
    circ1 = Circle((40, 40),
                   radius=1 / 0.168,
                   linewidth=1.,
                   hatch='/////',
                   fc='None',
                   ec='white')
    ax[1].add_patch(circ1)

    plt.subplots_adjust(wspace=0.2)

    if ax is None:
        return fig
    return ax


def display_scarlet_results_tigress(blend, aggr_mask=None, zoomin_size=None, ax=None, show_loss=False, show_mask=False, show_gray_mask=True,
                                    show_ind=None, add_boxes=True,
                                    stretch=2, Q=1, minimum=0.0, channels='grizy', show_mark=True, pixel_scale=0.168, scale_bar=True,
                                    scale_bar_length=10.0, scale_bar_fontsize=15, scale_bar_y_offset=0.3, scale_bar_color='w',
                                    scale_bar_loc='left', add_text=None, usetex=False, text_fontsize=30, text_y_offset=0.80, text_color='w'):
    '''
    Display the scene on Tiger, including
    - zoom-in raw img w/ gray initial mask
    - zoom-in scarlet model w/ gray (final) mask
    - zoom-in residual with dark final mask
    - loss curve

    Arguments:
        blend (scarlet.blend): the blend of observation and sources
        aggr_mask (numpy 2-D array): aggressive mask
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
    from scarlet.display import AsinhMapping

    if ax is None:
        if show_loss:
            fig = plt.figure(figsize=(24, 6))
            ax = [fig.add_subplot(1, 4, n + 1) for n in range(4)]
        else:
            fig = plt.figure(figsize=(18, 6))
            ax = [fig.add_subplot(1, 3, n + 1) for n in range(3)]

    # Observation
    observation = blend.observations[0]
    loss = blend.loss

    ########## Figure 1 ###########
    # In Figure 1, we'd like to show boxes for all sources.
    # The zoomin cutout should be centered at the target galaxy.
    if zoomin_size is not None:
        y_cen, x_cen = blend.sources[0].center.astype(int)
        _, y_img_size, x_img_size = observation.data.shape
        # x_cen = observation.model_frame.shape[2] // 2
        # y_cen = observation.model_frame.shape[1] // 2
        size = int(zoomin_size / pixel_scale / 2)  # half-size
        # half-size should not exceed the image half-size
        size = min(size, y_cen, y_img_size - y_cen, x_cen, x_img_size - x_cen)

        # Image
        images = observation.data[:, y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1]
        # Weights
        weights = observation.weights[:, y_cen -
                                      size:y_cen + size + 1, x_cen - size:x_cen + size + 1]
        if aggr_mask is not None:
            aggr_mask = aggr_mask[y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1].astype(bool)
        # Compute model
        model = blend.get_model()[:, y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1]
        # this model is under `model_frame`, i.e. under the modest PSF
    else:
        # Image
        images = observation.data
        # Weights
        weights = observation.weights
        # Compute model
        model = blend.get_model()
        # this model is under `model_frame`, i.e. under the modest PSF

    # Render it in the observed frame
    model_ = observation.render(model)
    # Mask
    mask = (np.sum(weights == 0, axis=0) != 0)
    # Compute residual
    residual = images - model_

    # Create RGB images
    f_c = np.array([1.9, 1.2, 1., 0.85])
    channel_map = scarlet.display.channels_to_rgb(len(channels)) * f_c
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)

    img_rgb = scarlet.display.img_to_rgb(
        images, norm=norm, channel_map=channel_map)
    channel_map = scarlet.display.channels_to_rgb(len(channels))
    model_rgb = scarlet.display.img_to_rgb(
        model_, norm=norm, channel_map=channel_map)

    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q/2)
    residual_rgb = scarlet.display.img_to_rgb(
        residual, norm=norm, channel_map=channel_map)
    vmax = np.max(np.abs(residual_rgb))

    ax[0].imshow(img_rgb)
    ax[0].imshow(mask.astype(float), origin='lower',
                 alpha=0.1, cmap='Greys_r')
    ax[0].set_title("Data")

    if add_boxes:
        from matplotlib.patches import Rectangle
        for k, src in enumerate(blend.sources):
            box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

            if zoomin_size is not None:
                extent = get_extent(src.bbox, [x_cen - size, y_cen - size])
            else:
                extent = get_extent(src.bbox)
            rect = Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                **box_kwargs
            )
            xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
            ax[0].add_patch(rect)
            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)

    if show_mark:
        for k, src in enumerate(blend.sources):
            if isinstance(src, scarlet.source.PointSource):
                color = 'white'
            elif isinstance(src, scarlet.source.CompactExtendedSource):
                color = 'yellow'
            elif isinstance(src, scarlet.source.SingleExtendedSource):
                color = 'red'
            elif isinstance(src, scarlet.source.MultiExtendedSource):
                color = 'cyan'
            elif isinstance(src, scarlet.source.StarletSource):
                color = 'lime'
            else:
                color = 'gray'
            if hasattr(src, "center"):
                y, x = src.center
                if zoomin_size is not None:
                    y = y - y_cen + size
                    x = x - x_cen + size
                ax[0].text(x, y, k, color=color)
                ax[0].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')
                ax[1].text(x, y, k, color=color)
                ax[1].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')
                ax[2].text(x, y, k, color=color)
                ax[2].text(x, y, '+', color=color,
                           horizontalalignment='center', verticalalignment='center')

    ########## Figure 2 & 3 ###########
    # In Figure 2, we only show the model of target galaxy (by setting `show_ind`) with gray aggressive mask
    if show_ind is not None:
        sources = np.copy(blend.sources)
        gal_sources = np.array(sources)[show_ind]
        blend = scarlet.Blend(gal_sources, observation)

    if zoomin_size is not None:
        # Compute model
        model = blend.get_model()[:, y_cen - size:y_cen +
                                  size + 1, x_cen - size:x_cen + size + 1]
        # this model is under `model_frame`, i.e. under the modest PSF
    else:
        # Compute model
        model = blend.get_model()
        # this model is under `model_frame`, i.e. under the modest PSF

    # Render it in the observed frame
    model_ = observation.render(model)
    # Compute residual
    residual = images - model_

    # Create RGB images
    f_c = np.array([1.9, 1.2, 1., 0.85])
    channel_map = scarlet.display.channels_to_rgb(len(channels)) * f_c

    img_rgb = scarlet.display.img_to_rgb(
        images, norm=norm, channel_map=channel_map)
    model_rgb = scarlet.display.img_to_rgb(
        model_, norm=norm, channel_map=channel_map)
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q/2)
    residual_rgb = scarlet.display.img_to_rgb(
        residual, norm=norm, channel_map=channel_map)
    vmax = np.max(np.abs(residual_rgb))

    if show_mask and aggr_mask is None:
        ax[1].imshow(model_rgb * (~np.tile(mask.T, (3, 1, 1))).T)
    elif show_mask and aggr_mask is not None:
        ax[1].imshow(model_rgb * (~np.tile((aggr_mask | mask).T, (3, 1, 1))).T)
    elif show_gray_mask and aggr_mask is None:
        ax[1].imshow(model_rgb)
        ax[1].imshow(mask.astype(float), origin='lower',
                     alpha=0.15, cmap='Greys_r')
    elif show_gray_mask and aggr_mask is not None:
        ax[1].imshow(model_rgb)
        ax[1].imshow((aggr_mask | mask).astype(float), origin='lower',
                     alpha=0.15, cmap='Greys_r')
    else:
        ax[1].imshow(model_rgb)

    ax[1].set_title("Model")

    ########## Figure 3 ###########
    # In Figure 3, we show the residual when removing the model of target galaxy (by setting `show_ind`) with aggressive mask
    if aggr_mask is None:
        ax[2].imshow(residual_rgb * (~np.tile(mask.T, (3, 1, 1))).T,
                     vmin=-vmax, vmax=vmax, cmap='seismic')
        ax[2].set_title("Residual")
    else:
        ax[2].imshow(residual_rgb * (~np.tile((aggr_mask | mask).T, (3, 1, 1))).T,
                     vmin=-vmax, vmax=vmax, cmap='seismic')
        ax[2].set_title("Residual")

    (img_size_x, img_size_y) = images[0].shape
    if scale_bar:
        if scale_bar_length / pixel_scale > 0.7 * min(img_size_x, img_size_y):
            # if scale_bar_length is too large
            scale_bar_length = 0.3 * min(img_size_x, img_size_y) * pixel_scale

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

    if show_loss:
        ax[3].plot(-np.array(loss))
        ax[3].set_xlabel('Iteration', labelpad=-40)
        # ax[3].set_ylabel('log-Likelihood')
        ax[3].set_title("log-Likelihood")
        xlim, ylim = ax[3].axes.get_xlim(), ax[3].axes.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        ax[3].set_aspect((xrange / yrange), adjustable='box')
        ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    from matplotlib.ticker import MaxNLocator, NullFormatter
    for axx in ax:
        axx.yaxis.set_major_locator(MaxNLocator(5))
        axx.xaxis.set_major_locator(MaxNLocator(5))

    # Show the size of PSF. FWHM is 1 arcsec.
    from matplotlib.patches import Circle
    circ1 = Circle((40, 40),
                   radius=1 / 0.168,
                   linewidth=1.,
                   hatch='/////',
                   fc='None',
                   ec='white')
    ax[1].add_patch(circ1)

    plt.subplots_adjust(wspace=0.2)

    if ax is None:
        return fig
    return ax


def get_extent(bbox, new_cen=None):
    if new_cen is not None:
        return [bbox.start[-1] - new_cen[0], bbox.stop[-1] - new_cen[0], bbox.start[-2] - new_cen[1], bbox.stop[-2] - new_cen[1]]
    else:
        return [bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]]


def display_pymfit_model(blend, mod_params, mask_fn=None, cmap=plt.cm.gray_r, colorbar=False,
                         save_fn=None, show=True, band=None, subplots=None,
                         titles=True, pixscale=0.168, psf_fn=None, zpt=27.,
                         fontsize=20, **kwargs):
    """
    Show imfit results: HSC image, scarlet model, Sersic model, and residual. Modified based on `pymfit`.
    """
    from pymfit import Sersic

    from .utils import img_cutout
    zscale = ZScaleInterval()

    observation = blend.observations[0]
    src = blend.sources[0]

    hsc_img = observation.images.mean(axis=0)
    w = blend.frame.wcs
    scarlet_model = src.get_model().mean(axis=0)
    hsc_img = img_cutout(hsc_img, w, src.center[1], src.center[0],
                         size=src.bbox.shape[1:],
                         pixel_unit=True, save=False, img_header=None)
    hsc_img = hsc_img[0].data

    if subplots is None:
        subplot_kw = dict(xticks=[], yticks=[])
        fig, axes = plt.subplots(1, 5, subplot_kw=subplot_kw, **kwargs)
        fig.subplots_adjust(wspace=0.08)
    else:
        fig, axes = subplots

    s = Sersic(mod_params, pixscale=pixscale, zpt=zpt)
    sersic_model = s.array(scarlet_model.shape)

    if psf_fn is not None:
        psf = fits.getdata(psf_fn)
        psf /= psf.sum()
        sersic_model = convolve(sersic_model, psf)

    res1 = hsc_img - scarlet_model
    res2 = scarlet_model - sersic_model

    vmin, vmax = zscale.get_limits(hsc_img)

    param_labels = {}

    if titles:
        titles = ['HSC image', 'Scarlet', 'Sersic',
                  'HSC - Scarlet', 'Scarlet - Sersic']
    else:
        titles = ['']*5

    for i, data in enumerate([hsc_img, scarlet_model, sersic_model, res1, res2]):
        show = axes[i].imshow(data, vmin=vmin, vmax=vmax, origin='lower',
                              cmap=cmap, aspect='equal', rasterized=True)
        axes[i].set_title(titles[i], fontsize=fontsize + 4, y=1.01)

    if mask_fn is not None:
        mask = fits.getdata(mask_fn)
        mask = mask.astype(float)
        mask[mask == 0.0] = np.nan
        axes[0].imshow(mask, origin='lower', alpha=0.4,
                       vmin=0, vmax=1, cmap='rainbow_r')

    x = 0.05
    y = 0.93
    dy = 0.09
    dx = 0.61
    fs = fontsize
    if band is not None:
        m_tot = r'$m_'+band+' = '+str(round(s.m_tot, 1))+'$'
    else:
        m_tot = r'$m = '+str(round(s.m_tot, 1))+'$'
    r_e = r'$r_\mathrm{eff}='+str(round(s.r_e*pixscale, 1))+'^{\prime\prime}$'
    mu_0 = r'$\mu_0='+str(round(s.mu_0, 1))+'$'
    mu_e = r'$\mu_e='+str(round(s.mu_e, 1))+'$'
    n = r'$n = '+str(round(s.n, 2))+'$'

    c = 'b'

    axes[2].text(x, y, m_tot, transform=axes[2].transAxes,
                 fontsize=fs, color=c)
    axes[2].text(x, y-dy, mu_0, transform=axes[2].transAxes,
                 fontsize=fs, color=c)
    axes[2].text(x, y-2*dy, mu_e, transform=axes[2].transAxes,
                 fontsize=fs, color=c)
    axes[2].text(x+dx, y, n, transform=axes[2].transAxes,
                 fontsize=fs, color=c)
    axes[2].text(x+dx, y-dy, r_e, transform=axes[2].transAxes,
                 fontsize=fs, color=c)
    if band is not None:
        axes[2].text(0.9, 0.05, band, color='r', transform=axes[2].transAxes,
                     fontsize=25)
    if 'reduced_chisq' in list(mod_params.keys()):
        chisq = r'$\chi^2_\mathrm{dof} = ' +\
                str(round(mod_params['reduced_chisq'], 2))+'$'
        axes[2].text(x+dx, y-2*dy, chisq, transform=axes[2].transAxes,
                     fontsize=fs, color=c)

    # Put a color bar on the image
    if colorbar:
        ax_cbar = inset_axes(axes[4],
                             width='75%',
                             height='5%',
                             loc=1)
        cbar = plt.colorbar(show, ax=axes[4], cax=ax_cbar,
                            orientation='horizontal')

        cbar.ax.xaxis.set_tick_params(color='w')
        cbar.ax.yaxis.set_tick_params(color='w')
        cbar.outline.set_edgecolor('w')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                 color='w', fontsize=18)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                 color='w', fontsize=18)

    if save_fn is not None:
        dpi = kwargs.pop('dpi', 200)
        fig.savefig(save_fn, bbox_inches='tight', dpi=dpi)

    if not show:
        plt.close()


def plot_measurement(lsbg_cat, meas_cat):
    junk = (lsbg_cat['bad_votes'] > lsbg_cat['good_votes'])
    candy = (lsbg_cat['good_votes'] > lsbg_cat['bad_votes']) & (
        lsbg_cat['is_candy'] > lsbg_cat['is_galaxy'])
    gal = (~junk) & (~candy)
    print('# of Candy:', np.sum(candy))
    print('# of Gal:', np.sum(gal))
    print('# of Junk:', np.sum(junk))

    g_mag = meas_cat['mag'].data[:, 0]
    r_mag = meas_cat['mag'].data[:, 1]
    i_mag = meas_cat['mag'].data[:, 2]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    plt.sca(axes[0, 0])  # color-color figure
    plt.scatter((g_mag - i_mag)[candy], (g_mag - r_mag)
                [candy], color='g', zorder=10, label='Candy')
    plt.scatter((g_mag - i_mag)[junk], (g_mag - r_mag)
                [junk], color='r', label='junk', zorder=10)
    plt.scatter((g_mag - i_mag)[gal], (g_mag - r_mag)
                [gal], color='b', label='gal')
    color_bound = [0.1, 1.2]
    half_width = 0.25
    plt.vlines(color_bound[0], 0.7 * color_bound[0] - half_width,
               0.7 * color_bound[0] + half_width, color='k', ls='--', lw=2, zorder=12)
    plt.vlines(color_bound[1], 0.7 * color_bound[1] - half_width,
               0.7 * color_bound[1] + half_width, color='k', ls='--', lw=2, zorder=12)
    x = np.linspace(color_bound[0], color_bound[1], 100)
    plt.plot(x, 0.7 * x - half_width, color='k', ls='--', lw=2, zorder=12)
    plt.plot(x, 0.7 * x + half_width, color='k', ls='--', lw=2, zorder=12)
    plt.xlabel('g-i')
    plt.ylabel('g-r')
    plt.legend()
    plt.xlim(-0.5, 2.2)
    plt.ylim(-0.5, 2.2)

    plt.sca(axes[0, 1])  # SB-color figure
    plt.scatter((g_mag - i_mag)[candy], meas_cat['SB_0']
                [:, 0][candy], color='g', zorder=10)
    plt.scatter((g_mag - i_mag)[junk], meas_cat['SB_0'][:, 0][junk], color='r')
    plt.scatter((g_mag - i_mag)[gal], meas_cat['SB_0'][:, 0][gal], color='b')
    plt.xlabel('g-i')
    plt.ylabel('SB_0_g')
    plt.axhline(22., color='k', ls='--')
    plt.axhline(23.5, color='k', ls='--')
    plt.axvline(color_bound[0], color='k', ls='--', lw=2)
    plt.axvline(color_bound[1], color='k', ls='--', lw=2)
    plt.xlim(-0.7, 2.9)
    plt.ylim(17.9, 28.5)

    plt.sca(axes[0, 2])  # SB-color figure
    plt.scatter((g_mag - i_mag)[candy], meas_cat['SB_eff_avg']
                [:, 0][candy], color='g', zorder=10)
    plt.scatter((g_mag - i_mag)[junk],
                meas_cat['SB_eff_avg'][:, 0][junk], color='r', zorder=11)
    plt.scatter((g_mag - i_mag)[gal],
                meas_cat['SB_eff_avg'][:, 0][gal], color='b')
    plt.xlabel('g-i')
    plt.ylabel('SB_eff_g')
    plt.axhline(23.0, color='k', ls='--')
    plt.axhline(24.3, color='k', ls='--')
    plt.axvline(color_bound[0], color='k', ls='--', lw=2)
    plt.axvline(color_bound[1], color='k', ls='--', lw=2)
    plt.xlim(-0.7, 2.9)
    plt.ylim(19, 29)

    plt.sca(axes[0, 3])
    plt.scatter(meas_cat['M20'][candy], meas_cat['Gini']
                [candy], color='g', zorder=10)
    plt.scatter(meas_cat['M20'][junk], meas_cat['Gini']
                [junk], color='r', zorder=11)
    plt.scatter(meas_cat['M20'][gal], meas_cat['Gini'][gal], color='b')
    plt.xlabel('M20')
    plt.ylabel('Gini')
    plt.xlim(-3, 0)
    plt.ylim(0.2, 0.9)
    x = np.linspace(-3, 0, 10)
    plt.plot(x, -0.14 * x + 0.33, color='orange')
    x = np.linspace(-3, -1.6, 10)
    plt.plot(x, 0.136 * x + 0.788, color='orange')

    plt.sca(axes[1, 0])
    plt.scatter(meas_cat['C'][candy], meas_cat['A']
                [candy], color='g', zorder=10)
    plt.scatter(meas_cat['C'][junk], meas_cat['A'][junk], color='r', zorder=11)
    plt.scatter(meas_cat['C'][gal], meas_cat['A'][gal], color='b')
    plt.xlim(0.7, 4.8)
    plt.ylim(-.2, 1.2)
    plt.axvline(3.3)
    plt.axhline(0.1)
    plt.xlabel('C')
    plt.ylabel('A')

    plt.sca(axes[1, 1])  # SB-R_e figure
    plt.scatter(meas_cat['SB_0'][:, 0][candy],
                meas_cat['rhalf_circularized'][candy], color='g', zorder=10)
    plt.scatter(meas_cat['SB_0'][:, 0][junk],
                meas_cat['rhalf_circularized'][junk], color='r', zorder=11)
    plt.scatter(meas_cat['SB_0'][:, 0][~junk],
                meas_cat['rhalf_circularized'][~junk], color='b')
    plt.axvline(22., color='k', ls='--')
    plt.axhline(1.8 / 0.168, color='k', ls='--')
    plt.axhline(10 / 0.168, color='k', ls='--')
    plt.yscale('log')
    plt.xlabel('SB_0_g')
    plt.ylabel('R_e (pix)')
    plt.xlim(18, 28)

    plt.sca(axes[1, 2])  # SB-R_e figure
    plt.scatter(meas_cat['SB_eff_avg'][:, 0][candy],
                meas_cat['rhalf_circularized'][candy], color='g', zorder=10)
    plt.scatter(meas_cat['SB_eff_avg'][:, 0][junk],
                meas_cat['rhalf_circularized'][junk], color='r', zorder=11)
    plt.scatter(meas_cat['SB_eff_avg'][:, 0][~junk],
                meas_cat['rhalf_circularized'][~junk], color='b')
    plt.axvline(23., color='k', ls='--')
    plt.axhline(1.8 / 0.168, color='k', ls='--')
    plt.axhline(10 / 0.168, color='k', ls='--')
    plt.yscale('log')
    plt.xlabel('SB_eff_g')
    plt.ylabel('R_e (pix)')
    plt.xlim(19, 29)

    plt.sca(axes[1, 3])
    plt.scatter(meas_cat['rhalf_circularized'][candy],
                meas_cat['sersic_rhalf'][candy], color='g', zorder=10)
    plt.scatter(meas_cat['rhalf_circularized'][junk],
                meas_cat['sersic_rhalf'][junk], color='r', zorder=11)
    plt.scatter(meas_cat['rhalf_circularized'][gal],
                meas_cat['sersic_rhalf'][gal], color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('R_e')
    plt.ylabel('R_e (Sersic)')

    plt.subplots_adjust(wspace=0.3, hspace=0.2)
