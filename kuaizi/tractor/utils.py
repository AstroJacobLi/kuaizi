import numpy as np

from astropy import wcs
from astropy.table import Table, Column
from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

import kuaizi
import os
import sys
import types

from kuaizi.detection import makeCatalog


class HiddenPrints:
    """
    Hide the print statements from the console.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        pass


def _isLegal_shape(src):
    """
    Check if the given source is within physical limits.
    See `_set_bounds` on how to use it.
    """
    return (src.shape.getAllParams() <= src.shape.uppers) & (
        src.shape.getAllParams() >= src.shape.lowers)


def _isLegal_flux(src):
    """
    Check if the given source is within physical limits.
    See `_set_bounds` on how to use it.
    """
    return (src.brightness.getAllParams()[0] <= src.brightness.upper) & (
        src.brightness.getAllParams()[0] >= src.brightness.lower)


def _getLogPrior(src):
    """
    LogPrior for sources.
    See `_set_bounds` on how to use it.
    """
    if src.getSourceType() == 'PointSource':
        if not src.brightness.isLegal():
            return -np.inf
        else:
            return 0
    else:
        if not (src.brightness.isLegal() & src.shape.isLegal()):
            return -np.inf
        else:
            return 0


def _set_bounds(sources):
    """
    The big problem with default `tractor` is that sources can have
    legit negative fluxes, negative radius, etc. This function manually
    writes the upper and lower limit for parameters (flux, R_e, axis ratio).
    The prior is assumed to be a tophat within upper and lower limit.
    We implement this by overwriting the `isLegal` function in `tractor.utils`
    and the `getLogPrior` function in `tractor.utils.BaseParams`.
    """
    for src in sources:
        src.brightness.lower = 0  # no negative flux
        src.brightness.upper = 1e9
        src.brightness.isLegal = types.MethodType(_isLegal_flux, src)
        src.brightness.getLogPrior = types.MethodType(_getLogPrior, src)
        if src.getSourceType() != 'PointSource':
            src.shape.lowers = [0, 0, -1e9]  # no negative R_e, axis ratio
            # set a large enough number to be upper limit
            src.shape.uppers = [1e9, 1e9, 1e9]
            src.shape.isLegal = types.MethodType(_isLegal_shape, src)
    return sources


def _freeze_source(src, freeze_dict):
    """
    Freeze parameters of ONE source according to `freeze_dict`.

    Parameters
    ----------
    src : tractor source.
    freeze_dict : dict, e.g.,
        ```
        freeze_dict = {'pos': True, 'shape': True, 'shape.re': True,
        'shape.ab': True, 'shape.phi': True, 'sersicindex': True}
        ```
    """
    for item in freeze_dict:
        if item == 'shape.ab' and freeze_dict[item]:
            try:
                if src.getNamedParamIndex('shape') is not None and src.shape.re >= 0:
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('ab')
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('e1')
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('e2')
            except:
                pass
        if item == 'shape.phi' and freeze_dict[item]:
            try:
                if src.getNamedParamIndex('shape') is not None and src.shape.re >= 0:
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('phi')
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('e1')
                    src[src.getNamedParamIndex(
                        'shape')].freezeParam('e2')
            except:
                pass
        if item == 'shape.re' and freeze_dict[item]:
            if src.getNamedParamIndex('shape') is not None and src.shape.re >= 0:
                src[src.getNamedParamIndex(
                    'shape')].freezeParam('re')
        if item == 'shape' and freeze_dict[item] and item in src.namedparams:
            src.freezeParam(item)

        if item == 'pos' and freeze_dict[item] and item in src.namedparams:
            src.pos.addGaussianPrior('x', src.pos.x, 1)
            src.pos.addGaussianPrior('y', src.pos.y, 1)

        if item in ['sersicindex'] and item in src.namedparams:
            if freeze_dict[item] is True:
                src.freezeParam(item)
    return src


def _freeze_params(sources, freeze_dict, cen_ind=None, fix_all=False):
    """
    Freeze parameters of sources in `sources` according to `freeze_dict`.

    Parameters
    ----------
    sources : list of tractor sources.
    freeze_dict : dict, e.g.,
        ```
        freeze_dict = {'pos': True, 'shape': True, 'shape.re': True,
        'shape.ab': True, 'shape.phi': True, 'sersicindex': True}
        ```
    cen_ind : int, optional. The index of target object in `sources`.
        If provided, only the central object will be frozen.
    fix_all : bool, optional. If True, all sources are frozen according to `freeze_dict`.
    """
    if cen_ind is not None:
        sources[cen_ind] = _freeze_source(sources[cen_ind], freeze_dict)

    if fix_all:
        for i in range(len(sources)):
            sources[i] = _freeze_source(sources[i], freeze_dict)
    return sources


def _compute_invvars(allderivs):
    """
    Compute the inverse-variance of parameters.
    This is used to estimate the error of tractor fitting.
    """
    ivs = []
    for derivs in allderivs:
        chisq = 0
        for deriv, tim in derivs:
            h, w = tim.shape
            deriv.clipTo(w, h)
            ie = tim.getInvError()
            slc = deriv.getSlice(ie)
            chi = deriv.patch * ie[slc]
            chisq += (chi**2).sum()
        ivs.append(chisq)
    return ivs


def _regularize_attr(item):
    """
    Change attribute name to be consistent with output catalog.
    """
    item = item.replace('pos.x', 'x')
    item = item.replace('pos.y', 'y')
    item = item.replace('brightness.Flux', 'flux')
    item = item.replace('shape.re', 're')
    item = item.replace('shape.ab', 'ab')
    item = item.replace('shape.phi', 'phi')
    item = item.replace('sersicindex.SersicIndex', 'sersic')
    return item


def getTargetProperty(trac_obj, wcs=None, pixel_scale=kuaizi.HSC_pixel_scale, zeropoint=kuaizi.HSC_zeropoint):
    '''
    Write the properties of our target galaxy in a certain band into a dictionary.

    Paarameters
    ----------
    trac_obj: `Tractor` object, including the target galaxy
    wcs: wcs object of the input image.
    pixel_scale: pixel scale of the input image.
    zeropoint: zeropoint of the input image.

    Returns
    -------
    source_output (dict): contains many attributes of the target galaxy in `trac_obj`.
        source_output (dict): contains many attributes of the target galaxy in `trac_obj`.
    source_output (dict): contains many attributes of the target galaxy in `trac_obj`.
        Flux is in nanomaggies. (ZP=22.5),
        effective radius (`re`) is in arcsec.
    '''
    if trac_obj is None:
        return None

    trac_obj.thawAllRecursive()

    attri = trac_obj.catalog.getParamNames()
    attri = [_regularize_attr(item) for item in attri]

    values = dict(zip(attri, trac_obj.catalog.getParams()))
    with HiddenPrints():
        all_derivs = trac_obj.getDerivs()
        # first derivative is for sky bkg
        invvars = dict(zip(attri, _compute_invvars(all_derivs)[1:]))

    i = trac_obj.target_ind  # only extract information of our target galaxy
    src = trac_obj.catalog[i]

    keys_to_extract = [item for item in attri if f'source{i}.' in item]
    source_values = {key.replace(f'source{i}.', '')                     : values[key] for key in keys_to_extract}
    source_values['flux'] *= 10**((22.5 - zeropoint) / 2.5)  # in nanomaggy

    source_invvar = {key.replace(
        f'source{i}.', '') + '_ivar': invvars[key] for key in keys_to_extract}
    # in nanomaggy^-2
    source_invvar['flux_ivar'] *= 10**(- 2 * (22.5 - zeropoint) / 2.5)

    source_output = dict(**source_values, **source_invvar)

    if not 'sersic' in source_output.keys():  # doesn't have sersic index, need to assign according to its type
        if 'dev' in src.getSourceType().lower():
            source_output['sersic'] = 4.0
            source_output['sersic_ivar'] = 0.0
            source_output['type'] = 'DEV'

        if 'exp' in src.getSourceType().lower():
            source_output['sersic'] = 1.0
            source_output['sersic_ivar'] = 0.0
            rex_flag = (src.shape.getName() ==
                        'EllipseE' and src.shape.e1 == 0 and src.shape.e2 == 0)
            rex_flag |= (src.shape.getName() ==
                         'Galaxy Shape' and src.shape.ab == 1)

            if rex_flag:
                # Round exponential
                source_output['ab'] = 1.0
                source_output['phi'] = 0.0
                source_output['ab_ivar'] = 0.0
                source_output['phi_ivar'] = 0.0
                source_output['type'] = 'REX'
            else:
                source_output['type'] = 'EXP'

        if 'pointsource' in src.getSourceType().lower():
            source_output['sersic'] = 0.0
            source_output['sersic_ivar'] = 0.0
            source_output['type'] = 'PSF'
            source_output['ab'] = 1.0
            source_output['phi'] = 0.0
            source_output['ab_ivar'] = 0.0
            source_output['phi_ivar'] = 0.0
            source_output['re'] = 0.0
            source_output['re_ivar'] = 0.0
    else:
        source_output['type'] = 'SER'

    ## RA, DEC ##
    if wcs is not None:
        ra, dec = wcs.wcs_pix2world(source_output['x'], source_output['y'], 0)
        # Here we assume the WCS is regular and has no distortion!
        dec_ivar = source_output['y_ivar'] / (pixel_scale / 3600)**2
        ra_ivar = source_output['x_ivar'] / \
            (pixel_scale / 3600)**2 / np.cos(np.deg2rad(dec))
        source_output['ra'] = float(ra)
        source_output['dec'] = float(dec)
        source_output['ra_ivar'] = float(ra_ivar)
        source_output['dec_ivar'] = float(dec_ivar)

    # R_e is already in arcsec
    if 're' not in source_output.keys():
        source_output['re'] = 0
        source_output['re_ivar'] = 0

    return source_output


def _write_to_row(row, model_dict, channels=list('grizy') + ['N708', 'N540']):
    '''
    Write the output of `getTargetProperty` to a Row of astropy.table.Table.

    Parameters
    ----------
    row (astropy.table.Row): one row of the table.
    measurement (dict): the output dictionary of `kuaizi.tractor.utils.getTargetProperty`

    Returns
    -------
    row (astropy.table.Row)
    '''
    if 'i' in model_dict.keys() and model_dict['i'] is not None:
        # Positions from i-band
        meas_dict = getTargetProperty(model_dict['i'], wcs=model_dict['i'].wcs)
        for key in ['ra', 'ra_ivar', 'dec', 'dec_ivar', 're', 're_ivar', 'ab', 'ab_ivar', 'phi', 'phi_ivar', 'sersic', 'sersic_ivar']:
            row[key] = meas_dict[key]
        # flux
        flux = np.zeros(len(channels))
        flux_ivar = np.zeros(len(channels))
        for i, filt in enumerate(channels):
            meas_dict = getTargetProperty(model_dict[filt])
            if meas_dict is not None:
                flux[i] = meas_dict['flux']
                flux_ivar[i] = meas_dict['flux_ivar']
        row['flux'] = flux
        row['flux_ivar'] = flux_ivar
        return row
    else:
        return row


def initialize_meas_cat(obj_cat, channels=list('grizy') + ['N708', 'N540']):
    """
    Initialize an empty measurement catalog filled by zeros.
    """
    length = len(obj_cat)
    bands = len(channels)

    meas_cat = Table([
        Column(name='ID', length=length, dtype=int),
        Column(name='ra', length=length, dtype=float),
        Column(name='ra_ivar', length=length, dtype=float),
        Column(name='dec', length=length, dtype=float),
        Column(name='dec_ivar', length=length, dtype=float),
        Column(name='flux', length=length, shape=(bands,)),
        Column(name='flux_ivar', length=length, shape=(bands,)),
        Column(name='re', length=length, dtype=float),
        Column(name='re_ivar', length=length, dtype=float),
        Column(name='ab', length=length, dtype=float),
        Column(name='ab_ivar', length=length, dtype=float),
        Column(name='phi', length=length, dtype=float),
        Column(name='phi_ivar', length=length, dtype=float),
        Column(name='sersic', length=length, dtype=float),
        Column(name='sersic_ivar', length=length, dtype=float),
    ])

    return meas_cat
