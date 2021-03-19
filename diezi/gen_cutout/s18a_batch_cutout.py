#!/usr/bin/env python

import fire
import os
import lsst.daf.persistence as dafPersist
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import kuaizi as kz
from astropy.table import Table
import astropy.units as u
from joblib import Parallel, delayed

from tiger_cutout_hsc import cutout_one, prepare_catalog  # generate_cutout, get_psf
import lsst.log
Log = lsst.log.Log()
Log.setLevel(lsst.log.ERROR)

# everything should be downloaded to /tigress/jiaxuanl/Data/HSC/LSBG/Cutout

kz.utils.set_env(project='HSC', name='LSBG',
                 data_dir='/tigress/jiaxuanl/Data/')

DATA_ROOT = '/tigress/HSC/DR/s18a_wide'
PIXEL_SCALE = 0.168  # arcsec / pixel


# Cutout speed is about 18s for one galaxy in 5 bands
def batch_cutout(obj_cat_dir, size=1.0, unit='arcmin', bands='grizy',
                 ra_name='ra', dec_name='dec',
                 name='Seq', prefix='s18a_wide', output='./Cutout',
                 label='deepCoadd_calexp', root=DATA_ROOT,
                 njobs=1, psf=True):
    '''
    Generate cutout for objects in a given catalog in multiple bands. 
    This function only runs on Tiger or Tigressdata. 
    Be sure to load proper LSST Pipeline environment, 
    see https://github.com/AstroJacobLi/diezi/blob/main/setup_env.sh.

    Parameters:
        obj_cat_dir (str): the directory of object catalogs.
        size (float): cutout size.
        unit (str): unit of cutout size, use [arcsec, arcmin, degree, pixel]。
        bands (str): filters name, only support [g, r, i, z, y].
        ra_name (str): the column name of RA in the input catalog.
        dec_name (str): the column name of DEC in the input catalog.
        name (str): the column name for galaxy name or index.
        prefix (str): prefix of the output file, such as 'lsbg' or 's18a'.
        output (str): output directory. The folder will be made if not existed. 
        label (str): the dataset type of HSC, only support [deepCoadd, deepCoadd_calexp].
        root (str): directory of HSC data on Tigress.
        njobs (int): number of threads running at the same time. 
            Based on my scaling analysis, `njobs = 2` has relatively high parallel efficiency. 
        psf (bool): whether retrieve PSF.

    Returns:
        None

    Notes:
        The files will be saved in FITS format. 
    '''
    t0 = perf_counter()

    butler = dafPersist.Butler(root)
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    print('\n Number of jobs:', njobs)

    obj_cat = Table.read(obj_cat_dir)[126:]
    print('\n Number of galaxies:', len(obj_cat))

    if label.strip() not in ['deepCoadd', 'deepCoadd_calexp']:
        raise ValueError(
            "Wrong coadd type. Only [deepCoadd, deepCoadd_calexp] are available.")

    if unit.strip() not in ['arcsec', 'arcmin', 'degree', 'pixel']:
        raise ValueError(
            "Wrong size unit. Please use [arcsec, arcmin, degree, pixel].")

    for filt in bands:
        if filt.lower() not in ['g', 'r', 'i', 'z', 'y']:
            raise ValueError(
                "Wrong filter name. Only [g, r, i, z, y] are available.")

        # Get cutout in each band
        cat = prepare_catalog(
            obj_cat, size, ra=ra_name, dec=dec_name,
            name=name, unit=unit, prefix=prefix, output=output)

        print(
            f'    - Generate cutouts for {len(cat)} galaxies in {filt}-band.')

        if njobs <= 1:
            _ = [cutout_one(butler, skymap, obj, filt, label, psf)
                 for obj in cat]
        else:
            Parallel(n_jobs=njobs)(delayed(cutout_one)(
                butler, skymap, obj, filt, label, psf) for obj in cat)

    print(f'Elapsed time: {(perf_counter() - t0):.2f} s')

    # Also save the catalog. Need to check if files exist.
    if prefix is None:
        prefix = 's18a_wide'

    image_flag = []
    for obj in cat:
        image_flag.append(
            [os.path.isfile(f"{obj['prefix']}_{filt}.fits") for filt in bands])
    cat['image_flag'] = image_flag

    psf_flag = []
    for obj in cat:
        psf_flag.append(
            [os.path.isfile(f"{obj['prefix']}_{filt}_psf.fits") for filt in bands])
    cat['psf_flag'] = psf_flag

    cat.write(os.path.join(
        output, f'{prefix}_cutout_cat.fits'), format='fits', overwrite=True)


if __name__ == '__main__':
    fire.Fire(batch_cutout)
