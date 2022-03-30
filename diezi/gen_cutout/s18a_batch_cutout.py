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

DATA_ROOT = '/tigress/HSC/DR/s18a_wide'
PIXEL_SCALE = 0.168  # arcsec / pixel


# Cutout speed is about 18s for one galaxy in 5 bands
def batch_cutout(data_dir, obj_cat_dir, low=0, high=None,
                 size=1.0, unit='arcmin', bands='grizy',
                 ra_name='ra', dec_name='dec',
                 name='Seq', prefix='s18a_wide', output='./Cutout',
                 catalog_dir='./Catalog', catalog_suffix='',
                 label='deepCoadd_calexp', root=DATA_ROOT, overwrite=False,
                 njobs=1, psf=True):
    '''
    Generate cutout for objects in a given catalog in multiple bands. 
    This function only runs on Tiger or Tigressdata. 
    Be sure to load proper LSST Pipeline environment, 
    see https://github.com/AstroJacobLi/diezi/blob/main/setup_env.sh.

    Parameters:
        data_dir (str): the directory for all Data, such as `/scrach/gpfs/jiaxuanl/Data`. 
            `/scratch/gpfs` is recommended! But don't download it directly to `/tigress/`!
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
    kz.utils.set_env(project='HSC', name='LSBG',
                     data_dir=data_dir)

    if not os.path.isdir(output):
        os.makedirs(output)
    if not os.path.isdir(catalog_dir):
        os.makedirs(catalog_dir)

    t0 = perf_counter()

    butler = dafPersist.Butler(root)
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    print('\n Number of jobs:', njobs)

    obj_cat = Table.read(obj_cat_dir)
    if high is None:
        high = len(obj_cat)
    if low is None:
        low = 0
    obj_cat = obj_cat[low:high]
    print('\n Number of galaxies:', len(obj_cat))

    # Adaptive cutout size# Normal objects, use 0.7 arcmin cutout.
    # Radius > 20 arcsec, use 1.0 arcmin cutout.
    # Radius > 30 arcsec, use 2.0 arcmin cutout.
    cutout_size = np.ones(len(obj_cat)) * size * u.arcmin

    cutout_size[obj_cat['flux_radius_ave_i'] >
                20] = 1.0 * u.arcmin  # shoud be larger
    cutout_size[obj_cat['flux_radius_ave_i'] >
                30] = 1.5 * u.arcmin  # should be larger

    obj_cat['cutout_size'] = cutout_size.value

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
            obj_cat, "cutout_size", ra=ra_name, dec=dec_name,
            name=name, unit=unit, prefix=prefix, output=output)

        cat.write(os.path.join(
            catalog_dir, f'{prefix}_cutout_cat_{catalog_suffix}.fits'),
            format='fits', overwrite=True)

        print(
            f'    - Generate cutouts for {len(cat)} galaxies in {filt}-band.')

        if njobs <= 1:
            _ = [cutout_one(butler, skymap, obj, filt, label, psf, overwrite=overwrite)
                 for obj in cat]
        else:
            Parallel(n_jobs=njobs)(delayed(cutout_one)(
                butler, skymap, obj, filt, label, psf, overwrite=overwrite) for obj in cat)

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
        catalog_dir, f'{prefix}_cutout_cat_{catalog_suffix}.fits'),
        format='fits', overwrite=True)


if __name__ == '__main__':
    fire.Fire(batch_cutout)
