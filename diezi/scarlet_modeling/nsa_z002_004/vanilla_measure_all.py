"""
This file measures the structural parameters for all NSA galaxies at z=0.02 and z=0.04.
"""

import fire
import os
import numpy as np
import kuaizi as kz
from kuaizi.fitting import fitting_vanilla_obs_tigress

from multiprocessing import Pool, Manager
from functools import partial

from astropy.table import Table, Column

import dill
from kuaizi.measure import _write_to_row, makeMeasurement

kz.utils.set_env(project='HSC', name='LSBG',
                 data_dir='/scratch/gpfs/jiaxuanl/Data')


def initialize_meas_cat(lsbg_cat):
    length = len(lsbg_cat)
    bands = 4

    meas_cat = Table([
        Column(name='ID', length=length, dtype=int),
        Column(name='flux', length=length, shape=(bands,)),
        Column(name='mag', length=length, shape=(bands,)),
        Column(name='SB_0', length=length, shape=(bands,)),
        Column(name='SB_eff_circ', length=length, shape=(bands,)),
        Column(name='SB_eff_ellip', length=length, shape=(bands,)),
        Column(name='xc_cen', length=length),
        Column(name='yc_cen', length=length),
        Column(name='xc_sym', length=length),
        Column(name='yc_sym', length=length),
        Column(name='ell_cen', length=length),
        Column(name='ell_sym', length=length),
        Column(name='PA_cen', length=length),
        Column(name='PA_sym', length=length),
        Column(name='rhalf_circ', length=length),
        Column(name='rhalf_ellip', length=length),
        Column(name='r20', length=length),
        Column(name='r50', length=length),
        Column(name='r80', length=length),
        Column(name='Gini', length=length),
        Column(name='M20', length=length),
        Column(name='F(G,M20)', length=length),
        Column(name='S(G,M20)', length=length),
        Column(name='C', length=length),
        Column(name='A', length=length),
        Column(name='A_outer', length=length),
        Column(name='A_shape', length=length),
        Column(name='S', length=length),
        Column(name='sersic_n', length=length),
        Column(name='sersic_rhalf', length=length),
        Column(name='sersic_ell', length=length),
        Column(name='sersic_PA', length=length),
        Column(name='sersic_xc', length=length),
        Column(name='sersic_yc', length=length),
        Column(name='sersic_amp', length=length),
        Column(name='flag', length=length, dtype=bool),
        Column(name='flag_sersic', length=length, dtype=bool),
    ])
    for col in meas_cat.columns:
        meas_cat[col] = np.nan * meas_cat[col]

    return meas_cat


def measure_vanilla(index, cat, meas_cat, model_dir='./Model/NSA/z002_004', global_logger=None):
    lsbg = cat[index]
    row = meas_cat[index]
    row['ID'] = lsbg['viz-id']

    try:
        with open(os.path.join(model_dir, f"nsa-{lsbg['viz-id']}-trained-model-vanilla.df"), "rb") as fp:
            blend, info, mask = dill.load(fp)
            fp.close()
    except Exception as e:
        print(f'IMCOMPLETE FILE FOR NSA-{index}', e)
        if global_logger is not None:
            global_logger.error(f'IMCOMPLETE FILE FOR NSA-{index}', e)

    # Measure!
    try:
        measurement, _ = makeMeasurement(list(np.array(blend.sources)[info['sed_ind']]),
                                         blend.observations[0],
                                         aggr_mask=mask.astype(bool),
                                         makesegmap=True, sigma=1,
                                         zeropoint=27.0, out_prefix=None,
                                         show_fig=False, asinh_a=0.02, framealpha=0.7)
        row = _write_to_row(row, measurement)
#         with open(f"./Measure/NSA/nsa-{lsbg['viz-id']}-wavelet.df", "wb") as fp:
#             dill.dump([measurement, morph], fp)
#             fp.close()
    except Exception as e:
        print(f'MEASUREMENT ERROR FOR NSA-{index}', e)
        if global_logger is not None:
            global_logger.error(f'MEASUREMENT ERROR FOR NSA-{index}', e)

    return meas_cat


def run_all(low=0, high=5476, lsbg_cat_dir='./Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits',
            filename='./Catalog/NSA/z002_004/_lsbg_measure_vanilla.fits', suffix=''):

    global_logger = kz.utils.set_logger(
        logger_name='nsa_measure_sample' + suffix, file_name='nsa_measure_log' + suffix, level='INFO')

    lsbg_cat = Table.read(lsbg_cat_dir)

    meas_cat = initialize_meas_cat(lsbg_cat)

    for ind in range(low, high):
        measure_vanilla(
            ind, lsbg_cat, meas_cat, model_dir='./Model/NSA/z002_004/', global_logger=global_logger)

    meas_cat.write(filename.rstrip('.fits') + suffix + '.fits', overwrite=True)


if __name__ == '__main__':
    fire.Fire(run_all)
