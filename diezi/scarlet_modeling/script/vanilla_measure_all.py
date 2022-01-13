
import fire
import os
import numpy as np
import kuaizi as kz
from kuaizi.fitting import fitting_vanilla_obs_tigress

from multiprocessing import Pool, Manager
from functools import partial

from astropy.table import Table, Column

import dill
from kuaizi.measure import _write_to_row, makeMeasurement, initialize_meas_cat


def measure_vanilla(index, cat, meas_cat, model_dir, PREFIX,
                    global_logger=None):
    lsbg = cat[index]
    row = meas_cat[index]
    row['ID'] = lsbg['viz-id']

    try:
        with open(os.path.join(model_dir, f"{PREFIX.lower()}-{lsbg['viz-id']}-trained-model-vanilla.df"), "rb") as fp:
            blend, info, mask = dill.load(fp)
            fp.close()
    except Exception as e:
        print(f'IMCOMPLETE FILE FOR {PREFIX.lower()}-{index}', e)
        if global_logger is not None:
            global_logger.error(
                f'IMCOMPLETE FILE FOR {PREFIX.lower()}-{index}', e)

    # Measure!
    try:
        measurement, _ = makeMeasurement(list(np.array(blend.sources)[info['sed_ind']]),
                                         blend.observations[0],
                                         aggr_mask=mask.astype(bool),
                                         makesegmap=True, sigma=1,
                                         zeropoint=27.0, out_prefix=None,
                                         show_fig=False, asinh_a=0.02, framealpha=0.7)
        row = _write_to_row(row, measurement)

    except Exception as e:
        print(f'MEASUREMENT ERROR FOR {PREFIX.lower()}-{index}', e)
        if global_logger is not None:
            global_logger.error(
                f'MEASUREMENT ERROR FOR {PREFIX.lower()}-{index}', e)

    return meas_cat


def run_all(DATADIR, OUTPUT_DIR, PREFIX, cat_dir, low=0, high=None,
            filename='_lsbg_measure_vanilla.fits', suffix=''):
    print('SET ENVIRONMENT')
    kz.utils.set_env(project='HSC', name='LSBG', data_dir=DATADIR)
    print("CURRENT WORKING DIRECTORY:", os.getcwd())

    global_logger = kz.utils.set_logger(
        logger_name=f'{PREFIX.lower()}_measure_sample' + suffix,
        file_name=f'{PREFIX.lower()}_measure_log' + suffix, level='INFO')

    lsbg_cat = Table.read(cat_dir)

    meas_cat = initialize_meas_cat(lsbg_cat)

    if high is None:
        high = len(lsbg_cat)

    for ind in range(low, high):
        measure_vanilla(
            ind, lsbg_cat, meas_cat, PREFIX=PREFIX,
            model_dir=os.path.join(
                OUTPUT_DIR, f'Model/{PREFIX.lower()}/'),
            global_logger=global_logger)

    meas_cat.write(os.path.join(OUTPUT_DIR, filename.rstrip(
        '.fits') + suffix + '.fits'), overwrite=True)

    print('Catalog written to:', os.path.join(OUTPUT_DIR, f'Catalog/{PREFIX.lower()}', filename.rstrip(
        '.fits') + suffix + '.fits'))


if __name__ == '__main__':
    fire.Fire(run_all)
