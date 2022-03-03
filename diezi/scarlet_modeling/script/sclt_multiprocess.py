import fire
import os
import numpy as np
import kuaizi as kz
from kuaizi.fit import fitting_obs_tigress

from multiprocessing import Pool, Manager
from functools import partial

from astropy.table import Table


def run_scarlet(index, cat, DATADIR, OUTPUT_DIR, PREFIX, OUTPUT_SUBDIR, method='vanilla',
                starlet_thresh=1, monotonic=True, bkg=True, variance=0.05**2, min_grad=0.01,
                scales=[0, 1, 2, 3, 4, 5],
                global_logger=None, fail_logger=None):
    print('STARTING FITTING for index:', index)
    blend = fitting_obs_tigress(
        {'project': 'HSC', 'name': 'LSBG', 'data_dir': DATADIR},
        cat[index],
        name='viz-id',
        method=method,
        channels='griz',
        starlet_thresh=starlet_thresh,
        monotonic=monotonic,
        bkg=bkg,
        min_grad=min_grad,
        variance=variance,
        scales=scales,
        prefix=PREFIX.lower(),
        figure_dir=os.path.join(
            OUTPUT_DIR, f'Figure/{OUTPUT_SUBDIR.lower()}/'),
        model_dir=os.path.join(OUTPUT_DIR, f'Model/{OUTPUT_SUBDIR.lower()}/'),
        log_dir=os.path.join(OUTPUT_DIR, f'log/{OUTPUT_SUBDIR.lower()}/'),
        show_figure=False,
        global_logger=global_logger,
        fail_logger=fail_logger)
    del blend
    return


def multiprocess_fitting(DATADIR, OUTPUT_DIR, OUTPUT_SUBDIR, PREFIX, njobs, cat_dir,
                         method='vanilla', ind_list=None, low=0, high=None, suffix='',
                         starlet_thresh=1, monotonic=True, bkg=True, min_grad=0.01,
                         variance=0.05**2, scales=[0, 1, 2, 3, 4, 5],):
    print('SET ENVIRONMENT')
    kz.utils.set_env(project='HSC', name='LSBG', data_dir=DATADIR)
    print("CURRENT WORKING DIRECTORY:", os.getcwd())

    print('Number of processor to use:', njobs)
    lsbg_cat = Table.read(cat_dir)
    # lsbg_cat.sort('viz-id')

    fail_logger = kz.utils.set_logger(
        logger_name=f'{PREFIX.lower()}_fail' + suffix, file_name=f'{PREFIX.lower()}_{low}_{high}_fail' + suffix, level='ERROR')
    if ind_list is not None:
        fail_logger = kz.utils.set_logger(
            logger_name=f'{PREFIX.lower()}_fail' + suffix, file_name=f'{PREFIX.lower()}_ind_list_fail' + suffix, level='ERROR')
    global_logger = kz.utils.set_logger(
        logger_name=f'{PREFIX.lower()}_sample' + suffix, file_name=f'{PREFIX.lower()}_{low}_{high}_log' + suffix, level='INFO')
    if ind_list is not None:
        global_logger = kz.utils.set_logger(
            logger_name=f'{PREFIX.lower()}_sample' + suffix, file_name=f'{PREFIX.lower()}_ind_list_log' + suffix, level='INFO')

    pool = Pool(njobs)

    if high is None:
        high = len(lsbg_cat)

    if ind_list is not None:
        iterable = ind_list
    else:
        iterable = np.arange(low, high, 1)

    pool.map(partial(run_scarlet, DATADIR=DATADIR, OUTPUT_DIR=OUTPUT_DIR,
                     OUTPUT_SUBDIR=OUTPUT_SUBDIR, PREFIX=PREFIX,
                     cat=lsbg_cat, method=method, starlet_thresh=starlet_thresh, min_grad=min_grad,
                     monotonic=monotonic, bkg=bkg, variance=variance, scales=scales,
                     global_logger=global_logger, fail_logger=fail_logger), iterable)
    pool.close()
    pool.join()


if __name__ == '__main__':
    fire.Fire(multiprocess_fitting)
