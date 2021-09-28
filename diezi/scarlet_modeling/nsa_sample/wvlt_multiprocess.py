import fire
import os
import numpy as np
import kuaizi as kz
from kuaizi.fitting import fitting_wavelet_obs_tigress

from multiprocessing import Pool, Manager
from functools import partial

from astropy.table import Table

kz.utils.set_env(project='HSC', name='LSBG',
                 data_dir='/scratch/gpfs/jiaxuanl/Data')

lsbg_cat = Table.read(
    '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/NSA/nsa_cutout_cat.fits')
lsbg_cat.sort('viz-id')


def run_scarlet_wvlt(index, starlet_thresh=0.5, global_logger=None, fail_logger=None):
    blend = fitting_wavelet_obs_tigress(
        {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/scratch/gpfs/jiaxuanl/Data'},
        lsbg_cat[index],
        name='viz-id',
        channels='griz',
        starlet_thresh=starlet_thresh,
        prefix='nsa',
        figure_dir='./Figure/NSA/',
        model_dir='./Model/NSA',
        show_figure=False,
        global_logger=global_logger,
        fail_logger=fail_logger)
    return


def multiprocess_fitting(njobs, ind_list=None, low=0, high=1, suffix='', starlet_thresh=0.5):
    print('Number of processor to use:', njobs)

    fail_logger = kz.utils.set_logger(
        logger_name='nsa_fail' + suffix, file_name='nsa_fail' + suffix, level='ERROR')
    global_logger = kz.utils.set_logger(
        logger_name='nsa_sample' + suffix, file_name='nsa_log' + suffix, level='INFO')

    pool = Pool(njobs)
    if ind_list is not None:
        iterable = ind_list
    else:
        iterable = np.arange(low, high, 1)

    pool.map(partial(run_scarlet_wvlt, starlet_thresh=starlet_thresh,
             global_logger=global_logger, fail_logger=fail_logger), iterable)
    pool.close()
    pool.join()


# python wvlt_multiprocess.py --njobs 6 --low 0 --high 100 --starlet_thresh 0.5

if __name__ == '__main__':
    fire.Fire(multiprocess_fitting)
