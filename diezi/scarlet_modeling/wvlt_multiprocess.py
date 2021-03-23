import fire
import os
import numpy as np
import kuaizi as kz
from kuaizi.fitting import fitting_wavelet_obs_tigress

from multiprocessing import Pool, Manager
from functools import partial

from astropy.io import fits
from astropy.table import Table

kz.utils.set_env(project='HSC', name='LSBG', data_dir='/tigress/jiaxuanl/Data')

lsbg_cat = Table.read('/tigress/jiaxuanl/Data/HSC/LSBG/Cutout/Candy/candy_cutout_cat.fits')

fail_logger = None #kz.utils.set_logger(logger_name='candy_fail', file_name='candy_fail', level='ERROR')
global_logger = None #kz.utils.set_logger(logger_name='candy_sample', file_name='candy_log', level='INFO')

def run_scarlet_wvlt(index, starlet_thresh=0.5):
    blend = fitting_wavelet_obs_tigress(
        {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/tigress/jiaxuanl/Data'}, 
        lsbg_cat[index],
        name='Seq',
        channels='griz',
        starlet_thresh=starlet_thresh,
        prefix='candy',
        show_figure=False, 
        global_logger=global_logger,
        fail_logger=fail_logger)
    return


def multiprocess_fitting(njobs, ind_list=None, low=0, high=10, starlet_thresh=0.5):
    print('Number of processor to use:', njobs)
    pool = Pool(njobs)
    if ind_list is not None:
        iterable = ind_list
    else:
        iterable = np.arange(low, high, 1)
        
    pool.map(run_scarlet_wvlt, iterable)
    pool.close()
    pool.join()

    
# python wvlt_multiprocess.py --njobs 6 --low 0 --high 100 --starlet_thresh 0.5
# 181, 182, 196, 197
if __name__ == '__main__':
    fire.Fire(multiprocess_fitting)
