import fire
import os
import kuaizi as kz
from kuaizi.fitting import fitting_wavelet_obs_tigress

from astropy.io import fits
from astropy.table import Table

lsbg_cat = Table.read('/tigress/jiaxuanl/Data/HSC/LSBG/Cutout/Candy/candy_cutout_cat.fits')

def run_scarlet_wvlt(index):
    blend = fitting_wavelet_obs_tigress(
        {'project': 'HSC', 'name': 'LSBG', 'data_dir': '/tigress/jiaxuanl/Data'},
        lsbg_cat[index],
        name='Seq',
        channels='griz',
        starlet_thresh=1,
        prefix='candy',
        show_figure=False)
    return


if __name__ == '__main__':
    fire.Fire(run_scarlet_wvlt)
