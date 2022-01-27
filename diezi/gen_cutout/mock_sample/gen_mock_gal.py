import os
import numpy as np

import kuaizi as kz
from kuaizi.utils import padding_PSF
from kuaizi.detection import Data

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column

from kuaizi.mock import Data, MockGal
import galsim
from tqdm import trange
import fire

env_dict = {'project': 'HSC', 'name': 'LSBG',
            'data_dir': '/scratch/gpfs/jiaxuanl/Data'}
kz.utils.set_env(**env_dict)
kz.utils.set_matplotlib(style='default', usetex=False)

output_dir = './Cutout/mock_sample/'
param_cat = Table.read('./Catalog/mock_sample/mock_gal_param_truth.fits')

length = len(param_cat)
bands = 4
lsbg_cat = Table([Column(name='viz-id', length=length, dtype=int),
                  Column(name='ra', length=length),
                  Column(name='dec', length=length),
                  Column(name='mag_auto_i', length=length),
                  Column(name='sersic_n', length=length),
                  Column(name='sersic_ell', length=length),
                  Column(name='sersic_PA', length=length),
                  Column(name='sersic_rhalf_circ', length=length),
                  Column(name='sersic_sed', length=length, shape=(bands,)),
                  Column(name='mags', length=length, shape=(bands,)),
                  Column(name='prefix', length=length, dtype='S65'),
                  ])


def gen_mock_gal(low, high):
    for ind in trange(low, high):
        ind = ind
        bkg_id = ind

        param = param_cat[ind]

        #### Load bkg ####
        channels = 'griz'
        try:
            cutout = [fits.open(
                f"./Cutout/mock_sample/bkg/mockbkg_{bkg_id}_{band}.fits") for band in channels]
            psf_list = [fits.open(
                f"./Cutout/mock_sample/bkg/mockbkg_{bkg_id}_{band}_psf.fits") for band in channels]
            w = wcs.WCS(cutout[0][1].header)
            images = np.array([hdu[1].data for hdu in cutout])
            masks = np.array([hdu[2].data for hdu in cutout])
            variances = np.array([hdu[3].data for hdu in cutout])
            psf_pad = padding_PSF(psf_list)  # Padding PSF cutouts from HSC
            bkg = Data(images, variances, masks, channels, w, psfs=psf_pad)
        except Exception as e:
            print('Incomplete file for bkgid = {}'.format(bkg_id))
            print(e)
            continue

        #### Gen mock gal ####
        try:
            q = 1 - param['ellip']
            gmag = param['mag_g']
            gr = param['g-r']
            gi = param['g-i']
            imag = gmag - gi
            sed = [10**(gi / (-2.5)), 10**(gi / (-2.5)) / 10 **
                   (gr / (-2.5)), 1, np.random.uniform(1.0, 1.2)]
            n = param['sersic_n']
            re = param['rhalf_circularized']  # in arcsec
            # in galsim, re is circularized

            comp1 = {
                'model': galsim.Sersic,
                'model_params': {
                    'n': n,
                    'half_light_radius': re,
                },
                'shear_params': {
                    'q': q,
                    'beta': np.random.uniform(low=-90, high=90) * galsim.degrees,
                },
                'sed': np.array(sed),
                # 'n_knots': np.random.randint(0, 20),
                # 'knots_frac': 0.1,
                # 'knots_sed': np.array([0.2866302 , 0.2387235 , 0.20486748, 0.11319384])
            }
            galaxy = {'comp': [comp1],
                      'imag': imag,  # total mag for all components
                      'flux_fraction': [1.0]
                      }
            mgal = MockGal(bkg)
            mgal.gen_mock_lsbg(galaxy, verbose=False)
            #     mgal.display()
            mgal.write(os.path.join(
                output_dir, f'mock_{ind}.pkl'), overwrite=True)
            mgal.write_fits(output_dir=output_dir,
                            prefix='mock',
                            obj_id=ind,
                            overwrite=True)

            obj = lsbg_cat[ind]
            obj['viz-id'] = ind
            obj['ra'] = mgal.mock.info['ra']
            obj['dec'] = mgal.mock.info['dec']
            obj['mag_auto_i'] = mgal.mock.info['imag']

            model_dict = comp1
            obj['sersic_n'] = model_dict['model_params']['n']
            obj['sersic_rhalf_circ'] = model_dict['model_params']['half_light_radius']
            obj['sersic_ell'] = 1 - model_dict['shear_params']['q']
            obj['sersic_PA'] = model_dict['shear_params']['beta'].deg
            obj['sersic_sed'] = model_dict['sed']
            obj['mags'] = [mgal.mock.info[f'{filt}mag']
                           for filt in list('griz')]
            obj['prefix'] = f'./Cutout/mock_sample/mock_{obj["viz-id"]}'
        except Exception as e:
            print('Mockgal failed for id = {}'.format(bkg_id))
            print(e)
            continue

    lsbg_cat.write(
        f'./Catalog/mock_sample/mock_obj_cat_{low}_{high}.fits', overwrite=True)


if __name__ == '__main__':
    fire.Fire(gen_mock_gal)


# python gen_mock_gal.py --low 0 --high 10
