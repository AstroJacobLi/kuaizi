from __future__ import division, print_function

import numpy as np

from astropy import wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Ellipse

from ..display import display_single, IMG_CMAP, SEG_CMAP
import kuaizi


import os
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#########################################################################
########################## The Tractor related ##########################
#########################################################################

# Add sources to tractor
def add_tractor_sources(obj_cat, sources, w, shape_method='manual', band='r'):
    '''
    Add tractor sources to the sources list.

    Parameters:
    ----------
    obj_cat: astropy Table, objects catalogue.
    sources: list, to which we will add objects.
    w: wcs object.
    shape_method: string, 'manual' or 'decals' or 'hsc'. 
        If 'manual', it will adopt the manually measured shapes. 
        If 'decals', it will adopt shapes in 'DECaLS' tractor catalog.
        If 'hsc', it will adopt shapes in HSC CModel catalog.

    Returns:
    --------
    sources: list of sources.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE
    from tractor.sersic import SersicGalaxy, SersicIndex

    #if shape_method is 'manual' or 'decals':
    obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))
    comp_galaxy = obj_cat[obj_type == 'COMP']
    dev_galaxy = obj_cat[obj_type == 'DEV']
    exp_galaxy = obj_cat[obj_type == 'EXP']
    rex_galaxy = obj_cat[obj_type == 'REX']
    ser_galaxy = obj_cat[obj_type == 'SER']
    psf_galaxy = obj_cat[np.logical_or(obj_type =='PSF', obj_type=='   ')]

    # elif shape_method is 'hsc':
    #     star_mask = obj_cat['{}_extendedness'.format(band)] < 0.5
    #     psf_galaxy = obj_cat[star_mask]

    #     fracdev = obj_cat['cmodel_fracdev'].values
    #     dev_galaxy = obj_cat[(fracdev >= 0.5) & (~star_mask)]
    #     exp_galaxy = obj_cat[(fracdev < 0.5) & (~star_mask)]
    # else:
    #     raise ValueError('Only "manual", "decals", or "hsc" is supported now.')

    if shape_method is 'manual':
        # Using manually measured shapes
        if sources is None:
            sources = []
        
        for obj in comp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
                                90.0 + obj['theta'] * 180.0 / np.pi),
                    Flux(0.6 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
                                90.0 + obj['theta'] * 180.0 / np.pi)))
        for obj in dev_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in exp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in ser_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                SersicGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi)), 
                    SersicIndex(2.0)
                    )
            )
        for obj in rex_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in psf_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        # for obj in obj_cat:
        #     pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
        #     if obj['type'].rstrip(' ') == 'COMP':
        #         sources.append(
        #             CompositeGalaxy(
        #                 PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
        #                             90.0 + obj['theta'] * 180.0 / np.pi),
        #                 Flux(0.6 * obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
        #                             90.0 + obj['theta'] * 180.0 / np.pi)))
        #     elif obj['type'].rstrip(' ') == 'DEV':
        #         sources.append(
        #             DevGalaxy(
        #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
        #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
        #     elif obj['type'].rstrip(' ') == 'EXP':
        #         sources.append(
        #             ExpGalaxy(
        #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
        #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
        #     elif obj['type'].rstrip(' ') == 'SER':
        #         sources.append(
        #             SersicGalaxy(
        #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
        #                             (90.0 + obj['theta'] * 180.0 / np.pi)), 
        #                 SersicIndex(2.0)
        #                 )
        #         )
        #     elif obj['type'].rstrip(' ') == 'REX':
        #         sources.append(
        #             ExpGalaxy(
        #                 PixPos(pos_x, pos_y), Flux(obj['flux']),
        #                 GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
        #                             (90.0 + obj['theta'] * 180.0 / np.pi))))
        #     elif obj['type'].rstrip(' ') == 'PSF' or obj['type'].rstrip(' ') == '   ':
        #         sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print(" - Now you have %d sources" % len(sources))

    elif shape_method is 'decals':
        ## Using DECaLS shapes
        if sources is None:
            sources = []
        for obj in comp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             obj['shapeexp_e2']), Flux(0.6 * obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             obj['shapedev_e2'])))
        for obj in dev_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             -obj['shapedev_e2'])))
        for obj in exp_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))
        for obj in rex_galaxy:
            #if obj['point_source'] > 0.0:
            #            sources.append(PointSource(PixPos(w.wcs_world2pix([[obj['ra'], obj['dec']]],1)[0]),
            #                                               Flux(obj['flux'])))
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))

        for obj in psf_galaxy:
            pos_x, pos_y = w.wcs_world2pix([[obj['ra'], obj['dec']]], 0)[0]
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print(" - Now you have %d sources" % len(sources))

    elif shape_method is 'hsc':
        from unagi import catalog
        ## Using HSC CModel catalog
        if sources is None:
            sources = []
        
        for obj in ser_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
            r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                                        obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
                                        to_pixel=False, update=False) # arcsec, degree
            sources.append(
                SersicGalaxy(
                    PixPos(pos_x, pos_y), Flux(flux),
                    GalaxyShape(r_gal, ba_gal, pa_gal + 90),
                    SersicIndex(1.0)
                    )
                )

        for obj in dev_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
            r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                                        obj, shape_type='cmodel_dev_ellipse', axis_ratio=True,
                                        to_pixel=False, update=False) # arcsec, degree
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(flux),
                    GalaxyShape(r_gal, ba_gal, pa_gal + 90), 
                    )
            )
        
        for obj in exp_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
            r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                                        obj, shape_type='cmodel_exp_ellipse', axis_ratio=True,
                                        to_pixel=False, update=False) # arcsec, degree
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(flux),
                    GalaxyShape(r_gal, ba_gal, pa_gal + 90)
                    )
                )


        for obj in rex_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)
            r_gal, ba_gal, pa_gal = catalog.moments_to_shape(
                                        obj, shape_type='cmodel_ellipse', axis_ratio=True,
                                        to_pixel=False, update=False) # arcsec, degree
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(flux),
                    GalaxyShape(r_gal, ba_gal, pa_gal)
                    )
                )

        for obj in psf_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            flux = 10**((kuaizi.HSC_zeropoint - obj['cmodel_mag']) / 2.5)

            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(flux)))
        
        print(" - Now you have %d sources" % len(sources))
    else:
         raise ValueError('Cannot use this shape method') 
    return sources

# Do tractor iteration
def tractor_iteration(obj_cat, w, img_data, invvar, psf_obj, pixel_scale, 
                      shape_method='manual', freeze_pos=False,
                      kfold=4, first_num=50, band_name=None, fig_name=None, verbose=False):
    '''
    Run tractor iteratively.

    Parameters:
    -----------
    obj_cat: objects catalogue.
    w: wcs object.
    img_data: 2-D np.array, image.
    invvar: 2-D np.array, inverse variance matrix of the image.
    psf_obj: PSF object, defined by tractor.psf.PixelizedPSF() class.
    pixel_scale: float, pixel scale in unit arcsec/pixel.
    shape_method: if 'manual', then adopt manually measured shape. If 'decals', then adopt DECaLS shape from tractor files.
    kfold: int, how many iterations you want to run.
    first_num: how many objects will be fit in the first run.
    fig_name: string, if not None, it will save the tractor subtracted image to the given path.

    Returns:
    -----------
    sources: list, containing tractor model sources.
    trac_obj: optimized tractor object after many iterations.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE

    step = int((len(obj_cat) - first_num) / (kfold - 1))
    for i in range(kfold):
        if i == 0:
            obj_small_cat = obj_cat[:first_num]
            sources = add_tractor_sources(obj_small_cat, None, w, shape_method=shape_method)
        else:
            obj_small_cat = obj_cat[first_num + step * (i - 1) : first_num + step * (i)]
            sources = add_tractor_sources(obj_small_cat, sources, w, shape_method=shape_method)

        with HiddenPrints():
            tim = Image(data=img_data,
                        invvar=invvar,
                        psf=psf_obj,
                        wcs=NullWCS(pixscale=pixel_scale),
                        sky=ConstantSky(0.0),
                        photocal=NullPhotoCal()
                        )
            trac_obj = Tractor([tim], sources)
            trac_mod = trac_obj.getModelImage(0, minsb=0.0)

            if freeze_pos:
                for src in sources:
                    src.freezeParam('pos')
            trac_obj.freezeParam('images')
            trac_obj.optimize_loop()
        
        
        ########################
        plt.rc('font', size=20)
        if i % 2 == 1 or i == (kfold - 1) :
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18,8))

            with HiddenPrints():
                trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[:])

            ax1 = display_single(img_data, ax=ax1, scale_bar=False)
            if band_name is not None:
                ax1.set_title(f'{band_name}-band raw image')
            else:
                ax1.set_title('raw image')
            ax2 = display_single(trac_mod_opt, ax=ax2, scale_bar=False, contrast=0.02)
            ax2.set_title('tractor model')
            ax3 = display_single(abs(img_data - trac_mod_opt), ax=ax3, scale_bar=False, color_bar=True, contrast=0.05)
            ax3.set_title('residual')

            if i == (kfold - 1):
                if fig_name is not None:
                    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
                    plt.show()
                    print('   The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data)) 
            elif verbose:
                plt.show()
                print('   The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data)) 
            else:
                plt.close()
                print('   The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data)) 

        #trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[1:])
        #ax4 = display_single(img_data - trac_mod_opt, ax=ax4, scale_bar=False, color_bar=True, contrast=0.05)
        #ax4.set_title('remain central galaxy')


    return sources, trac_obj, fig


def tractor_hsc(obj_name, coord, s_ang, filt, channels, data, hsc_dr, use_cmodel_filt=None, freeze_pos=True, verbose=False):
    '''
    Run `the tractor` on HSC images, for Merian survey.

    Parameters:
        obj_name (str): name of the object. 
        coord (astropy.coordinate.SkyCoord): Coordinate of the object.
        s_ang (astropy.units.arcsec): searching (angular) radius.
        filt (str): filter name, such as 'r'.
        channels (str): all filters, such as 'grizy'.
        data (kuaizi.detection.Data): an data structure which contains images, weights, wcs, PSFs, etc.
        hsc_dr: archive of HSC data, such as using `pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')`.
        use_cmodel_filt (str): if not None (such as `use_cmodel_filt='i'`), 
            models in all bands will be initialized using the CModel catalog in this band. 
        freeze_pos (bool): whether freezing the positions of objects during fitting.
        verbose (bool): whether being verbose.

    Return: 

    '''
    print("### `" + obj_name + f"` {filt}-band")
    layer_ind = channels.index(filt)
    
    if use_cmodel_filt is not None:
        cmodel_filt = use_cmodel_filt
    else:
        cmodel_filt = filt


    ## Initialize `unagi`
    from unagi import hsc, config
    from unagi import plotting
    from unagi import task, catalog

    # Retrieve HSC catalog
    cutout_objs = task.hsc_box_search(
        coord, box_size=s_ang * 1.1, archive=hsc_dr,
        verbose=True, psf=True, cmodel=True, aper=True, shape=True,
        meas=cmodel_filt, flux=False, aper_type='3_20')
    
    cutout_clean, clean_mask = catalog.select_clean_objects(
        cutout_objs, return_catalog=True, verbose=False) # Select "clean" images
    
    x, y = data.wcs.wcs_world2pix(cutout_clean['ra'], cutout_clean['dec'], 0)
    cutout_clean['x'] = x
    cutout_clean['y'] = y

    # sort by magnitude
    cutout_clean.sort('cmodel_mag')

    # Remove weird objects: abs(i_psf_mag - i_cmodel_mag) > 1
    cutout_clean = cutout_clean[(cutout_clean['psf_mag'] - cutout_clean['cmodel_mag']) < 2.5]
    
    # Plot HSC CModel catalog on top of the rgb image
    if filt == 'i':
        stretch = 1
        Q = 0.5
        channel_map = scarlet.display.channels_to_rgb(len(channels))

        img_rgb = scarlet.display.img_to_rgb(
            data.images,
            norm=scarlet.display.AsinhMapping(minimum=-0.2, stretch=stretch, Q=Q),
            channel_map=channel_map)
    
        _ = plotting.cutout_show_objects(
            img_rgb, cutout_clean, cutout_wcs=data.wcs, xsize=8, show_weighted=True) # Exp is brown. Dev is dashed-white.
        plt.savefig(obj_name + '_cmodel_i.png', bbox_inches='tight')
    
    # Find out target galaxy
    catalog_c = SkyCoord(cutout_clean['ra'], cutout_clean['dec'], unit='deg')
    dist = coord.separation(catalog_c)
    cen_obj_ind = np.argsort(dist)[0]
    cen_obj = cutout_clean[cen_obj_ind]

    ## Assign types to each object
    obj_type = np.empty_like(cutout_clean['object_id'], dtype='S4')
    star_mask = cutout_clean['{}_extendedness'.format(filt)] < 0.5
    obj_type[star_mask] = 'PSF'

    fracdev = cutout_clean['cmodel_fracdev']
    obj_type[(fracdev >= 0.5) & (~star_mask)] = 'DEV' # dev_galaxy
    obj_type[(fracdev < 0.5) & (~star_mask)] = 'EXP' # exp_galaxy
    obj_type[cen_obj_ind] = 'SER'
    
    cutout_clean['type'] = obj_type
    
    psf_obj = PixelizedPSF(data.psfs[layer_ind]) # Construct PSF
    
    
    kfold = 4
    while True:
        try:
            if kfold == 1:
                break 
            sources, trac_obj, fig = tractor_iteration(
                cutout_clean,
                data.wcs,
                data.images[layer_ind],
                data.weights[layer_ind],
                psf_obj,
                kuaizi.HSC_pixel_scale,
                shape_method='hsc',
                freeze_pos=freeze_pos,
                kfold=kfold,
                first_num=cen_obj_ind + 1,
                band_name=filt, 
                fig_name=obj_name + '_tractor_' + filt, 
                verbose=verbose)
            
        except Exception as e:
            print('   ' + str(e))
            kfold -= 1
            pass
        else:
            break
        
    return trac_obj