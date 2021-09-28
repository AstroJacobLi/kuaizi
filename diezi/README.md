# Diezi 🥘
Diezi (碟子): Fill a small dish with HSC low surface brightness galaxies

This repo is for the notebook and scripts used for modeling LSBGs in HSC using scarlet.

`diezi/setup_env.sh` can be used on `Tiger` and `Tigressdata` to setup the proper environment for the task. Under LSST Pipeline, only Python 3.7.8 is available. Please run `diezi/setup_env.sh` before running any code in `diezi`.


**Working directories:**
- Data: `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/`
- Code/Notebooks: `/home/jiaxuanl/Research/Packages/kuaizi/diezi`



### Data Structure

Cutouts (and corresponding PSFs) with size of 0.6 arcmin in 5 bands takes 11 Mb. 

After pickling, one galaxy in 5 bands takes 7.1 Mb. 

Thumb of rule: test the script/notebook on `tiger`, run the whole sample on `/tiger/scratch/gpfs/jiaxuanl`, only copy the figures to `/tigress/jiaxuanl/public_html` for displaying. 

Once we have a final sample, we will move catalogs, modeling outputs to `tigressdata` for storage.


### Work flow

1. Raw catalogs (from Johnny Greco) are saved at `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog`. Please check `diezi/gen_cutout/candy_sample_cutout.ipynb` for modifying the Candy sample catalog. I add `cutout_size` as well as the cutout directory in the new catalog, saved as `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/Candy/candy_cutout_cat.fits`. 

2. Please use `diezi/gen_cutout/lsbg_cutout_s18a.sh` to generate cutouts. Files should be saved at `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout`. For very few galaxies, HSC pipeline fails to generate PSFs at their locations. In this case, we use the default PSF (randomly choosen from a location), they are named as `psf_x.fits` in the `Cutout` folder. 


### Image Gallery

Cutout images of the candy sample: http://tigress-web.princeton.edu/~jiaxuanl/candy/cutout_figure/page1.html


### Structural measurement
pip install statmorph --user