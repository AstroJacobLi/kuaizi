# Diezi 🥘
Diezi (碟子): Fill a small dish with HSC low surface brightness galaxies

This repo is for the notebook and scripts used for modeling LSBGs in HSC using scarlet.

`setup_env.sh` can be used on `Tiger` and `Tigressdata` to setup the proper environment for the task. Under LSST Pipeline, you can only use Python 3.7.8. 


### Data Structure

Cutouts (and PSFs) with size of 0.6 arcmin in 5 bands takes 11 Mb. 

After pickling, one galaxy in 5 bands takes 7.1 Mb. 

Thumb of rule: test the script/notebook on `tigressdata`, run the whole sample on `/tiger/scratch/gpfs/jiaxuanl`, only copy the figures to `/tigress/jiaxuanl/public_html` for displaying.

1. Catalogs are saved at `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog`. Please check `diezi/gen_cutout/candy_sample_cutout.ipynb` for modifying the Candy sample catalog (saved as `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/Candy/candy_cutout_cat.fits`).

2. Please use `diezi/gen_cutout/lsbg_cutout_s18a.sh` to generate cutouts. Files should be saved at `/tiger/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout`. 

3. 

### Image Gallery

Cutout images of the candy sample: http://tigress-web.princeton.edu/~jiaxuanl/candy/cutout_figure/page1.html