## Generat cutout images for NSA MW-like hosts

This folder is for generating cutout images for LSBGs matched with MW-like hosts. We currently focus on two redshift bins: 0.01-0.02 and 0.02-0.04. 
The catalog of matched LSBGs is `/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/lsbg_NSA_MW_match.fits`.

- In `_nsa_sample_cutout.ipynb`, we creat catalogs for both redshift bins, and save them as `./Catalog/NSA/z001_002/lsbg_NSA_MW_z001_002.fits` and `./Catalog/NSA/z002_004/lsbg_NSA_MW_z002_004.fits`. 

Run `/home/jiaxuanl/Research/Packages/kuaizi/diezi/gen_cutout/nsa_sample/lsbg_cutout_nsa.sh`, remember to change the corresponding input arguments.

In the end, we get `$gpfs/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits`. 

Cutout images are saved at `$gpfs/Data/HSC/LSBG/Cutout/NSA/`.

In principle, you can generate RGB images of the cutouts and display them on webpages in `gen_rgb.ipynb`.