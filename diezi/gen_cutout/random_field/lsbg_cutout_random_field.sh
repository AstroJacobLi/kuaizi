#!/bin/sh

# . /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

export DATADIR="/scratch/gpfs/jiaxuanl/Data" # Directory of all data
export OUTPUT_DIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG" # Directory of all data
export CATALOG_DIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG"

# everything should be downloaded to /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout
# cat['radius'] is always the cutout size we should use!!!!
python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $CATALOG_DIR"/Catalog/random_field/lsbg_random_field3.fits" \
    --bands grizy --ra_name ra --dec_name dec \
    --name "viz-id" --output $OUTPUT_DIR"/Cutout/random_field3" \
    --catalog_dir $OUTPUT_DIR"/Catalog/random_field3" --catalog_suffix "random_field3" \
    --size 0.7 --prefix "random_field3" \
    --njobs 8 --psf True --overwrite False