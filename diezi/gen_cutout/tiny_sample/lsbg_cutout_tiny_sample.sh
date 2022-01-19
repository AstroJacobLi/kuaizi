#!/bin/sh

. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

export DATADIR="/tigress/jiaxuanl/Data" # Directory of all data
export OUTPUT_DIR="/tigress/jiaxuanl/Data/HSC/LSBG" # Directory of all data
export CATALOG_DIR="/tigress/jiaxuanl/Data/HSC/LSBG"

# everything should be downloaded to /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout
# cat['radius'] is always the cutout size we should use!!!!
python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $CATALOG_DIR"/Catalog/tiny_sample/lsbg_tiny_2.fits" \
    --bands grizy --ra_name ra --dec_name dec \
    --name "viz-id" --output $OUTPUT_DIR"/Cutout/tiny_sample" \
    --catalog_dir $OUTPUT_DIR"/Catalog/tiny_sample" --catalog_suffix "tiny_sample_2" \
    --size "cutout_size" --prefix "tiny_sample" \
    --njobs 30 --psf True --overwrite True