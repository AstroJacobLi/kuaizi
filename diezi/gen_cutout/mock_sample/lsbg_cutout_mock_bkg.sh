#!/bin/sh

#. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

export DATADIR="/tigress/jiaxuanl/Data" # Directory of all data
export OUTPUT_DIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG" # Directory of all data
export CATALOG_DIR="/tigress/jiaxuanl/Data/HSC/LSBG"

# everything should be downloaded to /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout
# cat['radius'] is always the cutout size we should use!!!!
python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $CATALOG_DIR"/Catalog/mock_sample/skyobj_pos_simple.fits" \
    --bands griz --ra_name g_ra --dec_name g_dec \
    --name "index" --output $OUTPUT_DIR"/Cutout/mock_sample/bkg/" \
    --catalog_dir $OUTPUT_DIR"/Catalog/mock_sample" --catalog_suffix "mock_sample" \
    --size 0.7 --prefix "MockBkg" \
    --njobs 30 --psf True --overwrite False \
    --low 10 --high 1000