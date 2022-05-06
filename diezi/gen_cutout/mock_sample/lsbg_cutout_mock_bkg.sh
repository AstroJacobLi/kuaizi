#!/bin/sh

. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
setup obs_subaru
export DATADIR="/tigress/jiaxuanl/Data" # Directory of all data
export OUTPUT_DIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG" # Directory of all data
export CATALOG_DIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG"

# everything should be downloaded to /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout
# cat['radius'] is always the cutout size we should use!!!!
python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $CATALOG_DIR"/Catalog/mock_sample/skyobj_pos_clean_simple.fits" \
    --bands g --ra_name ra --dec_name dec \
    --name "index" --output $OUTPUT_DIR"/Cutout/mock_sample/bkg/" \
    --catalog_dir $OUTPUT_DIR"/Catalog/mock_sample" --catalog_suffix "mock_sample_2" \
    --size 1 --prefix "MockBkg" \
    --njobs 8 --psf True --overwrite False \
    --low 2008 --high 2009

# python gen_mock_gal.py --low 2000 --high 2010