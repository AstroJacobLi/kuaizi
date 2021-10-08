#!/bin/sh

. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

export DATADIR="/scratch/gpfs/jiaxuanl/Data/" # Directory of all data
export LSBGDIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG"

# everything should be downloaded to /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout
# cat['radius'] is always the cutout size we should use!!!!
python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $LSBGDIR"/Catalog/nsa_test_sample_211007.fits" \
    --bands grizy --ra_name ra --dec_name dec \
    --name "viz-id" --output $LSBGDIR"/Cutout/NSA" \
    --size "cutout_size" --prefix "nsa" \
    --njobs 3 --psf