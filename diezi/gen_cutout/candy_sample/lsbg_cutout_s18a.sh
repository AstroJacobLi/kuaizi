#!/bin/sh

. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

export DATADIR="/scratch/gpfs/jiaxuanl/Data/" # Directory of all data
export LSBGDIR="/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/"

# python3 s18a_batch_cutout.py \
#     $DATADIR\
#     $LSBGDIR"/Catalog/LSB-Greco2018.fits" \
#     --bands grizy --ra_name RAJ2000 --dec_name DEJ2000 \
#     --name Seq --output $LSBGDIR"/Cutout" \
#     --size 1.0 --unit arcmin \
#     --njobs 2 --psf

# everything should be downloaded to /tigress/jiaxuanl/Data/HSC/LSBG/Cutout

python3 ../s18a_batch_cutout.py \
    $DATADIR\
    $LSBGDIR"/Catalog/candy_sample_210313.fits" \
    --bands grizy --ra_name ra --dec_name dec \
    --name Seq --output $LSBGDIR"/Cutout" \
    --size cutout_size --prefix candy \
    --njobs 2 --psf