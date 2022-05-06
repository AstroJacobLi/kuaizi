#!/bin/bash
#SBATCH -J mock_i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=40G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o mock_bkg_i.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

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
    --bands i --ra_name ra --dec_name dec \
    --name "index" --output $OUTPUT_DIR"/Cutout/mock_sample/bkg/" \
    --catalog_dir $OUTPUT_DIR"/Catalog/mock_sample" --catalog_suffix "mock_sample_2" \
    --size 1 --prefix "MockBkg" \
    --njobs 10 --psf True --overwrite False \
    --low 2850 --high 5000

# python gen_mock_gal.py --low 2000 --high 5000

now=$(date +"%T")
echo "end time ... $now"
