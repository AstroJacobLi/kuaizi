#!/bin/bash
#SBATCH -J hsc_cutout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o hsc_cutout.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu


now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1
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
    --njobs 25 --psf True --overwrite False

# python3 ../s18a_batch_cutout.py \
#     $DATADIR\
#     $CATALOG_DIR"/Catalog/random_field/lsbg_random_field3.fits" \
#     --bands grizy --ra_name ra --dec_name dec \
#     --name "viz-id" --output $OUTPUT_DIR"/Cutout/random_field3" \
#     --catalog_dir $OUTPUT_DIR"/Catalog/random_field3" --catalog_suffix "random_field3" \
#     --size "cutout_size" --prefix "random_field3" \
#     --njobs 25 --psf True --overwrite False

now=$(date +"%T")
echo "end time ... $now"