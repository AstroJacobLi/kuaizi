#!/bin/bash
#SBATCH -J random_field_0_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o random_field_0_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 8 \
   --DATADIR '/tigress/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'RANDOM_FIELD' \
   --PREFIX 'RANDOM_FIELD' \
   --cat_dir '/tigress/jiaxuanl/Data/HSC/LSBG/Catalog/random_field/random_field_cutout_cat_random_field.fits' \
   --low 0 --high None \
   --suffix '' --starlet_thresh 1 --method wavelet

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/tigress/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'RANDOM_FIELD' \
    --PREFIX 'RANDOM_FIELD' \
    --cat_dir '/tigress/jiaxuanl/Data/HSC/LSBG/Catalog/random_field/random_field_cutout_cat_random_field.fits' \
    --filename '_random_meas_wavelet.fits' \
    --low 0 --high None --method wavelet\
    --ncpu=8 --suffix '' --sigma=0.5


now=$(date +"%T")
echo "end time ... $now"
