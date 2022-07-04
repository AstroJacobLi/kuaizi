#!/bin/bash
#SBATCH -J rdm3_vnl.2.0k_1500_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o ./log/rdm3_vnl.2.0k_1500_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

source /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python ../script/sclt_multiprocess.py --njobs 12 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'RANDOM_FIELD3/vanilla' \
   --PREFIX 'RANDOM_FIELD' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/random_field3/random_field3_cutout_cat_random_field3.fits' \
   --low 1500 --high None \
   --suffix '' --starlet_thresh 0.5 --method vanilla \
   --monotonic True --bkg True --variance 0.0009 --scales '[0, 1, 2, 3, 4, 5, 6]' --min_grad -0.02

python ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'RANDOM_FIELD3/vanilla' \
    --PREFIX 'RANDOM_FIELD' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/random_field3/random_field3_cutout_cat_random_field3.fits' \
    --filename '_lsbg_meas_vanilla.fits' \
    --low 1500 --high None --method vanilla\
    --ncpu=12 --suffix '' --sigma=0.05


now=$(date +"%T")
echo "end time ... $now"
