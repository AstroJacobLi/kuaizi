#!/bin/bash
#SBATCH -J random1_0_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o random1_0_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1



python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'RANDOM_FIELD1/vanilla' \
    --PREFIX 'RANDOM_FIELD' \
    --cat_dir '/tigress/jiaxuanl/Data/HSC/LSBG/Catalog/random_field/random_field_cutout_cat_random_field.fits' \
    --filename '_lsbg_meas_vanilla.fits' \
    --low 0 --high None --method vanilla\
    --ncpu=12 --suffix '' --sigma=0.05


now=$(date +"%T")
echo "end time ... $now"
