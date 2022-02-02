#!/bin/bash
#SBATCH -J wvlt_3000_4000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o wvlt_3000_4000.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

# python3 ../script/sclt_multiprocess.py --njobs 16 \
#    --DATADIR '/tigress/jiaxuanl/Data/' \
#    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
#    --OUTPUT_SUBDIR 'NSA_Z002_004' \
#    --PREFIX 'NSA' \
#    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
#    --low 3000 --high 4000 \
#    --suffix '' --starlet_thresh 1 --method wavelet

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/tigress/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'NSA_Z002_004' \
    --PREFIX 'NSA' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
    --filename '_lsbg_meas_wavelet.fits' \
    --low 3000 --high 4000 --method wavelet\
    --ncpu=16 --suffix '' --sigma=0.5


now=$(date +"%T")
echo "end time ... $now"
