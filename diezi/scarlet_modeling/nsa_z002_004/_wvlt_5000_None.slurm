#!/bin/bash
#SBATCH -J wvlt_5000_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o wvlt_5000_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 16 \
   --DATADIR '/tigress/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'NSA_Z002_004' \
   --PREFIX 'NSA' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
   --low 5000 --high None \
   --suffix '' --starlet_thresh 1 --method wavelet

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/tigress/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'NSA_Z002_004' \
    --PREFIX 'NSA' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
    --filename '_lsbg_meas_wavelet.fits' \
    --low 5000 --high None --method wavelet\
    --ncpu=16 --suffix '' --sigma=0.5


now=$(date +"%T")
echo "end time ... $now"
