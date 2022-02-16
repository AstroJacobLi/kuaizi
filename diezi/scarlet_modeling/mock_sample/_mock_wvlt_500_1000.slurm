#!/bin/bash
#SBATCH -J mock_wvlt_500_1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o mock_wvlt_500_1000.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 16 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
   --PREFIX 'MOCK' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
   --low 500 --high 1000 \
   --suffix '' --starlet_thresh 1 --method wavelet
   --monotonic True --variance 0.05**2 --scales '[0, 1, 2, 3, 4, 5, 6]'

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
    --filename '_lsbg_meas_wavelet_monotonic.fits' \
    --low 500 --high 1000 --method wavelet\
    --ncpu=16 --suffix '' --sigma=0.02


now=$(date +"%T")
echo "end time ... $now"
