#!/bin/bash
#SBATCH -J mock_wvlt_200_400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=10G
#SBATCH --time=5:59:00
#SBATCH --export=ALL
#SBATCH -o mock_wvlt_200_400.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 12 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
   --PREFIX 'MOCK' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
   --low 200 --high 400 \
   --suffix '' --starlet_thresh 0.3 --method wavelet
   --monotonic True --bkg True --variance 0.000225 --scales '[0, 1, 2, 3, 4]' --min_grad -0.04

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
    --filename '_lsbg_meas_wavelet_monotonic.fits' \
    --low 200 --high 400 --method wavelet\
    --ncpu=12 --suffix '' --sigma=0.02


now=$(date +"%T")
echo "end time ... $now"
