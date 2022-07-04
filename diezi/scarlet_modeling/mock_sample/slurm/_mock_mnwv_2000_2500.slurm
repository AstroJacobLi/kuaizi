#!/bin/bash
#SBATCH -J mock_mnwv_2000_2500
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o ./log/mock_mnwv_2000_2500.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

source /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

# python ../script/sclt_multiprocess.py --njobs 16 \
#    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
#    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
#    --OUTPUT_SUBDIR 'MOCK_SAMPLE/wavelet' \
#    --PREFIX 'MOCK' \
#    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_5000.fits' \
#    --low 2000 --high 2500 \
#    --suffix '' --starlet_thresh 0.3 --method wavelet \
#    --monotonic True --bkg True --variance 0.000225 --scales '[0, 1, 2]' --min_grad -0.2

python ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE/wavelet' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_5000.fits' \
    --filename '_lsbg_meas_wavelet_monotonic.fits' \
    --low 2000 --high 2500 --method wavelet\
    --ncpu=32 --suffix '' --sigma=0.02


now=$(date +"%T")
echo "end time ... $now"
