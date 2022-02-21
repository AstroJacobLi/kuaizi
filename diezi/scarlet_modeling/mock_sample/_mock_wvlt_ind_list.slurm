#!/bin/bash
#SBATCH -J mock_wvlt_ind_list
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o mock_wvlt_ind_list.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 8 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
   --PREFIX 'MOCK' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
   --ind_list "[25, 28, 30, 49, 52, 67, 76, 89, 94, 103, 108, 115, 116, 125, 127, 137, 140, 142, 154, 155, 156, 157, 161, 192, 194, 197, 199, 223, 227, 228, 241, 245, 252, 263, 284, 298, 303, 308, 315, 316, 322, 331, 336, 337, 349, 355, 358, 361, 366, 378, 383, 384, 391, 396, 405, 411, 412, 420, 421, 423, 426, 452, 455, 469, 473, 474, 478, 481]" \
   --suffix '' --starlet_thresh 0.01 --method wavelet
   --monotonic True --variance 0.01**2 --scales '[0, 1, 2, 3, 4, 5]' --min_grad 0

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \
    --filename '_lsbg_meas_wavelet_monotonic.fits' \
    --low 0 --high 500 \
    --ncpu=8 --suffix '' --sigma=0.02


now=$(date +"%T")
echo "end time ... $now"
