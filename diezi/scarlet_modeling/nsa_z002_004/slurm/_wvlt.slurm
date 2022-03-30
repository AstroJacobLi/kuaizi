#!/bin/bash
#SBATCH -J wvlt_0_1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o wvlt_0_1000.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 32 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --PREFIX 'MOCK' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_1000.fits' \
   --low 0 --high 1000 \
   --suffix '' --starlet_thresh 1 --method 'wavelet'now=$(date +"%T")
echo "end time ... $now"
