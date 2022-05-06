#!/bin/bash
#SBATCH -J candy_spgl_0_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0:20:00
#SBATCH --export=ALL
#SBATCH -o candy_spgl_0_4.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

source /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
# export OMP_NUM_THREADS=1
# export QT_DEBUG_PLUGINS=1

python ../script/sclt_multiprocess.py --njobs 2 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'candy/spergel' \
   --PREFIX 'candy' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/candy/candy_cutout_cat_candy.fits' \
   --low 0 --high 2 \
   --suffix '' --starlet_thresh 0.5 --method spergel \
   --monotonic True --bkg True --variance 0.0009 --scales '[0, 1, 2, 3, 4, 5, 6]' --min_grad -0.1

# python ../script/sclt_meas_mp.py \
#     --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
#     --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
#     --OUTPUT_SUBDIR 'candy/spergel' \
#     --PREFIX 'candy' \
#     --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/candy/candy_cutout_cat_candy.fits' \
#     --filename '_lsbg_meas_spergel.fits' \
#     --low 0 --high 4 --method spergel\
#     --ncpu=4 --suffix '' --sigma=0.05


now=$(date +"%T")
echo "end time ... $now"
