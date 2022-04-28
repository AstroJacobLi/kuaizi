#!/bin/bash
#SBATCH -J candy_spgl_0_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15
#SBATCH --mem-per-cpu=12G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o candy_spgl_0_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python ../script/sclt_multiprocess.py --njobs 15 \
   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
   --OUTPUT_SUBDIR 'candy/spergel' \
   --PREFIX 'candy' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/candy/candy_cutout_cat_candy.fits' \
   --low 0 --high None \
   --suffix '' --starlet_thresh 0.5 --method spergel \
   --monotonic True --bkg True --variance 0.0009 --scales '[0, 1, 2, 3, 4, 5, 6]' --min_grad -0.1

python ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'candy/spergel' \
    --PREFIX 'candy' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/candy/candy_cutout_cat_candy.fits' \
    --filename '_lsbg_meas_spergel.fits' \
    --low 0 --high None --method spergel\
    --ncpu=15 --suffix '' --sigma=0.05


now=$(date +"%T")
echo "end time ... $now"
