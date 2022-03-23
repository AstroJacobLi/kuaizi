#!/bin/bash
#SBATCH -J nsa_spgl_ind_list
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o nsa_spgl_ind_list.o
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
   --OUTPUT_SUBDIR 'NSA_z001_002/spergel' \
   --PREFIX 'NSA' \
   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z001_002/nsa_cutout_cat_z001_002.fits' \
   --ind_list /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/z001_002_failed.npy \
   --suffix '' --starlet_thresh 0.5 --method spergel \
   --monotonic True --bkg True --variance 0.0009 --scales '[0, 1, 2, 3, 4, 5, 6]' --min_grad -0.1

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'NSA_z001_002/spergel' \
    --PREFIX 'NSA' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z001_002/nsa_cutout_cat_z001_002.fits' \
    --filename '_lsbg_meas_spergel.fits' \
    --ind_list /scratch/gpfs/jiaxuanl/Data/HSC/LSBG/z001_002_failed.npy \
    --ncpu=16 --suffix '' --sigma=0.05


now=$(date +"%T")
echo "end time ... $now"
