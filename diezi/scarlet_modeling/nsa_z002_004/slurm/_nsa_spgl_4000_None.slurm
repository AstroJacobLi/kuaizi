#!/bin/bash
#SBATCH -J nsa_spgl_4000_None
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=11:59:00
#SBATCH --export=ALL
#SBATCH -o nsa_spgl_4000_None.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

# python3 ../script/sclt_multiprocess.py --njobs 16 \
#    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
#    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
#    --OUTPUT_SUBDIR 'NSA_Z002_004/spergel' \
#    --PREFIX 'NSA' \
#    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
#    --low 4000 --high None \
#    --suffix '' --starlet_thresh 0.5 --method spergel \
#    --monotonic True --bkg True --variance 0.0009 --scales '[0, 1, 2, 3, 4, 5, 6]' --min_grad -0.1

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'NSA_Z002_004/spergel' \
    --PREFIX 'NSA' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
    --filename '_lsbg_meas_spergel.fits' \
    --low 4000 --high None --method spergel\
    --ncpu=16 --suffix '' --sigma=0.02


now=$(date +"%T")
echo "end time ... $now"
