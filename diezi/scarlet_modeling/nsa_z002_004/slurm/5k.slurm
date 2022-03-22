#!/bin/bash
#SBATCH --job-name=scltv_5k       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jiaxuanl@princeton.edu

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 30 \
    --DATADIR "/tigress/jiaxuanl/Data/" \
    --OUTPUT_DIR "/tigress/jiaxuanl/Data/HSC/LSBG/" \
    --PREFIX "NSA" \
    --OUTPUT_SUBDIR "NSA_Z002_004" \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
    --low 4000 --high 5000 \
    --suffix "" --starlet_thresh 0.5 --method 'vanilla'

python3 ../script/vanilla_measure_all.py \
    --DATADIR '/tigress/jiaxuanl/Data' \
    --OUTPUT_DIR '/tigress/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'NSA_Z002_004' \
    --PREFIX 'nsa' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \
    --filename '_lsbg_meas_vanilla.fits' \
    --low 4000 --high 5000 --suffix '_5k_'