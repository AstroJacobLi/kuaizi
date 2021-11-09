#!/bin/bash
#SBATCH --job-name=scarlet-wvlt-nsa       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jiaxuanl@princeton.edu

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

# python3 wvlt_multiprocess.py --njobs 4 --ind_list "[51, 53, 110, 112, 196]" --suffix "_complement" --starlet_thresh 0.5

# python3 wvlt_multiprocess.py --njobs 1 --ind_list "[0]" --suffix "" --starlet_thresh 0.5

python3 wvlt_multiprocess.py --njobs 4 --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/NSA/nsa_cutout_cat_1013.fits' --low 0 --high 10 --suffix "" --starlet_thresh 0.5

