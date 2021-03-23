#!/bin/bash
#SBATCH --job-name=scarlet-wvlt-200-300        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jiaxuanl@princeton.edu

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh

python3 wvlt_multiprocess.py --njobs 4 --ind_list "[181, 182, 196, 197]" --starlet_thresh 0.5