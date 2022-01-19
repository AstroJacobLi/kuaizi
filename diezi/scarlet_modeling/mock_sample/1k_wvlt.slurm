#!/bin/bash
#SBATCH --job-name=scltv_random      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jiaxuanl@princeton.edu

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 25 \
--DATADIR "/tigress/jiaxuanl/Data/" \
--OUTPUT_DIR "/tigress/jiaxuanl/Data/HSC/LSBG/" \
--PREFIX "MOCK" \
--OUTPUT_SUBDIR "MOCK_SAMPLE" \
--cat_dir "/tigress/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat.fits" \
--low 0 --high None \
--suffix "" --starlet_thresh 0.5 --method 'wavelet'