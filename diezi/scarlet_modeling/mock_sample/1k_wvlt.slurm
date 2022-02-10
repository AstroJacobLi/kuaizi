#!/bin/bash
#SBATCH --job-name=mock_wvlt      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=jiaxuanl@princeton.edu

module purge
. /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh
export OMP_NUM_THREADS=1

python3 ../script/sclt_multiprocess.py --njobs 30 \
    --DATADIR "/scratch/gpfs/jiaxuanl/Data/" \
    --OUTPUT_DIR "/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/" \
    --PREFIX "MOCK" \
    --OUTPUT_SUBDIR "MOCK_SAMPLE" \
    --cat_dir "/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_1000.fits" \
    --low 0 --high None \
    --suffix "" --starlet_thresh 1.0 --method "wavelet"

python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_1000.fits' \
    --filename '_lsbg_meas.fits' \
    --low 0 --high None --method 'wavelet' --ncpu=30  --sigma=0.5
