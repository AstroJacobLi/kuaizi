'''
python script to deploy slurm jobs for constructing training spectra
'''
import os
import sys
import fire


def deploy_modeling_job(low=0, high=1000, name='wvlt', ncpu=32, starlet_thresh=1, method='wavelet'):
    ''' create slurm script and then submit 
    '''
    time = "48:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J {name}_{low}_{high}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=%i" % ncpu,
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH -o {name}_{low}_{high}.o",
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=jiaxuanl@princeton.edu",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "module purge",
        ". /home/jiaxuanl/Research/Packages/kuaizi/diezi/setup_env.sh",
        "export OMP_NUM_THREADS=1",
        "",
        f"python3 ../script/sclt_multiprocess.py --njobs {ncpu} \\",
        "   --DATADIR '/tigress/jiaxuanl/Data/' \\",
        "   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \\",
        "   --OUTPUT_SUBDIR 'NSA_Z002_004' \\",
        "   --PREFIX 'NSA' \\",
        "   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \\",
        f"   --low {low} --high {high} \\",
        f"   --suffix '' --starlet_thresh {starlet_thresh} --method {method}",
        "",
        f"python3 ../script/sclt_meas_mp.py \\",
        "    --DATADIR '/tigress/jiaxuanl/Data/' \\",
        "    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \\",
        "    --OUTPUT_SUBDIR 'NSA_Z002_004' \\",
        "    --PREFIX 'NSA' \\",
        "    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits' \\",
        f"    --filename '_lsbg_meas_{method}.fits' \\",
        f"    --low {low} --high {high} --method {method}\\",
        f"    --ncpu={ncpu} --suffix '' --sigma=0.5",
        "",
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    # create the slurm script execute it and remove it
    f = open(f'_{name}_{low}_{high}.slurm', 'w')
    f.write(cntnt)
    f.close()
    os.system(f'sbatch _{name}_{low}_{high}.slurm')
    #os.system('rm _train.slurm')
    return None


if __name__ == '__main__':
    fire.Fire(deploy_modeling_job)

# python deploy_z002_004.py --name wvlt --ncpu=16 --starlet_thresh=1 --method=wavelet --low=0 --high=1000
