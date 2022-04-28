'''
python script to deploy slurm jobs for constructing training spectra
'''
import os
import sys
import fire


def deploy_modeling_job(low=0, high=1000, ind_list=None, name='mock_wvlt', ncpu=32,
                        method='wavelet', starlet_thresh=0.5, monotonic=True, bkg=True, min_grad=-0.001,
                        variance=0.03**2, scales=[0, 1, 2, 3, 4, 5, 6], sigma=0.02):
    ''' create slurm script and then submit 
    '''
    time = "11:59:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J {name}_{low}_{high}" if ind_list is None else f"#SBATCH -J {name}_ind_list",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=%i" % ncpu,
        "#SBATCH --mem-per-cpu=10G",
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH -o {name}_{low}_{high}.o" if ind_list is None else f"#SBATCH -o {name}_ind_list.o",
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
        f"python ../script/sclt_multiprocess.py --njobs {ncpu} \\",
        "   --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \\",
        "   --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \\",
        f"   --OUTPUT_SUBDIR 'MOCK_SAMPLE/{method}' \\",
        "   --PREFIX 'MOCK' \\",
        "   --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \\",
        f"   --low {low} --high {high} \\" if ind_list is None else f"   --ind_list {ind_list} \\",
        f"   --suffix '' --starlet_thresh {starlet_thresh} --method {method} \\",
        f"   --monotonic {monotonic} --bkg {bkg} --variance {variance} --scales '{scales}' --min_grad {min_grad}",
        "",
        f"python ../script/sclt_meas_mp.py \\",
        "    --DATADIR '/scratch/gpfs/jiaxuanl/Data/' \\",
        "    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \\",
        f"    --OUTPUT_SUBDIR 'MOCK_SAMPLE/{method}' \\",
        "    --PREFIX 'MOCK' \\",
        "    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_2000.fits' \\",
        f"    --filename '_lsbg_meas_{method}_monotonic.fits' \\",
        f"    --low {low} --high {high} --method {method}\\" if ind_list is None else f"    --ind_list {ind_list} \\",
        f"    --ncpu={ncpu} --suffix '' --sigma={sigma}",
        "",
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    # create the slurm script execute it and remove it
    if not os.path.isdir('./slurm'):
        os.mkdir('./slurm')
    f = open(f'./slurm/_{name}_{low}_{high}.slurm', 'w') if ind_list is None else open(
        f'./slurm/_{name}_ind_list.slurm', 'w')
    f.write(cntnt)
    f.close()
    os.system(f'sbatch ./slurm/_{name}_{low}_{high}.slurm') if ind_list is None else os.system(
        f'sbatch ./slurm/_{name}_ind_list.slurm')
    #os.system('rm _train.slurm')
    return None


if __name__ == '__main__':
    fire.Fire(deploy_modeling_job)

# python deploy_mock.py --name mock_wvlt --ncpu=16 --starlet_thresh=0.5 \
#  --method=wavelet --low=0 --high=500 --monotonic=True --variance=0 --min_grad 0.01 \
#  --scales="[0, 1, 2, 3]" --sigma=0.02

############ VANILLA #############
# python deploy_mock.py --name mock_vnla --ncpu=16 --method=vanilla \
# --low=0 --high=250 --monotonic=True --bkg=True -min_grad=-0.02 --sigma=0.02

# python deploy_mock.py --name mock_vnla --ncpu=16 --method=vanilla \
# --low=250 --high=500 --monotonic=True --bkg=True -min_grad=-0.02 --sigma=0.02

# python deploy_mock.py --name mock_vnla --ncpu=16 --method=vanilla \
# --low=500 --high=750 --monotonic=True --bkg=True -min_grad=-0.02 --sigma=0.02

# python deploy_mock.py --name mock_vnla --ncpu=16 --method=vanilla \
# --low=750 --high=1000 --monotonic=True --bkg=True -min_grad=-0.02 --sigma=0.02

# python deploy_mock.py --name mock_vnla --ncpu=16 --method=vanilla \
# --low=1000 --high=2000 --monotonic=True --bkg=True -min_grad=-0.02 --sigma=0.02

############ SPERGEL #############
# python deploy_mock.py --name mock_spgl --ncpu=16 --method=spergel \
# --low=1000 --high=1250 --monotonic=True --bkg=True -min_grad=-0.1 --sigma=0.02

# python deploy_mock.py --name mock_spgl --ncpu=16 --method=spergel \
# --low=1250 --high=1500 --monotonic=True --bkg=True -min_grad=-0.1 --sigma=0.02

# python deploy_mock.py --name mock_spgl --ncpu=16 --method=spergel \
# --low=1500 --high=1750 --monotonic=True --bkg=True -min_grad=-0.1 --sigma=0.02

# python deploy_mock.py --name mock_spgl --ncpu=16 --method=spergel \
# --low=1750 --high=2000 --monotonic=True --bkg=True -min_grad=-0.1 --sigma=0.02
