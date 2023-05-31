#!/bin/sh

# Setup hscPipe enviroment
# module load rh/devtoolset/8
# . /tigress/HSC/LSST/stack3_tiger/loadLSST.bash
LSST_CONDA_ENV_NAME=lsst-scipipe-4.0.1
# source /projects/HSC/LSST/stack/loadLSST.bash
source /scratch/gpfs/HSC/LSST/stack/loadLSST.sh

setup lsst_distrib
setup lsst_apps
setup obs_subaru

# Remove the default scarlet path from PYTHONPATH. I use my own version of scarlet.
# See https://askubuntu.com/questions/345562/permanently-remove-item-from-path
export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20220527/conda/envs/lsst-scipipe-4.0.0/share/eups/Linux64/scarlet_extensions/g9d18589735+cc492336a9/lib/python"/""}
export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20220527/conda/envs/lsst-scipipe-4.0.0/share/eups/Linux64/scarlet/gd32b658ba2+4083830bf8/lib/python"/""}
export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20220527/conda/envs/lsst-scipipe-4.0.0/share/eups/Linux64/scarlet/gd32b658ba2+4083830bf8/lib/python/scarlet"/""}

echo "LOAD ENVIRONMENT LSSTPIPE-4.0.0"
