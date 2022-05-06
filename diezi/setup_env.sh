#!/bin/sh

# Setup hscPipe enviroment
# module purge
source /tigress/HSC/LSST/stack3_tiger/loadLSST.bash
# module load rh/devtoolset/6
# source /projects/HSC/LSST/stack/loadLSST.bash

setup lsst_apps
# setup obs_subaru # load this if you need to generate

# Remove the default scarlet path from PYTHONPATH. I use my own version of scarlet.
# See https://askubuntu.com/questions/345562/permanently-remove-item-from-path
export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20200903/stack/miniconda3-py37_4.8.2-cb4e2dc/Linux64/scarlet/lsst-dev-g52409ed422+f31336177f/lib/python"/""}
# export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20200903/conda/miniconda3-py37_4.8.2/envs/lsst-scipipe-cb4e2dc/lib/python3.7/site-packages/PyQt5"/""}

echo "LOAD ENVIRONMENT"