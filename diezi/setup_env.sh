#!/bin/sh

# Setup hscPipe enviroment
module load rh/devtoolset/6
. /tigress/HSC/LSST/stack3_tiger/loadLSST.bash

setup lsst_apps
setup obs_subaru

# Remove the default scarlet path from PYTHONPATH. I use my own version of scarlet.
# See https://askubuntu.com/questions/345562/permanently-remove-item-from-path
export PYTHONPATH=${PYTHONPATH/":/projects/HSC/LSST/stack_20200903/stack/miniconda3-py37_4.8.2-cb4e2dc/Linux64/scarlet/lsst-dev-g52409ed422+f31336177f/lib/python"/""}