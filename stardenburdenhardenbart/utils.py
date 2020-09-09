from __future__ import division, print_function
import os
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import Table


def set_env(project='HSC', name='HSC_LSBG'):
    import os
    
    # Master directory
    try:
        data_dir = os.path.join(
            os.getenv('HOME'), 'Research/Data/', project, name)
    except:
        raise Exception("Can not recognize this dataset!")
        
    os.chdir(data_dir)
    
    return data_dir