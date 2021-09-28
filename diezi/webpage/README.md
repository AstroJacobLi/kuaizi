## Generate webpage for displaying results

- `generate_webpage.py`: display RGB **cutout** image of each object. To use this script, check `diezi/gen_cutout/nsa_sample/gen_rgb.ipynb`. Basically call the function like following:

```python

sys.path.append('/home/jiaxuanl/Research/Packages/kuaizi/diezi/webpage/')
from gen_web_cutout import webpage_cutout

#FOR NSA SAMPLE
FIGURE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB/figure'
WEBPAGE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB/'
SCARLET_DIR = '/tigress/jiaxuanl/public_html/candy/gpfs_scarlet_zoomin/'
sample_name = 'NSA'

webpage_cutout(FIGURE_DIR, WEBPAGE_DIR, sample_name, SCARLET_DIR=SCARLET_DIR, 
               col_num=6, row_num=4)
```
