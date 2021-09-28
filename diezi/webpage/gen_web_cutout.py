'''
Generete webpage for displaying RGB cutouts

'''

import os, re
import numpy as np
import kuaizi as kz
from astropy.table import Table

DATADIR = '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG'
CATALOG_DIR = '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Cutout/Candy/candy_cutout_cat.fits'
FIGURE_DIR = '/tigress/jiaxuanl/public_html/candy/cutout_figure/' # RGB images should be in Tigress

# find existing cutout rgb images
kz.utils.set_env(project='HSC', name='LSBG', data_dir=DATADIR)
lsbg_cat = Table.read(CATALOG_DIR)

figlist = os.listdir(FIGURE_DIR)
figlist = [item for item in figlist if '_cutout.png' in item] # only select png
index_list = [int(re.findall('\d+', item)[0]) for item in figlist]
index_list = np.sort(index_list)
print(f'You have {len(index_list)} galaxies to be displayed')


# Format of webpage
col_num = 7
row_num = 4

page_num = len(index_list) // (col_num * row_num) + 1
print('Total pages:', page_num)

for k in range(page_num):
    f = open(f'/tigress/jiaxuanl/public_html/candy/cutout_figure/page{k + 1}.html', 'w')
    f.write('<!DOCTYPE html> \n<html><head> \n<link rel="stylesheet" type="text/css" href="../../mystyle.css">\n')
    f.write('<script src="//code.jquery.com/jquery-1.11.3.min.js"></script>\n')
    f.write('<script type="text/javascript" src="../../myjs.js"></script> \n')
    f.write('<title>Candy Sample Cutout</title> \n')
    f.write('\n\n</head><body> \n\n')
    
    f.write('<div class="header"> \n<h1>Candy Sample Cutout</h1> \n</div> \n\n')
    f.write('<div class="navigator"> \n')
    
    # Write navigator
    if k == 0:
        f.write('   <a href="#" class="previous">&laquo; Previous</a> \n')
    else:
        f.write(f'   <a href="page{k}.html" class="previous">&laquo; Previous</a> \n')
    
    f.write(f'   <a href="#" class="current">Page {k + 1}</a> \n')
    
    if k == page_num - 1:
        f.write('   <a href="#" class="next">Next &raquo;</a> \n')
    else:
        f.write(f'   <a href="page{k + 2}.html" class="next">Next &raquo;</a> \n')
        
    # Write search box
    f.write('<div class="searchbox"> \n<form id="jumper" method="get" onsubmit="return jumptopage_cutout()">\n')
    f.write('   <label class="text_index">Index: </label>\n   <input type="text" id="galind" placeholder="123" name="GalaxyIndex">\n')
    f.write('   <button type="submit" class="buttonjump" value="submit">Jump</button>\n <button type="reset" class="buttonreset">Reset</button>\n ')
    f.write('</form>\n </div>\n')
    
    f.write('</div> \n\n')
    
    for i in range(col_num * row_num):
        if (i) % col_num == 0 or i == 0:
            f.write('<div class="column"> \n')
        ind = col_num * row_num * k + i
        if ind >= len(index_list):
            f.write('</div> \n')
            break
        page = int(np.ceil((ind + 1) / 20))
        f.write(f'      <figure> <img src="candy_{index_list[ind]}_cutout.png" id="candy{index_list[ind]}">')
        f.write(f'<figcaption><a class="clickable" target="_blank" href="../gpfs_scarlet_zoomin/page{page}.html#candy{index_list[ind]}">Candy {index_list[ind]}</a></figcaption></figure> \n')
        if (i + 1) % 7 == 0:
            f.write('</div> \n\n')
    

    f.write('</body></html> \n')
    f.close()