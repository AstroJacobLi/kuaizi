'''
Generete webpage for displaying scarlet results

'''

import os, re
import numpy as np
import kuaizi as kz
from astropy.table import Table


kz.utils.set_env(project='HSC', name='LSBG', data_dir='/tigress/jiaxuanl/Data')
lsbg_cat = Table.read('./Cutout/Candy/candy_cutout_cat.fits')

## get the indices of successful modeling
figlist = os.listdir('/tigress/jiaxuanl/Data/HSC/LSBG/Figure/')
figlist = [item for item in figlist if '-zoomin-wavelet.png' in item] # only select png
ind_array = np.asarray([re.findall('-\d+-', item)[0].strip('-') for item in figlist], dtype=int)
ind_array.sort()
fail_array = np.setdiff1d(np.arange(0, len(lsbg_cat)), ind_array)
print(f'Failed {len(fail_array)} galaxies (based on checking figures):', fail_array)

## Check log files, find out error items
with open('/tigress/jiaxuanl/Data/HSC/LSBG/candy_fail_0_100', 'r') as f:
    log = f.read()
    f.close()    
with open('/tigress/jiaxuanl/Data/HSC/LSBG/candy_fail_100_200', 'r') as f:
    log = f.read()
    f.close()
with open('/tigress/jiaxuanl/Data/HSC/LSBG/candy_fail_200_300', 'r') as f:
    log = f.read()
    f.close()
with open('/tigress/jiaxuanl/Data/HSC/LSBG/candy_fail_complement', 'r') as f:
    log = f.read()
    f.close()

loglist = log.split('\n')
loglist = [item for item in loglist if len(item) > 0]
loglist = [item for item in loglist if 'ERROR' in item] # find out error items
fail_log_array = np.unique(np.array([re.findall('candy_\d+', item)[0].strip('candy_') for item in loglist], dtype=int))
print(f'Failed {len(fail_log_array)} galaxies (based on checking log files):', fail_log_array)


## Copy figures to public_html
os.system('cp /tigress/jiaxuanl/Data/HSC/LSBG/Figure/candy-*-zoomin-*.png /tigress/jiaxuanl/public_html/candy/scarlet_zoomin')

## Count figures in public_html/candy/scarlet_zoomin
print(f'You have {len(ind_array)} galaxies to be displayed')

row_num = 10
page_num = len(lsbg_cat) // (row_num) + 1
print('Total pages:', page_num)

# Write HTML
for k in range(page_num):
    f = open(f'/tigress/jiaxuanl/public_html/candy/scarlet_zoomin/page{k + 1}.html', 'w')
    f.write('<!DOCTYPE html> \n<html><head> \n<link rel="stylesheet" type="text/css" href="../../mystyle.css"> \n</head><body> \n\n')
    f.write('<script src="//code.jquery.com/jquery-1.11.3.min.js"></script>\n')
    f.write('<script type="text/javascript" src="../../myjs.js"></script> \n')
    f.write('<title>Scarlet Modeling Candy Sample</title> \n')
    f.write('\n\n</head><body> \n\n')
    
    f.write('<div class="header"> \n<h1>Candy Sample: "griz" bands, starlet_thresh = 0.5</h1> \n</div> \n\n')
    f.write('<div class="navigator"> \n')
    
    # Write navigator
    f.write('   <a href="page1.html" class="first">First</a> ')
    if k == 0:
        f.write('   <a href="#" class="previous">&laquo; Previous</a> \n')
    else:
        f.write(f'   <a href="page{k}.html" class="previous">&laquo; Previous</a> \n')
    
    f.write(f'   <a href="#" class="current">Page {k + 1}</a> \n')
    
    if k == page_num - 1:
        f.write('   <a href="#" class="next">Next &raquo;</a> \n')
    else:
        f.write(f'   <a href="page{k + 2}.html" class="next">Next &raquo;</a> \n')
    f.write(f'   <a href="page{page_num}.html" class="last">Last</a>')
    
    # Write search box
    f.write('<div class="searchbox"> \n<form id="jumper" method="get" onsubmit="return jumptopage_scarlet()">\n')
    f.write('   <label class="text_index">Index: </label>\n   <input type="text" id="galind" placeholder="123" name="GalaxyIndex">\n')
    f.write('   <button type="submit" class="buttonjump" value="submit">Jump</button>\n   <button type="reset" class="buttonreset">Reset</button>\n ')
    f.write('</form>\n </div>\n')
    
    f.write('</div> \n\n')
    
    for i in range(row_num):
        ind = row_num * k + i
        
        if ind in fail_array:
            f.write(f'<div class="row fail" id="candy{ind}"> \n')
            f.write(f'   <figure><h1>Failed!</h1><figcaption>Candy {ind}</figcaption> </figure>\n')
        elif ind in ind_array:
            f.write(f'<div class="row" id="candy{ind}"> \n')
            f.write(f'   <figure><img src="candy-{ind}-zoomin-wavelet.png" style="width:100%"> <figcaption>Candy {ind}</figcaption> </figure>\n')
        
        f.write(f'</div> \n\n')

    f.write('</body></html> \n')
    f.close()