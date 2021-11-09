'''
Generete webpage for displaying scarlet results

'''

import os
import re
import numpy as np
import kuaizi as kz
from shutil import copy
from astropy.table import Table


def webpage_scarlet(CATALOG_DIR, FIGURE_DIR, WEBPAGE_DIR, sample_name, ind_name, suffix='wavelet', row_num=10):
    lsbg_cat = Table.read(CATALOG_DIR)
    lsbg_cat.sort(ind_name)

    # get the indices of successful modeling
    figlist = os.listdir(FIGURE_DIR)
    # only select png
    figlist = [item for item in figlist if f'-zoomin-{suffix}.png' in item]
    ind_array = np.asarray([re.findall('-\d+-', item)[0].strip('-')
                            for item in figlist], dtype=int)
    ind_array.sort()
    fail_array = np.setdiff1d(lsbg_cat[ind_name], ind_array)
    print(
        f'Failed {len(fail_array)} galaxies (based on checking figures):', fail_array)

    #kz.utils.set_env(project='HSC', name='LSBG', data_dir='/tigress/jiaxuanl/Data')

    # Check log files, find out error items
    with open(f'/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/{sample_name}_fail', 'r') as f:
        log = f.read()
        f.close()
    loglist = log.split('\n')
    loglist = [item for item in loglist if len(item) > 0]
    # find out error items
    loglist = [item for item in loglist if 'ERROR' in item]
    fail_log_array = np.unique(np.array(
        [re.findall(f'{sample_name}_\d+', item)[0].strip(f'{sample_name}_') for item in loglist], dtype=int))
    print(
        f'Failed {len(fail_log_array)} galaxies (based on checking log files):', fail_log_array)

    # Copy figures to public_html
    if not os.path.exists(WEBPAGE_DIR):
        os.mkdir(WEBPAGE_DIR)

    if not os.path.isdir(os.path.join(WEBPAGE_DIR, 'figure')):
        os.mkdir(os.path.join(WEBPAGE_DIR, 'figure'))

    for file in figlist:
        copy(
            os.path.join(FIGURE_DIR, file), os.path.join(WEBPAGE_DIR, 'figure'))

    # Count figures in public_html/sample_name/scarlet_zoomin
    print(f'You have {len(ind_array)} galaxies to be displayed')

    page_num = len(lsbg_cat) // (row_num) + 1
    print('Total pages:', page_num)

    # Copy the css in `kuaizi/diezi/css` to `public_html`
    copy('/home/jiaxuanl/Research/Packages/kuaizi/diezi/webpage/css/myjs.js', WEBPAGE_DIR)
    copy('/home/jiaxuanl/Research/Packages/kuaizi/diezi/webpage/css/mystyle.css', WEBPAGE_DIR)

    # Write HTML
    os.chdir(WEBPAGE_DIR)
    # Now FIGURE_DIR is the directory for figures in public_html
    FIGURE_DIR = os.path.join(WEBPAGE_DIR, 'figure')
    # change to relative path, as required by HTML
    FIGURE_DIR = os.path.relpath(FIGURE_DIR)

    for k in range(page_num):
        f = open(os.path.join(WEBPAGE_DIR, f'page{k + 1}.html'), 'w')
        f.write('<!DOCTYPE html> \n<html><head> \n<link rel="stylesheet" type="text/css" href="mystyle.css"> \n</head><body> \n\n')
        f.write('<script src="//code.jquery.com/jquery-1.11.3.min.js"></script>\n')
        f.write('<script type="text/javascript" src="myjs.js"></script> \n')
        f.write(
            f'<title>Scarlet Modeling {sample_name.upper()} Sample</title> \n')
        f.write('\n\n</head><body> \n\n')

        f.write(
            f'<div class="header"> \n<h1>{sample_name.upper()} Sample: "griz" bands, starlet_thresh = 0.5</h1> \n</div> \n\n')
        f.write('<div class="navigator"> \n')

        # Write navigator
        f.write('   <a href="page1.html" class="first">First</a> ')
        if k == 0:
            f.write('   <a href="#" class="previous">&laquo; Previous</a> \n')
        else:
            f.write(
                f'   <a href="page{k}.html" class="previous">&laquo; Previous</a> \n')

        f.write(f'   <a href="#" class="current">Page {k + 1}</a> \n')

        if k == page_num - 1:
            f.write('   <a href="#" class="next">Next &raquo;</a> \n')
        else:
            f.write(
                f'   <a href="page{k + 2}.html" class="next">Next &raquo;</a> \n')
        f.write(f'   <a href="page{page_num}.html" class="last">Last</a>')

        # Write search box
        f.write('<div class="searchbox"> \n<form id="jumper" method="get" onsubmit="return jumptopage_scarlet()">\n')
        f.write('   <label class="text_index">Index: </label>\n   <input type="text" id="galind" placeholder="123" name="GalaxyIndex">\n')
        f.write('   <button type="submit" class="buttonjump" value="submit">Jump</button>\n   <button type="reset" class="buttonreset">Reset</button>\n ')
        f.write('</form>\n </div>\n')

        f.write('</div> \n\n')

        for i in range(row_num):
            if row_num * k + i >= len(lsbg_cat):
                break
            else:
                ind = lsbg_cat[ind_name][row_num * k + i]

            if ind in fail_array:
                f.write(f'<div class="row fail" id="{sample_name}{ind}"> \n')
                f.write(
                    f'   <figure><h1>Failed!</h1><figcaption>{sample_name.upper()} {ind}</figcaption> </figure>\n')

            elif ind in ind_array:
                f.write(
                    f'<div class="row" id="{sample_name.upper()}{ind}"> \n')
                caption = ''
                # caption = f'{sample_name.upper()} {ind} ()'
                gal = lsbg_cat[lsbg_cat['viz-id'] == ind][0]
                if gal['is_galaxy'] >= 1:
                    caption += f'<p style="color:red;">{sample_name.upper()} {ind} Galaxy</p>'
                if gal['is_candy'] >= 1:
                    caption += f'<p style="color:green;">{sample_name.upper()} {ind} Candy</p>'
                if gal['is_junk'] + gal['is_cirrus'] + gal['is_outskirts'] + gal['is_tidal'] >= 1:
                    caption += f'<p style="color:black;">{sample_name.upper()} {ind} Junk</p>'
                if gal['good_votes'] + gal['bad_votes'] == 0:
                    caption += f'<p style="color:orange;">{sample_name.upper()} {ind} No votes!</p>'
                f.write(
                    f'   <figure><img src="{FIGURE_DIR}/{sample_name.lower()}-{ind}-zoomin-{suffix}.png" style="width:100%"> <figcaption>{caption}</figcaption> </figure>\n')

            f.write(f'</div> \n\n')

        f.write('</body></html> \n')
        f.close()
