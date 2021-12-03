#!/usr/bin/env python

import fire
import os
import re
import numpy as np
import kuaizi as kz
# from astropy.table import Table
from shutil import copy

# FOR NSA SAMPLE
# FIGURE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB/figure'
# WEBPAGE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB'
# sample_name = 'NSA'


def webpage_cutout(FIGURE_DIR, WEBPAGE_DIR, sample_name, sample_title='', SCARLET_DIR=None, col_num=7, row_num=4):
    '''
    This function writes HTML webpage for displaying RGB cutout images.


    Parameters:
        FIGURE_DIR (str): the directory of the RGB cutout figure,
            such as `FIGURE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB/figure'`.
        WEBPAGE_DIR (str): the directory for the generated HTML file,
            such as `WEBPAGE_DIR = '/tigress/jiaxuanl/public_html/NSA/cutout_RGB/'`.
        sample_name (str): the prefix of the RGB figures, such as 'NSA'.
        SCARLET_DIR (str): the directory of scarlet fitting result figures, default is None.
            If not None, a clickable link will be created.
        col_num (int): number of columns on the webpage.
        row_num (int): number of rows on the webpage.

    Returns:
        None
    '''
    # find existing cutout rgb images
    if not os.path.isdir(WEBPAGE_DIR):
        os.mkdir(WEBPAGE_DIR)
    else:
        import glob
        fileList = glob.glob(WEBPAGE_DIR + '*.html', recursive=True)
        [os.remove(file) for file in fileList];
    
    os.chdir(WEBPAGE_DIR)
    # change to relative path, as required by HTML
    FIGURE_DIR = os.path.relpath(FIGURE_DIR)
    if SCARLET_DIR is not None:
        SCARLET_DIR = os.path.relpath(SCARLET_DIR)

    figlist = os.listdir(FIGURE_DIR)
    # only select png
    figlist = [item for item in figlist if '_cutout.png' in item]
    index_list = [int(re.findall('\d+', item)[0]) for item in figlist]
    index_list = np.sort(index_list)
    print(f'You have {len(index_list)} galaxies to be displayed')

    # Write HTML
    page_num = len(index_list) // (col_num * row_num) + 1
    print('Total pages:', page_num)

    # Copy the css in `kuaizi/diezi/css` to `public_html`
    copy('/home/jiaxuanl/Research/Packages/kuaizi/diezi/webpage/css/myjs.js', WEBPAGE_DIR)
    copy('/home/jiaxuanl/Research/Packages/kuaizi/diezi/webpage/css/mystyle.css', WEBPAGE_DIR)

    for k in range(page_num):
        f = open(os.path.join(WEBPAGE_DIR, f'page{k + 1}.html'), 'w')
        f.write(
            '<!DOCTYPE html> \n<html><head> \n<link rel="stylesheet" type="text/css" href="mystyle.css">\n')
        f.write('<script src="//code.jquery.com/jquery-1.11.3.min.js"></script>\n')
        f.write('<script type="text/javascript" src="myjs.js"></script> \n')
        f.write(
            f'<title>{sample_name.upper()} {sample_title} Sample Cutout</title> \n')
        f.write('\n\n</head><body> \n\n')

        f.write(
            f'<div class="header"> \n<h1>{sample_name.upper()} {sample_title} Sample Cutout</h1> \n</div> \n\n')
        f.write('<div class="navigator"> \n')

        # Write navigator
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
            f.write(
                f'      <figure> <img src="{FIGURE_DIR}/{sample_name.lower()}_{index_list[ind]}_cutout.png" id="{sample_name}{index_list[ind]}">')
            if SCARLET_DIR is not None:
                page = int(np.ceil((ind + 1) / 10))
                f.write(
                    f'<figcaption><a class="clickable" target="_blank" href="{SCARLET_DIR}/page{page}.html#{sample_name}{index_list[ind]}">{sample_name} {index_list[ind]}</a></figcaption></figure> \n')
            if (i + 1) % col_num == 0:
                f.write('</div> \n\n')

        f.write('</body></html> \n')
        f.close()


if __name__ == '__main__':
    fire.Fire(webpage_cutout)
