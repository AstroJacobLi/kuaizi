# Diezi 🥘
Diezi (碟子): Fill a small dish with HSC low surface brightness galaxies

This repo is for the notebook and scripts used for modeling LSBGs in HSC using scarlet.

`setup_env.sh` can be used on `Tiger` and `Tigressdata` to setup the proper environment for the task. Under LSST Pipeline, you can only use Python 3.7.8. 


### Data Structure

Cutouts (and PSFs) with size of 0.6 arcmin in 5 bands takes 11 Mb. 

After pickling, one galaxy in 5 bands takes 7.1 Mb. 