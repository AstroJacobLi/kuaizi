1. We generate cutout using code in `cutout/nsa_sample`. In the end, we get catalog at `$gpfs/Data/HSC/LSBG/Catalog/NSA/z002_004/nsa_cutout_cat_z002_004.fits`. Cutout images are saved at `$gpfs/Data/HSC/LSBG/Cutout/NSA/`.

2. We run scarlet using sbatch `vanilla_multiprocess.slurm`

3. We move figures to `public_html` and generate webpage