python3 ../script/sclt_meas_mp.py \
    --DATADIR '/scratch/gpfs/jiaxuanl/Data' \
    --OUTPUT_DIR '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/' \
    --OUTPUT_SUBDIR 'MOCK_SAMPLE' \
    --PREFIX 'MOCK' \
    --cat_dir '/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_1000.fits' \
    --filename '_lsbg_mp_meas.fits' \
    --low 0 --high None --method 'wavelet' --ncpu=16