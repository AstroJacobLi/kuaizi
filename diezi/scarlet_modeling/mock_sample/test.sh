python3 ../script/sclt_multiprocess.py --njobs 1 \
    --DATADIR "/scratch/gpfs/jiaxuanl/Data/" \
    --OUTPUT_DIR "/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/" \
    --PREFIX "MOCK" \
    --OUTPUT_SUBDIR "MOCK_SAMPLE" \
    --cat_dir "/scratch/gpfs/jiaxuanl/Data/HSC/LSBG/Catalog/mock_sample/mock_obj_cat_0_1000.fits" \
    --low 69 --high 70 \
    --suffix "" --starlet_thresh 1 --method "wavelet" \
    --monotonic True --variance 0.0025 --scales '[0, 1, 2, 3, 4, 5, 6]'

python deploy_mock.py --name mock_wvlt --ncpu=16 --starlet_thresh=1 \
 --method=wavelet --low=0 --high=1 --monotonic=True --variance=0.0025 \
 --scales="[0, 1, 2, 3, 4, 5, 6]" --sigma=0.02