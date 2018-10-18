# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:51:25 2016

@author: davide
"""

import os

path_to_preprocessed_dataset_folder =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_25_postrun"

path_to_encoding_model =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_25_prerun/model_0.8_threshold"

#path_to_experimental_grid =\
#"/mnt/data3/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/encodingmodel_precomputations/23_08_2017/environment/grid.npy"
#path_to_experimental_grid =\
#"/mnt/data3/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/grid.npy"
path_to_experimental_grid =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/encodingmodel_precomputations/25_08_2017/environment/grid.npy"

min_run_duration = 1
run_speed = 8.5 # cm/s
bin_size_run = 0.2 # s
offset = 1e-9 # Hz

save_decoding_errors = True
path_to_decoding_errors =\
    "/home/fklab/Dropbox/Davide/Replay_Analysis/dataset_2017_08_" +\
    os.path.dirname(path_to_encoding_model)[-9:-7] +\
    "/postrun/errors_with_analysis_compression_threshold/run_decoding_errors_" +\
    path_to_encoding_model[-13:-10] + ".npy"
