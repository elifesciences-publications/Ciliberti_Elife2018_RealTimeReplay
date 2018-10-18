#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:10:16 2018

@author: davide
"""

path_to_preprocessed_dataset =\
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_sleep/2017-08-24_09-36-44.hdf5"

path_to_encoding_model =\
"/home/davide/Dropbox/Davide/dataset_2017_08_24_prerun/model_2.0_threshold"

path_to_environment =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment"

path_to_performance =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/sleep/reference_performance_meter.p"

path_to_prerun_preprocessed_dataset =\
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_prerun/2017-08-24_09-36-44.hdf5"

path_to_grid =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/grid.npy"

path_to_random_indices =\
"/home/davide/Dropbox/Davide/Replay_Analysis/tetrode_degradation_detectability/combo_indices_randomized.npy"

start_random_index = 0
n_max_bootstrap = 50

n_removed = 13


save_values = False
path_to_results = "/home/fklab/Dropbox/Davide/Replay_Analysis/tetrode_degradation_detectability/sleep"
