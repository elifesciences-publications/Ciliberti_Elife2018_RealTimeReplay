# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 19:49:24 2016

@author: davide
"""

path_to_preprocessed_dataset=\
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_prerun/2017-08-24_09-36-44.hdf5"

stopping_speed_cm = 180 # cm/s (18 cm for sleep session gives the same result of 18 cm)
bin_size_mua = .005 # s
smooth_mua_bw = 0 #s
mua_thr_std_H = 2.5
mua_thr_std_L = 0.5 # 0.75
min_duration_cand = 0.010 # s
allowable_gap = 0.020 # s

detrend = False
span_ewma = 750

save_events = False
plot_mua = False

save_mua = True
mua_filename =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/prerun/mua.npy"
mua_times_filename =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/prerun/mua_times.npy"

# only relevant if plotting 
path_to_online_detections =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_04_08_2017/2017-08-04_09-31-48/Events.nev"
detection_string = "detection"