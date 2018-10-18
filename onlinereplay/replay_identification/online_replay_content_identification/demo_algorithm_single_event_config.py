#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:41:57 2017

@author: davide
"""

from os.path import join

path_to_nlx_preprocessed_dataset =\
"/home/davide/Data/ReplayDisruption/PreprocessedDatasets/dataset_2018_03_06_delay/2018-03-06_08-38-44.hdf5"
path_to_falcon_dataset =\
"/home/davide/Data/ReplayDisruption/dataset_2018_03_06/2018-03-06_08-38-44/falcon/output/delay/20180306_102402"
path_to_environment =\
"/home/davide/Data/ReplayDisruption/dataset_2018_03_06/2018-03-06_08-38-44/falcon/encodingmodel_precomputations/06_03_2018/environment"
output_path = "/home/davide/Dropbox/Davide/Replay_Analysis/Realtime_ReplayDetection_Videos/FM009"

mua_path = join( path_to_falcon_dataset, "mua.npy" )

CANDIDATE_ID = 120
content = 1 # 0, 1, 2
detection = 0

mua_threshold_std = 3.
sharpness_threshold = 0.3

clamp_posteriors = 0.1

#replay_string = "TTL Input on AcqSystem1_0 board 0 port 3 value (0x0040)."
#replay_string = "REPLAY"
replay_string = "non-target"

extension_s = 0.2

# plot params
fs = 14
padsize = 10
lw = 4
alpha_detections = 0.5
z_order_detections = 20
z_order_thresholds = 10
fig_width = 21
fig_height = 10
ylabel_coord1 = -0.085 #x
ylabel_coord2 = .40 #y

