#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:33:00 2017

@author: davide
"""

path_falcon_output =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_09_08_2017/2017-08-09_09-23-57/falcon/output/sleep_postrun"
path_to_falcon_posteriors=\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_09_08_2017/2017-08-09_09-23-57/falcon/output/sleep_postrun/20170809_100204/sink_synch/sink_synch.0_synchronizer.estimates.0.bin"
index_dataset_name = 1
path_to_candidate_events =\
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_09_sleep/candidate_events.p"
path_to_reference_content =\
"/home/davide/Dropbox/Davide/Replay_Analysis/new_datasets/reference_content_4.3bZ_0.1lfs.npy"
path_falcon_mua =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_09_08_2017/2017-08-09_09-23-57/falcon/encodingmodel_precomputations/09_08_2017/mua/mua.npy"
path_to_grid =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_09_08_2017/2017-08-09_09-23-57/falcon/encodingmodel_precomputations/09_08_2017/environment/grid.npy"
path_to_replay_identifier_sink=\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_09_08_2017/2017-08-09_09-23-57/falcon/output/sleep_postrun/20170809_100204/sink_replay/sink_replay.0_replayidentifier.events.0.yaml"


HALF_INTEGRATION_WINDOW = 3 # in grid units

save_simulator = False
output_path =\
"/home/davide/Dropbox/Davide/Replay_Analysis/playback/simulators - new_datasets/half_win_" + str(HALF_INTEGRATION_WINDOW)

# not relevant, just for testing
MUA_THR = 3.
PEAKINESS_thr = 0.5
PEAKINESS_HISTORY_AVG_thr = 0.5 # 0.65(start value)

