# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:47:03 2015

@author: davide
"""
from os.path import join

arm_length_cm = 110

# decoding
spike_features_filename =\
"/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/falcon/output/20160124_181944/all_spike_amplitudes.npy"
path_to_encoding_model =\
"/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/falcon/encoding_model/test7"

use_adjusted_nodes = False
path_to_adjusted_nodes = "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2016_01_25/sleep/7Oct/adjusted_content_nodes.npy"

# candidate events detection
mua_filepath = "/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/falcon/output/20160124_181944"
mua_thr_std_H = 2.5
mua_thr_std_L = 0.75
min_duration_cand = 45 # [ms]

# replay identification
path_to_environment = join( path_to_encoding_model, "environment" )

# parse online detections from serialized event data out of the replay identifier
path_to_eventdata =\
"/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/falcon/output/20160124_181944/sink_replay/sink_replay.0_replayidentifier.events.0.yaml"
path_to_likelihooddata = \
"/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/falcon/output/20160124_181944/sink_synch/sink_synch.0_synchronizer.estimates.0.bin"
path_to_nlx_eventdata = "/home/davide/Data/RealTimeReplayDetection/FK005/dataset_2016_01_24/post_run/2016-01-24_18-20-20/Events.nev"
#replay_string = "TTL Input on AcqSystem1_0 board 0 port 3 value (0x0040)."
replay_string = "REPLAY"

# plotting
plot_event_posteriors = False
plot_one_detection_per_event = True
plot_color_bar = True
n_horiz_plots = 5
n_vertical_plots = 4
max_n_events = 800

# validation
n_shuffles = 20
p_value_significance = 0.09
min_score = 0
use_loaded_pvalues = False
p_values_filepath = "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2016_01_25/run/10Oct/p_values_0.5L_250sh.npy"
only_tp_latencies = False

# save performance results
save_performance_results = False
path_to_performance_results = "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2016_01_24/run/L_0.75_H_2.5/performance.p"
