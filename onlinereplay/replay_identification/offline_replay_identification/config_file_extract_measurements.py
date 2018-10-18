# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:47:03 2015

@author: davide
"""
from os.path import dirname, join, split

path_to_environment =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/encodingmodel_precomputations/23_08_2017/environment"
path_to_preprocessed_dataset = \
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_23_sleep/2017-08-23_09-42-01.hdf5"
path_to_eventdata =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/output/sleep2/20170823_105721/sink_replay/sink_replay.0_replayidentifier.events.0.yaml"

path_to_nlx_eventdata = join( dirname( path_to_preprocessed_dataset ), "Events.nev" )

#replay_string = "TTL Input on AcqSystem1_0 board 0 port 3 value (0x0040)."
#replay_string = "REPLAY"
replay_string = "detection"

# plotting
plot_event_posteriors = False
plot_color_bar = True
plot_line = False
n_horiz_plots = 5
n_vertical_plots = 4
max_n_events = 300
stop_after_plot = False

# validation
n_shuffles = 2

compute_linefit_score_against_shuffles = False
n_cores = 62 # only used if computing linefit score against shuffles

# SAVING
# it will save bias, bias_score, line fit scores and latencies (absolute, relateive, added)
save_measurements = False 
dataset_id = split(split(path_to_preprocessed_dataset)[0])[1]
dataset_epoch = dataset_id[18-len(dataset_id)+1:]
save_measurements_dump = join("/home/davide/Dropbox/Davide/Replay_Analysis",\
  dataset_id[:18],dataset_epoch, "offline_posteriors" )
save_online_content = False
online_content_filepath_dump = dirname( save_measurements_dump )
save_final_content_nodes = False
content_nodes_filepath_dump = save_measurements_dump
save_added_latencies = False
latencies_filepath_dump = join( dirname(save_measurements_dump), "latencies" )