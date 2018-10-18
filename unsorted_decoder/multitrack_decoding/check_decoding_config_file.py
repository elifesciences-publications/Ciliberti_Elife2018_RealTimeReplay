# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:47:03 2015

@author: davide
"""

from os.path import dirname

#fullpath_to_VTdata =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/VT1.nvt"
#fullpath_to_VTdata =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/output/postrun/20170823_111253/sink_vt/sink_vt.0_vt_reader.vt.0.bin"
#path_to_environmentfile =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/encodingmodel_precomputations/23_08_2017/environment/env.yaml"
#path_to_decoded_position =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/output/postrun/20170823_111253/sink_behavior/sink_behavior.0_behavior_estimator.behavior.0.bin"

#fullpath_to_VTdata =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/VT1.nvt"
#fullpath_to_VTdata =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/output/postrun/20170824_110102/sink_vt/sink_vt.0_vt_reader.vt.0.bin"
#path_to_environmentfile =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/env.yaml"
#path_to_decoded_position =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/output/postrun/20170824_110102/sink_behavior/sink_behavior.0_behavior_estimator.behavior.0.bin"

fullpath_to_VTdata =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/VT1.nvt"
#fullpath_to_VTdata =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/output/postrun/20170825_110216/sink_vt/sink_vt.0_vt_reader.vt.0.bin"
path_to_environmentfile =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/encodingmodel_precomputations/25_08_2017/environment/env.yaml"
path_to_decoded_position =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/output/postrun/20170825_110216/sink_behavior/sink_behavior.0_behavior_estimator.behavior.0.bin"


epoch = [] # set to [] if all dataset has be used

path_to_environment = dirname( path_to_environmentfile ) # contains grid.npy and content_nodes.npy

# processing of behavior data (position)
path_length_cm = 110
max_dist_to_path_occlusions = 200 # in pixels
max_occlusion_time = 0.20 # in seconds
max_dist_to_path = 15 # in pixels
extra_distance = 400 # you need to know what was used when building initial model

# data alignement
bin_factor = 1

# data selection
grid_element_size = 4 # in pixels (for info only, not used in any calculation)
min_speed = 8.5 # cm/s for selecting RUN epochs
velocity_bw = 0.15 # sec [not higher than 0.4] .25
min_run_duration = .5 # in sec

# plotting
plot_env = False
plot_dec_behav = True
plot_dec_error = False

# saving
save_errors = False
#path_to_errors =\
#"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_23/postrun/run_decoding_errors.npy"
#path_to_errors =\
#"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/postrun/run_decoding_errors.npy"
path_to_errors =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_25/postrun/run_decoding_errors.npy"
