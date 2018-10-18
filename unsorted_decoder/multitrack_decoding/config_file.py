# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:47:03 2015

@author: davide
"""
import time
from os.path import join

# most frequently changed params
path_length_cm = 110
compression_threshold = 1.3

n_partitions = 2 # at least 2, must be even
partition_index = 1 # must be <= n_partitions/2 when n_partitions >2
fold =0 # 0 or 1
min_n_spikes = 5
bin_size_run = 0.2 # in seconds


# can be a NVT file or BIN file generated in Falcon
fullpath_to_VTdata =\
"/mnt/data1/FalconOutput/ContentSpecificReplayDisruption/encoding/_last_run_group/_last_run/sink_vt/sink_vt.0_vt_reader.vt.0.bin"
spike_features_filename =\
"/mnt/data1/FalconOutput/ContentSpecificReplayDisruption/encoding/_last_run_group/_last_run/all_spike_amplitudes.npy"
path_to_encoding_models =\
"/home/fklab/Dropbox/ReplayDisruption/encoding_models/FK026"
path_to_daily_model = join( path_to_encoding_models, time.strftime("%d_%m_%Y") )
path_to_environment = join( path_to_daily_model, "environment" )
path_to_environmentfile = join( path_to_environment, "env.yaml" )
path_to_room_image = join( path_to_environment, "room.jpg" )

# processing of behavior data (position)
max_dist_to_path_occlusions = 200 # in pixels
max_occlusion_time = 0.20 # in seconds
max_dist_to_path = 15 # in pixels
extra_distance = 400
content_node_extension = 3 # for content identification of grid points

# data selection
min_speed = 8.5 # cm/s for selecting RUN epochs
velocity_bw = 0.2 # sec [not higher than 0.4] .25
speed_from_velocity = False
min_run_duration = .5 # in sec
partition_method = 'block' # 'sequence' or 'block'


# encoding - decoding
grid_element_size = 4 # in pixels
behav_bw_cm = 8.0 #cm
spf_bw = 0.03 # mV
offset = 1e-9 # Hz

# saving 
save_encoding_model = True
save_environment = True # grid and content nodes
unit_grid = "pixel"
save_linear_d_single_track = False
#path_to_processed_behavior =\
#"/media/fklab/_data1/EncodingModels/test2/processed_behavior/linear_d_single_tracks.npy"

# plotting
plot_env = True
plot_train_test = True
plot_dec_behav = True
plot_dec_error = True
plot_posterior_run = False
plot_occupancy = True
