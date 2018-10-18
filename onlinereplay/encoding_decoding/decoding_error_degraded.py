# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:32:41 2016

@author: davide

This script validates and generates an encoding model from an input dataset

"""
from __future__ import division
import numpy as np
from scipy import interpolate
import itertools
from scipy import special
from os.path import join

from fklab.segments import Segment as seg
from fklab.decoding import prepare_decoding as prep_dec
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder, create_covariance_array

from support_replay_analysis.tools import extract_preprocessed_dataset_fn

import decoding_error_degraded_config_file as config


# load and check data
behavior, ephys_orig, _ = read_dataset( extract_preprocessed_dataset_fn( \
                       config.path_to_preprocessed_dataset_folder ) )
check_ephys_dataset( ephys_orig )
n_tt_dataset = len(ephys_orig)
print "Data was loaded and checked"

print "\n{0} tetrodes excluded" .format(config.n_removed)

# select run segments
run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
run = run[ run.duration > config.min_run_duration ]
# select test and train partitions
test, train = prep_dec.partition( run, config.n_partitions,\
    partition_index=config.partition_index, fold=config.fold,\
    method=config.partition_method, coincident=False )
test_binned = test.split( size=config.bin_size_run )
print "Data was partitioned"

train_behav = prep_dec.extract_train_behav( train, behavior )
# uncompressed density of behavior data
covars = create_covariance_array( config.behav_bw_cm, config.spf_bw_mV )
mcd_behav = mcd( train_behav[0], sample_covariance=covars[:1], threshold=0,\
    name='behavior' )
training_time = np.sum( train.duration )
print "Encoding prepared"
        
# define decoding grid
with open( join( config.path_to_preprocessed_dataset_folder,\
"preprocessing_info.yaml" ), 'r' ) as f_yaml:
    info_preprocess = yaml.load( f_yaml )
pixel_to_cm = info_preprocess["ConversionFactorPixelsToCm"]
added_distance_cm = info_preprocess["AddedDistance_pixel"] / pixel_to_cm
n_tracks = info_preprocess["NumTracks"]
if config.path_to_experimental_grid is None:
    half_grid_size_cm = config.grid_element_size_cm / 2
    xgrid_vector = np.arange( np.nanmin( behavior["linear_position"] ) + half_grid_size_cm,\
        np.nanmax( behavior["linear_position"] ) - half_grid_size_cm, config.grid_element_size_cm )
    grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
else:
    grid_pixel = np.load( config.path_to_experimental_grid )
    grid_pixel = grid_pixel.reshape( (len(grid_pixel),1) )
    grid = grid_pixel / pixel_to_cm

true_behavior = interpolate.interp1d( behavior["time"], behavior["linear_position"],\
        kind='linear', axis=0 ) ( test_binned.center ) 

# compute borders of linearized distance (to segment arms) and errors per arm
borders = np.zeros( n_tracks-1 )
borders[0] = (added_distance_cm - info_preprocess['ArmLength_cm'])/2 +\
    info_preprocess['ArmLength_cm']
for b in range(1, n_tracks-1):
    borders[b] = borders[b-1] + added_distance_cm
all_borders = np.zeros( n_tracks+1 )
all_borders[1:n_tracks] = borders
all_borders[n_tracks] = np.nanmax( true_behavior ) + 1
arm_mask = np.zeros( (len(true_behavior), n_tracks), dtype='bool' )
for t in range(n_tracks):
    arm_mask[:,t] = np.logical_and( true_behavior>=all_borders[t],\
            true_behavior<all_borders[t+1]) 

combo_indices_all = np.load( config.path_to_random_indices )
n_combos = int(special.binom( n_tt_dataset, config.n_removed ))
n_bootstrap = min(n_combos, config.n_max_bootstrap )        
first = config.start_random_index
last = config.start_random_index+n_bootstrap
combo_indices = combo_indices_all[config.n_removed][first:last]

errors = []
errors_per_arm = []

j = 0


for c,combo in enumerate(itertools.combinations( range(n_tt_dataset), config.n_removed )):
    
    if c not in combo_indices:
        continue

    ephys = ephys_orig.copy()
    
    for tt in [ ephys.keys()[tt] for tt in combo ]:
        ephys.pop( tt )
        
    n_tt = len(ephys)

    n_active_channels = np.array( [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
        for key in ephys.keys() ] )

    # prepare encoding
    train_spike, tetrode_inclusion_mask =\
        prep_dec.extract_train_spike( train, ephys, config.min_n_encoding_spikes )
    encoding_points = [ prep_dec.attach( train_behav, tt ) for tt in train_spike ] 
    
    # create compressed (empty) joint density with all tetrodes (even if they have too few spikes)
    mcd_spikebehav = []
    id_selected_sensors = np.array(ephys.keys())[tetrode_inclusion_mask].tolist()
    for i in range(n_tt):
        cov_tt = covars[:n_active_channels[i]+1]
        mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
            cov_tt, method='bandwidth', threshold=config.compression_threshold,\
            name=(ephys.keys()[i] ) ) )
    
    # fill joint compressed density with encoding points
    for i, dec in enumerate(mcd_spikebehav):
        if tetrode_inclusion_mask[i]:
            points = encoding_points[i]
            dec.addsamples( points )
    
    decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, config.offset )
    spike_ampl_mask_list = np.array( [ np.ones( ephys[key]["spike_amplitudes"].shape[1], dtype='bool')\
        for key in ephys.keys() ] )[tetrode_inclusion_mask]
    _, logpos, _, _ = decoder.decode(\
        [ ephys[key] for key in ephys.keys() ],\
        test_binned,\
        tetrode_inclusion_mask,\
        spike_ampl_mask_list,\
        config.bin_size_run,\
        sf_keys=["spike_times", "spike_amplitudes"] )
    decoded_behavior = grid[np.nanargmax(logpos, axis=1)].flatten()    
          
    if j==0: assert(len(true_behavior) == len(decoded_behavior))
    n_test_bins = len(decoded_behavior)
    errors.append( np.array( [np.linalg.norm(pred_i - true_behav_i) \
        for pred_i, true_behav_i in zip(decoded_behavior, true_behavior)]) )
            
    j += 1
    
    if j%10 == 0:
        print "\n\t{0} % completed" .format(int(j/n_bootstrap*100))
        
    
if config.save_decoding_errors:
    fname = join( config.path_to_decoding_errors, "errors_" +\
        str(config.n_removed) + "removed_" + str(first) + "_" + str(last) + "bs.npy")
    np.save( open( fname, 'w'), errors )
    np.save( open( join( config.path_to_decoding_errors, "arm_mask.npy"), 'w'), arm_mask )
    print "Decoding errors saved in " + config.path_to_decoding_errors
        
    
