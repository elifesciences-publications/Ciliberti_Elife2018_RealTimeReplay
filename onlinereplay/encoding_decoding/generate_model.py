# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:32:41 2016

@author: davide

This script validates and generates an encoding model from an input dataset

"""
from __future__ import division
import numpy as np
from scipy import interpolate
import collections
from matplotlib import pyplot as plt
import os
from os.path import dirname, join
import time

from fklab.segments import Segment as seg
from fklab.decoding import prepare_decoding as prep_dec
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder, create_covariance_array, save_mixture

from support_replay_analysis.tools import boundaries, remove_added_distance,\
    extract_preprocessed_dataset_fn

import config_file as config


# load and check data
behavior, ephys, _ = read_dataset( extract_preprocessed_dataset_fn( \
                       config.path_to_preprocessed_dataset_folder ) )
check_ephys_dataset( ephys )
if config.use_falcon_spikes:
    falcon_spikes = np.load( config.path_to_falcon_spikes )
    ephys = collections.OrderedDict()
    for i, tt in enumerate(falcon_spikes):
        ephys["TT"+str(i+1)] = {"spike_amplitudes" : tt['value'],\
              "spike_times" : tt['time']}
    print "Ephys data from Neuralynx was overwritten with Falcon data"
n_tt_dataset = len(ephys)
n_active_channels = np.array( [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ] )
print "Data was loaded and checked"

# select run segments
run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
run = run[ run.duration > config.min_run_duration ]
# select test and train partitions
test, train = prep_dec.partition( run, config.n_partitions,\
    partition_index=config.partition_index, fold=config.fold,\
    method=config.partition_method, coincident=False )
test_binned = test.split( size=config.bin_size_run )
print "Data was partitioned"

# reduce train data
if config.max_training_time is not None:
    while ( train.duration.sum() > config.max_training_time*60 ):
        last_idx = len(train) - 1
        del train[last_idx]
    print "Training data reduced to {0} s" .format(train.duration.sum())
else:
     print "Training data unmodified (epoch duration: {0:.1f} s)" .format(\
         train.duration.sum())
     
# prepare encoding
train_behav = prep_dec.extract_train_behav( train, behavior )
train_spike, tetrode_inclusion_mask =\
    prep_dec.extract_train_spike( train, ephys, config.min_n_encoding_spikes )
covars = create_covariance_array( config.behav_bw_cm, config.spf_bw_mV )
# created encoding points
encoding_points = [ prep_dec.attach( train_behav, tt ) for tt in train_spike ] 
# create compressed (empty) joint density with all tetrodes (even if they have too few spikes)
mcd_spikebehav = []
id_selected_sensors = np.array(ephys.keys())[tetrode_inclusion_mask].tolist()
for i in range(n_tt_dataset):
    cov_tt = covars[:n_active_channels[i]+1]
    mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
        cov_tt, method='bandwidth', threshold=config.compression_threshold,\
        name=(ephys.keys()[i] ) ) )

# fill joint compressed density with encoding points
for i, dec in enumerate(mcd_spikebehav):
    if tetrode_inclusion_mask[i]:
        points = encoding_points[i]
        dec.addsamples( points )
        print "Encoding points " + ephys.keys()[i] + " added"

# uncompressed density of behavior data
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
print "Decoding grid prepared"

# decode during test run epochs
print "I'm decoding {0} run epochs with {1} grid elements " .format( len(test), len(grid) )
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, config.offset )
print "Decoders created"
spike_ampl_mask_list = np.array( [ np.ones( ephys[key]["spike_amplitudes"].shape[1], dtype='bool')\
    for key in ephys.keys() ] )[tetrode_inclusion_mask]
t0 = time.time()
posterior, logpos, n_spikes, n_tt = decoder.decode(\
    [ ephys[key] for key in ephys.keys() ],\
    test_binned,\
    tetrode_inclusion_mask,\
    spike_ampl_mask_list,\
    config.bin_size_run,\
    sf_keys=["spike_times", "spike_amplitudes"] )
decoded_behavior = grid[np.nanargmax(logpos, axis=1)].flatten()
t1 = time.time()
decoding_time = t1 - t0
print "\nDecoding time: {0:f} s, {1:.4f} ms/spike\n" .format( decoding_time, decoding_time/np.sum(n_spikes) * 1e3 )
print "Total #spikes = {0}" .format(np.sum(n_spikes))


# compute decoding error
true_behavior = interpolate.interp1d( behavior["time"], behavior["linear_position"],\
    kind='linear', axis=0 ) ( test_binned.center )       
assert(len(true_behavior) == len(decoded_behavior))
n_test_bins = len(decoded_behavior)
errors = np.array( [np.linalg.norm(pred_i - true_behav_i) \
    for pred_i, true_behav_i in zip(decoded_behavior, true_behavior)])
print "\nCompressed decoding completed with {0} tetrodes done with median error of:\t{1:.2f} cm"\
    .format(n_tt, np.nanmedian(errors))
print "\nThreshold used for compression : {0:.2f}" .format( config.compression_threshold )
print "\nGrid element size : {0} cm" .format( config.grid_element_size_cm )


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
for a in range(n_tracks):
    print "\nMedian error (arm {0}):\t{1:.3f} cm" .format(a+1, np.median(errors[arm_mask[:,a]]))


# save model
if config.save_model:
    
    if not os.path.isdir( config.path_to_encoding_model ):
        os.mkdir( config.path_to_encoding_model )
    
    if config.n_partitions != 1 or config.compression_threshold > 0:
        print "\nModel was not computed using all available information!"
    
    np.save( open( join( config.path_to_encoding_model, "covars.npy" ), "w"), covars )
    np.save( open( join( config.path_to_encoding_model, "T.npy" ), "w" ), training_time )
    np.save (open ( join( config.path_to_encoding_model, "tetrode_inclusion_mask.npy" ), "w"),\
        tetrode_inclusion_mask)
    np.save( open( join( config.path_to_encoding_model, "channel_mask.npy" ), "w" ),\
        spike_ampl_mask_list )
    np.save( open( join( config.path_to_encoding_model, "pix.npy" ), "w" ), decoder.pix() )  
    np.save( open( join( config.path_to_encoding_model, "grid_cm.npy" ), 'w'), grid.ravel() )
    save_mixture( mcd_behav, join( config.path_to_encoding_model, "mixture_" +\
        mcd_behav.name + ".dat" ) )
    for i, density in enumerate(mcd_spikebehav):
        save_mixture( density, join( config.path_to_encoding_model, density.name +\
            "_mixture" + ".dat" ) )
        np.save( open( join( config.path_to_encoding_model, density.name + "_lx" + \
            ".npy" ), "w"), decoder.lx()[i] )
        np.save( open( join( config.path_to_encoding_model, density.name + "_mu" + \
            ".npy" ), "w"), decoder.mu[i] )
    with open( join( config.path_to_encoding_model, "model_info.yaml" ), 'w' ) as yaml_file:
        d = {'threshold_compression': config.compression_threshold,
            'grid_size_cm': config.grid_element_size_cm,
            'n_test_epochs' : len(test),
            'offset' : config.offset,
            'min_speed_run' : config.run_speed,
            'min_run_duration' : config.min_run_duration,
            'min_n_encoding_spikes_per_tt' : config.min_n_encoding_spikes,
            'n_partitions' : config.n_partitions,
            'use_corrected_positions' : config.use_corrected_positions }
        yaml.dump( d, yaml_file, default_flow_style=False )
            
    print "\nEncoding model saved in " + config.path_to_encoding_model
        

if config.use_corrected_positions:
    low_boundaries, high_boundaries = boundaries( true_behavior, added_distance_cm,\
        n_tracks )     
    true_behavior = remove_added_distance( true_behavior, added_distance_cm,\
        n_tracks, low_boundaries, high_boundaries )
    decoded_behavior = remove_added_distance( decoded_behavior, added_distance_cm,\
        n_tracks, low_boundaries, high_boundaries )
        
# plot train and test behavior
if config.plot_train_test:

    train_behav_plot = train_behav[0]
    test_behavior_plot = behavior["linear_position"][test.contains(behavior["time"])[0]]
    if config.use_corrected_positions:
        low_boundaries, high_boundaries = boundaries( train_behav[0], added_distance_cm,\
            n_tracks )     
        train_behav_plot = remove_added_distance( train_behav[0], added_distance_cm,\
            n_tracks, low_boundaries, high_boundaries )
        test_behavior_plot = remove_added_distance(\
            behavior["linear_position"][test.contains(behavior["time"])[0]],\
            added_distance_cm, n_tracks, low_boundaries, high_boundaries )
    
    fig2, ax1 = plt.subplots( nrows=2, ncols=1,figsize=(20, 12) )
    ax1[0].plot( train_behav_plot, linewidth=3 )
    ax1[0].set_title("train behavior")
    ax1[0].set_xlabel("behavior samples")
    ax1[0].set_ylabel("linear position [cm]")
    ax1[0].grid()
    ax1[1].plot( test_behavior_plot, linewidth=3  )
    ax1[1].set_xlabel("behavior samples")
    ax1[1].set_title("test behavior")
    ax1[1].set_ylabel("linear position [cm]")
    ax1[1].grid()
    
# plot decoded and true position during test run epochs
if config.plot_dec_behav:
    
    time_points = np.arange(0, len(true_behavior))
    
    fig3 = plt.figure(3, figsize=(30,5))
    plt.clf()
    plt.plot( time_points, true_behavior, color='black', linewidth=2 )
    plt.plot(time_points, decoded_behavior, color='red', linewidth=2 )
    plt.figtext(0.25, 0.96, "true position", fontsize='xx-large',\
        color='black', ha ='right', weight='bold')
    plt.figtext(0.50, 0.96, "decoded position", fontsize='xx-large',\
        color='red', ha ='right', weight='bold')
    plt.yticks(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.grid()
    
if config.plot_dec_error:
    fig4 = plt.figure(4)
    plt.boxplot( errors[np.logical_not( np.isnan(errors))] )
    plt.ylim(ymax = 30)
    plt.ylabel("decoding error [cm]", fontsize='xx-large',  weight='bold')
    plt.xticks([""])
    plt.yticks(fontsize="xx-large", weight = 'bold')
    
    fig5 = plt.figure(5) # plot the cumulative function
    values, base = np.histogram( errors[np.logical_not( np.isnan(errors))], bins=400 )
    cumulative = np.double( np.cumsum(values) )
    plt.plot(base[:-1], cumulative/cumulative.max(), c='blue')
    
if config.plot_occupancy:
    fig7 = plt.figure(7, figsize=(10,4))
    plt.title("occupancy")
    fig5 = plt.plot(decoder.pix())

if config.save_decoded_behavior:
    np.save( open( config.path_to_decoded_behavior, 'w'), decoded_behavior )
    print "Decoded behavior saved in " + config.path_to_decoded_behavior
    
if config.save_decoding_errors:
    if config.max_training_time is None:
        np.save( open( config.path_to_decoding_errors, 'w'), errors )
        print "Decoding errors saved in " + config.path_to_decoding_errors
        for a in range(n_tracks):
            path = config.path_to_decoding_errors[:-4] + '_arm' + str(a+1) + config.path_to_decoding_errors[-4:]
            np.save( open( path, 'w'), errors[arm_mask[:,a]] )
            print "\nDecoding errors arm {0} saved in {1}". format( a+1, path )
    else:
        path_to_reduced_training = join( dirname(config.path_to_decoding_errors),\
            "max_training_time_" + str(int(config.max_training_time*60)) + "s.npy" )
        np.save( open( path_to_reduced_training, 'w'), errors )
        print "Decoding errors saved in " + config.path_to_decoding_errors
        for a in range(n_tracks):
            path = path_to_reduced_training[:-4] + '_arm' + str(a+1) + path_to_reduced_training[-4:]
            np.save( open( path, 'w'), errors[arm_mask[:,a]] )
            print "\nDecoding errors arm {0} saved in {1}". format( a+1, path )
    

plt.show()
