# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:39 2016

@author: davide
"""

import numpy as np
import collections
from os.path import join, isfile
from scipy import interpolate
import natsort

from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml
from fklab.segments import Segment as seg

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder

from support_replay_analysis.tools import extract_preprocessed_dataset_fn

import decode_with_existing_model_config as config


def extract_encoding_tt_ids( path_to_model ):
    """
    """
    TT_ids = []
    j = 0
    for i in range(40):
        tt_id = "TT" + str( i+1 )
        filename_mixture = join( config.path_to_encoding_model, tt_id + "_mixture.dat" )
        if isfile( filename_mixture ):
            TT_ids.append( tt_id )
            j += 1
    
    return TT_ids
    
    
def extract_mismatch_tt_ids( small, large ):
    """
    """
    assert( len(large) > len(small) )
    assert( natsort.natsorted(small) == small )
    assert( natsort.natsorted(large) == large )
    mismatches = []
    for i, el in enumerate(large):
        if el not in small:
            mismatches.append( i )
    return mismatches
    
    
# load and check data
behavior, ephys, _ = read_dataset( extract_preprocessed_dataset_fn(\
    config.path_to_preprocessed_dataset_folder ) )
check_ephys_dataset( ephys )
n_tt_dataset = len(ephys)
n_active_channels = np.array( [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ] )
print "Data was loaded and checked"

# select run segments
run = seg.fromlogical( behavior["speed"] > config.run_speed, x=behavior["time"] )
run = run[ run.duration > config.min_run_duration ]
test_binned = run.split( size=config.bin_size_run )

print( "Loading encoding model\n" )
covars = np.load( join( config.path_to_encoding_model, "covars.npy" ) )
spike_ampl_mask_list = np.load( join( config.path_to_encoding_model,\
    "channel_mask.npy" ) )
tt_included = np.load( join( config.path_to_encoding_model,\
    "tetrode_inclusion_mask.npy") )
assert( len(spike_ampl_mask_list) == sum(tt_included) ) 
spike_ampl_mask_list = spike_ampl_mask_list.tolist()
spike_features = [ ephys[key] for key in ephys.keys() ]
          
# extract the ids of the tetrodes used for encoding
TT_ids = extract_encoding_tt_ids( config.path_to_encoding_model )

# if the decoding dataset has LESS tetrodes than the encoding dataset,
# add one or more tetrodes with no amplitudes; anyway, they will be safely excluded
# by the tetrode selection mask
if len(spike_features) < len(tt_included):
    print "decoding dataset has {0} tetrodes less than the encoding dataset." \
    .format( len(tt_included) - len(spike_features) )
    mismatches = extract_mismatch_tt_ids( ephys.keys(), TT_ids )
    tt_included[mismatches] = False
    print "Tetrode inclusion mask was fixed"
    for mm in mismatches:
        del spike_ampl_mask_list[mm]
    for j in np.nonzero(np.logical_not(tt_included))[0]:
        spike_features = np.insert(\
            spike_features, j, {'spike_times':np.zeros(1), 'spike_amplitudes':np.zeros(max(n_active_channels))})
        print j
# if the decoding dataset has MORE tetrodes than the encoding dataset,
# expand the tetrode inclusion mask properly so that the extra tetrodes are
# not used for generating the posteriors
elif len(spike_features) > len(tt_included):
    mismatches = extract_mismatch_tt_ids( TT_ids, ephys.keys() )
    print "decoding dataset has {0} more tetrodes than the encoding dataset." \
    .format( len(spike_features) - len(tt_included) )
    tt_included = np.insert( tt_included, mismatches, False )
    for mm in mismatches:
        spike_ampl_mask_list.insert( mm, np.zeros( max(n_active_channels), dtype='bool' ) )
    print "Tetrode inclusion mask was expanded with {} false elements"\
    .format( len( mismatches) )
  
# load model
training_time = np.load ( join( config.path_to_encoding_model, "T.npy" ) )  
mcd_behav = mcd()
mcd_behav.load_from_mixturefile( join( config.path_to_encoding_model,\
    "mixture_behavior.dat" ) )
mcd_spikebehav = []
with open( join( config.path_to_encoding_model, "model_info.yaml" ), 'r') as f_yaml:
    info_model = yaml.load( f_yaml )
for tt_id in TT_ids:
    filename_mixture = join( config.path_to_encoding_model, tt_id + "_mixture.dat" )
    density = mcd()
    density.load_from_mixturefile( filename_mixture )
    mcd_spikebehav.append( density )
if len(mcd_spikebehav) == 0:
    raise ValueError("No encoding model read!")
else:
    print "\nEncoding model was loaded from " + config.path_to_encoding_model
      
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
print "Decoding {0} run epochs with {1} grid elements " .format( len(run), len(grid) )
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, config.offset )
spike_ampl_mask_list = np.array( [ np.ones( ephys[key]["spike_amplitudes"].shape[1], dtype='bool')\
    for key in ephys.keys() ] )[tt_included]
posterior, logpos, n_spikes, n_tt = decoder.decode(\
    [ ephys[key] for key in ephys.keys() ],\
    test_binned,\
    tt_included,\
    spike_ampl_mask_list,\
    config.bin_size_run,\
    sf_keys=["spike_times", "spike_amplitudes"] )
decoded_behavior = grid[np.nanargmax(logpos, axis=1)].flatten()
    
# compute true behavior
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

# compute and save decoding error     
assert(len(true_behavior) == len(decoded_behavior))
n_test_bins = len(decoded_behavior)
errors = np.array( [np.linalg.norm(pred_i - true_behav_i) \
    for pred_i, true_behav_i in zip(decoded_behavior, true_behavior)])
print "\nCompressed decoding completed with {0} tetrodes done with median error of:\t{1:.2f} cm"\
    .format(n_tt, np.nanmedian(errors))
for a in range(n_tracks):
    print "\nMedian error (arm {0}):\t{1:.3f} cm" .format(a+1, np.median(errors[arm_mask[:,a]]))
if config.save_decoding_errors:
    np.save( open( config.path_to_decoding_errors, 'w'), errors )
    print "Decoding errors saved in " + config.path_to_decoding_errors
    for a in range(n_tracks):
        path = config.path_to_decoding_errors[:-4] + '_arm' + str(a+1) + config.path_to_decoding_errors[-4:]
        np.save( open( path, 'w'), errors[arm_mask[:,a]] )
        print "\nDecoding errors arm {0} saved in {1}". format( a+1, path )
