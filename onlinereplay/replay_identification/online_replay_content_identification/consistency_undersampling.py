# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:39 2016

@author: davide
"""
from __future__ import division
import numpy as np
from os.path import join, dirname
import cPickle as pickle
import itertools
from scipy import special
import time
import support_replay_analysis.tools as tools

from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml
from fklab.segments import Segment as Seg

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder

import consistency_undersampling_config as config

BIN_SIZE_REPLAY = 0.01
HISTORY = 3

# read data
_, ephys, startstop = read_dataset( config.path_to_preprocessed_dataset ) 
check_ephys_dataset( ephys )
n_tt_dataset = len(ephys)
n_active_channels = [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ]

# load preprocessed data info
with open( join( dirname( config.path_to_preprocessed_dataset ),\
"preprocessing_info.yaml" ), 'r' ) as prepr_f:
    info_preprocess = yaml.load( prepr_f )

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
TT_ids = tools.extract_encoding_tt_ids( config.path_to_encoding_model )

# if the decoding dataset has LESS tetrodes than the encoding dataset,
# add one or more tetrodes with no amplitudes; anyway, they will be safely excluded
# by the tetrode selection mask
if len(spike_features) < len(tt_included):
    print "decoding dataset has {0} tetrodes less than the encoding dataset." \
    .format( len(tt_included) - len(spike_features) )
    mismatches = tools.extract_mismatch_tt_ids( ephys.keys(), TT_ids )
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
    mismatches = tools.extract_mismatch_tt_ids( TT_ids, ephys.keys() )
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
    
# load environment and candidate events
xgrid_vector = np.load( join( config.path_to_encoding_model, "grid_cm.npy" ) )
grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
candidate = pickle.load( open( join( dirname( config.path_to_preprocessed_dataset ),\
    "candidate_events.p"), 'r') )
grid_element_length = np.mean( np.diff( grid.ravel() ) )
content_nodes = np.load( join ( config.path_to_environment, "content_nodes.npy" ) )
n_contents = len(content_nodes)
new_content_nodes, equalized_content_nodes = tools.fix_content_nodes(\
    content_nodes, equalize=True,\
    arm_length_grid_units=np.int(info_preprocess["ArmLength_cm"]/grid_element_length+1 ) ) 
adjusted_grid = grid[:new_content_nodes[-1, -1]+1] 
grid_size = len(adjusted_grid)

# add two bin to each candidate
candidate_ext = Seg.fromevents(candidate.start-(HISTORY-1)*BIN_SIZE_REPLAY, candidate.stop)

# load performance meter
pm = pickle.load( open( config.path_to_performance, 'r') )
assert( len(candidate_ext) == pm.n_events )

# separate non-candidate and with and without content events
non_candidate_ext = Seg.fromarray( [startstop[0], startstop[1]]).difference( candidate_ext )
candidate_ext_with = candidate_ext[np.logical_or(pm.is_joint, pm.is_single)]
candidate_ext_without = candidate_ext[~np.logical_or(pm.is_joint, pm.is_single)]

epoch_split = Seg.fromarray( startstop ).split( BIN_SIZE_REPLAY )
candidate_ext_split_with = candidate_ext_with.split( BIN_SIZE_REPLAY )
candidate_ext_split_without = candidate_ext_without.split( BIN_SIZE_REPLAY )
non_candidate_ext_split = non_candidate_ext.split( size=BIN_SIZE_REPLAY )

n_bins_with = len(candidate_ext_split_with)
n_bins_without = len(candidate_ext_split_without)
n_bins_outside = len(non_candidate_ext_split)

print "\n{0} tetrodes excluded" .format(config.n_removed)

# compute posteriors of the candidate event with decreasing number of tetrodes
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, info_model["offset"] )
print "\nDecoding entire dataset"
_ = decoder.decode( spike_features,
                epoch_split, tt_included,\
                spike_ampl_mask_list, BIN_SIZE_REPLAY,\
                sf_keys=["spike_times", "spike_amplitudes"] )[0]

n_tt = sum(tt_included)

win_half_low, win_half_high = tools.win_halves( grid_size, 6 )

decoding_machine = tools.DecodingMachine( tt_included, spike_ampl_mask_list, decoder, spike_features,\
    equalized_content_nodes, grid_size, win_half_low, win_half_high, BIN_SIZE_REPLAY)
## bug with lazy_decode: bins cannot be  changed. This is workaround
#decoding_machine2 = tools.DecodingMachine( tt_included, spike_ampl_mask_list, decoder, spike_features,\
#    equalized_content_nodes, grid_size, win_half_low, win_half_high, BIN_SIZE_REPLAY)

print "\nDecoding machine initiated"

assert( config.n_removed < n_tt )

n_combos = int(special.binom( n_tt, config.n_removed ))

n_bootstrap = min(n_combos, config.n_max_bootstrap )

consistency_ratio_with = np.zeros( n_bootstrap )
consistency_ratio_without = np.zeros( n_bootstrap )
consistency_ratio_outside = np.zeros( n_bootstrap )

# load random combinations (without replacement) of indices for tetrode selection
combo_indices_all = np.load( config.path_to_random_indices )
first = config.start_random_index
last = config.start_random_index+n_bootstrap
combo_indices = combo_indices_all[config.n_removed][first:last]

start_time = time.time()
j = 0

for i,combo in enumerate(itertools.combinations( range(n_tt), config.n_removed )):
    
    if i not in combo_indices:
        continue

    consistency_ratio_with[j] = decoding_machine.compute_consistency_combo(\
        combo, candidate_ext_split_with, lazy_decode=True )
    consistency_ratio_without[j] = decoding_machine.compute_consistency_combo(\
        combo, candidate_ext_split_without, lazy_decode=True )
    consistency_ratio_outside[j] = decoding_machine.compute_consistency_combo(\
        combo, non_candidate_ext_split, lazy_decode=True )
    
    j += 1
    
    if j%10 == 0:
        print "\n\t{0} % completed" .format(int(j/n_bootstrap*100))
    
print("--- %s seconds ---" % (time.time() - start_time))
      
      
if config.save_values:
    
    suffix = str(config.n_removed) + "removed_" + str(first) + "_" + str(last) + "_bs.npy"
   
    filename = join( config.path_to_results, "consistency_ratio_with_" + suffix )
    np.save( open(filename, 'w'), consistency_ratio_with )
    
    filename = join( config.path_to_results, "consistency_ratio_without_" + suffix )
    np.save( open(filename, 'w'), consistency_ratio_without )
    
    filename = join( config.path_to_results, "consistency_ratio_outside_" + suffix )
    np.save( open(filename, 'w'), consistency_ratio_outside )

    print "\nValues saved"

