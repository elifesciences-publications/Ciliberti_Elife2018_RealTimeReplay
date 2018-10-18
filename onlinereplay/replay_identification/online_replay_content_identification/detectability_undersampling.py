# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:39 2016

@author: davide
"""
from __future__ import division
import numpy as np
from scipy import special
from os.path import join, dirname
import cPickle as pickle
import itertools

from simulator import OnlineReplaySimulator
import support_replay_analysis.tools as tools

from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml
from fklab.segments import Segment as Seg

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder

import detectability_undersampling_config as config


BIN_SIZE_REPLAY = 0.01
HISTORY = 3
HALF_INTEGRATION_WINDOW = 6

mua_range = np.arange( 0, 12.25, 0.25 )
pkns_range = np.arange( 0, 1.05, 0.05 )

print "\n{0} tetrodes excluded" .format(config.n_removed)

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
            spike_features, j, {'spike_times':np.zeros(1),\
                                'spike_amplitudes':np.zeros(max(n_active_channels))})
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
        spike_ampl_mask_list.insert( mm, np.zeros( max(n_active_channels),\
                                                  dtype='bool' ) )
    print "Tetrode inclusion mask was expanded with {} false elements"\
    .format( len( mismatches) )
n_tt = sum(tt_included)
  
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
    
# load environment
xgrid_vector = np.load( join( config.path_to_encoding_model, "grid_cm.npy" ) )
grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
grid_element_length = np.mean( np.diff( grid.ravel() ) )
content_nodes = np.load( join ( config.path_to_environment, "content_nodes.npy" ) )
n_contents = len(content_nodes)
new_content_nodes, equalized_content_nodes = tools.fix_content_nodes(\
    content_nodes, equalize=True,\
    arm_length_grid_units=np.int(info_preprocess["ArmLength_cm"]/grid_element_length+1 ) ) 
adjusted_grid = grid[:new_content_nodes[-1, -1]+1] 
grid_size = len(adjusted_grid)

# load performance meter
pmj = pickle.load( open( config.path_to_performance, 'r') )

# compute pax for all tetrodes of the entire dataset
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, info_model["offset"] )
epoch_split = Seg.fromarray( startstop ).split( BIN_SIZE_REPLAY )

print "\nDecoding entire dataset"
_ = decoder.decode( spike_features,
                epoch_split, tt_included,\
                spike_ampl_mask_list, BIN_SIZE_REPLAY,\
                sf_keys=["spike_times", "spike_amplitudes"] )[0]

win_half_low, win_half_high = tools.win_halves( grid_size, HALF_INTEGRATION_WINDOW )

decoding_machine = tools.DecodingMachine(\
    tt_included, spike_ampl_mask_list, decoder, spike_features,\
    equalized_content_nodes, grid_size, win_half_low, win_half_high, BIN_SIZE_REPLAY)

print "\n{0} tetrodes excluded" .format(config.n_removed)

n_combos = int(special.binom( n_tt, config.n_removed ))
n_bootstrap = min(n_combos, config.n_max_bootstrap)

results = {}
results["Matthews correlation coefficient"] = np.zeros( n_bootstrap )
results["optimal thresholds"] = np.zeros( (n_bootstrap, 2) )
results["confusion matrix"] = np.zeros( (n_bootstrap, 2, 2 ), dtype=int )
results["optimal_MUA_thresholds_absolute"] = np.zeros( n_bootstrap )
results["out_of_candidate_rate"] = np.zeros( n_bootstrap )
results["inter_content_accuracy"] = np.zeros( n_bootstrap )
results["median relative latency"] = np.zeros( n_bootstrap )

# load random combinations (without replacement) of indices for tetrode selection
combo_indices_all = np.load( config.path_to_random_indices )
first = config.start_random_index
last = config.start_random_index+n_bootstrap
if last > n_combos:
    last = n_combos
combo_indices = combo_indices_all[config.n_removed][first:last]

j = 0

path_to_grid = join( config.path_to_environment, "grid.npy" )

_, ephys_prerun, _  = read_dataset( config.path_to_prerun_preprocessed_dataset )
spikes_encoding = tools.transform_spike_format( ephys_prerun )

for i,combo in enumerate(itertools.combinations( range(n_tt), config.n_removed )):
    
    if i not in combo_indices:
        continue
    
    posteriors, n_spikes = decoding_machine.compute_posteriors_combo(\
                                 combo, epoch_split, lazy_decode=True )
    
    sim = OnlineReplaySimulator.from_temp_posteriors( posteriors,\
         epoch_split.start, np.sum(n_spikes, axis=0), path_to_grid,\
         spikes_encoding, discard=combo )
    sim.compute_variables( HALF_INTEGRATION_WINDOW )
    
    performance, mua_absolute_thresholds = tools.performance_mua_pnks(\
                                              sim, pmj, mua_range, pkns_range )
    mcc_array = np.array( [[pf['Matthews correlation coefficient']\
                     for pf in perf] for perf in performance] )
    argmax = np.argmax( mcc_array )
    argmax_unraveled = np.unravel_index( argmax, mcc_array.shape )
    results["Matthews correlation coefficient"][j] = mcc_array[argmax_unraveled]
    results["optimal thresholds"][j,0] = mua_range[argmax_unraveled[0]]
    results["optimal thresholds"][j,1] = pkns_range[argmax_unraveled[1]]
    results["optimal_MUA_thresholds_absolute"][j] =\
        mua_absolute_thresholds[argmax_unraveled[0]][argmax_unraveled[1]]
    results["confusion matrix"][j,:,:] = np.int64(\
        performance[argmax_unraveled[0]][argmax_unraveled[1]]["confusion matrix"])
    results["out_of_candidate_rate"][j] =\
        performance[argmax_unraveled[0]][argmax_unraveled[1]]["out_of_candidate_rate"]
    results["inter_content_accuracy"][j] =\
        performance[argmax_unraveled[0]][argmax_unraveled[1]]["inter_content_accuracy"]
    results["median relative latency"][j] =\
        performance[argmax_unraveled[0]][argmax_unraveled[1]]["median relative latency"]
    
    j += 1
    print "\n{0} % completed" .format(int(j/n_bootstrap*100))
    

if config.save_values:
    filename = join( config.path_to_results, "results_detectability_" +\
        str(first) + "_" + str(last) + "_bs_" + str(config.n_removed) + "removed.npy")
    np.save( open( filename, 'w'), results )
    print "\nResults saved in " + filename

