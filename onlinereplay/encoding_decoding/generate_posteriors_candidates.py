# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:39 2016

@author: davide
"""

import numpy as np
from os.path import join, isfile, dirname
import cPickle as pickle
import natsort

from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.utilities import yaml

from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder

import config_file_decoding as config


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
    
    
# read data
_, ephys, _ = read_dataset( config.path_to_dataset ) 
check_ephys_dataset( ephys )
n_tt_dataset = len(ephys)
n_active_channels = [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ]

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
behav_ndim = 1  
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
    
# load grid and candidate events
xgrid_vector = np.load( join( config.path_to_encoding_model, "grid_cm.npy" ) )
grid = xgrid_vector.reshape( (len(xgrid_vector),1) )
candidate = pickle.load( open( join( dirname( config.path_to_dataset ),\
    "candidate_events.p"), 'r') )
n_candidate_events = len(candidate)
assert( n_candidate_events > 0 )

# compute posteriors during extended candidate replay events
print "\nDecoding over {0} candidate events detected \
across the entire dataset with average duration of {1:.2f} ms" .format(\
    n_candidate_events, np.mean( candidate.duration*1000 ) )
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid,\
    info_model["offset"] )
posteriors_candidate = []
n_spikes = np.empty( (len(spike_ampl_mask_list), 0) )
n_tt = []

for i in range(n_candidate_events):
    
    candidate_event_split = candidate[i].split( size=config.bin_size_replay )
    
    posterior_event, _, n_spikes_event, n_tt_ext_event =\
        decoder.decode( spike_features, candidate_event_split, tt_included,\
            spike_ampl_mask_list, config.bin_size_replay,\
            sf_keys=["spike_times", "spike_amplitudes"] )
            
    posteriors_candidate.append( posterior_event )
    n_spikes = np.hstack( (n_spikes, n_spikes_event) )
    n_tt.append( n_tt_ext_event )
    assert( np.sum( np.isnan( posterior_event ) ) == 0 )

n_spikes = np.int32( n_spikes )

if config.save_posteriors:
    np.save( open( config.path_to_posteriors_dump, 'w'), posteriors_candidate )
    print "\nPosteriors saved in " + config.path_to_posteriors_dump

if config.save_posterior_times:
    np.save( open( config.path_to_posteriors_times_dump, 'w'), candidate_event_split.start )
    print "\nPosteriors times saved in " + config.path_to_posteriors_times_dump

if config.save_nspikes:
    np.save( open( config.path_nspikes_dump, 'w'), np.sum(n_spikes, axis=0) )
    print "\nN_spikes saved in " + config.path_nspikes_dump