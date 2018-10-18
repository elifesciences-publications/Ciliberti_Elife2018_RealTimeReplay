# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:21:12 2015

@author: davide
"""
import numpy as np
from falcon.io.deserialization import load_spike_data
from os.path import join, isfile

import read_falcon_spikes_config as config

print "Loading spike features from Falcon output"

spike_features = []
for _id in config.tt_ids:   
    processor_id = config.prefix + str(_id)
    file_path_full = join( config.file_path, processor_id, processor_id +\
        config.intermediate + str(_id) + config.suffix )
    if isfile( file_path_full ):
        print "Loading from tetrode {0}" .format( _id )
        spike_data, tot_n_spikes, n_channels, data, header = load_spike_data( file_path_full )
        amplitudes_tt = dict()
        amplitudes_tt["value"] = spike_data['amplitudes']
        amplitudes_tt["time"] = spike_data['times']
        amplitudes_tt["id"] = "TT" + str(_id)
        spike_features.append( amplitudes_tt )
    
np.save( open( config.npy_amplitude_file, 'w'), spike_features )
print("Spike features loaded from Falcon output files and saved in " +\
    config.npy_amplitude_file)
