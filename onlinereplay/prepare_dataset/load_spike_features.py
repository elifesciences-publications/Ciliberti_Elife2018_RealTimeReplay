# -*- coding: utf-8 -*-
"""
Competes the preparation of a dataset by adding the spike features to a dataset
with only behavior data.

@author: davide
"""
from os.path import join, isfile
import h5py
from fklab.utilities import yaml

from fklab.io.extract_features import extract_spike_amplitudes

import config_file as config

dataset_id = config.path_to_dataset[-19:]
if dataset_id[-1] == '/':
    dataset_id = config.path_to_dataset[-20:-1]
    
filename = join( config.output_folder, dataset_id + ".hdf5" )
f_update = h5py.File( filename, 'r+' )
if not isfile( filename ):
    raise ValueError("No hdf5 file existing yes. Check your folder path and make"+\
    " you ran extract_behavior.py")

start_time = f_update["general/start_time"][0]
stop_time = f_update["general/stop_time"][0]
spf, channel_masks , _, _ = extract_spike_amplitudes(\
    config.path_to_dataset, min_ampl=config.min_ampl, perc=config.perc,\
    min_spike_rate=config.min_spike_rate, exclude=config.tt_exclude,\
    start=start_time, stop=stop_time, print_messages=True )

grp = f_update.create_group( "ephys" )

for i, tt in enumerate(spf):
    subgrp = grp.create_group( tt["original_id"] )
    dset = subgrp.create_dataset( "spikes/times", data=tt["time"] )
    dset = subgrp.create_dataset( "spikes/amplitudes", data=tt["value"] )
        
f_update.close()

# save pre-processing parameters
d = { "Min amplitude" : config.min_ampl,\
    "Percentile amplitude selection" : config.perc,\
    "Min spiking rate" : config.min_spike_rate,\
    "N tetrodes loaded" : len(spf),\
    "excluded tetrodes" : config.tt_exclude }
filename_params_file = join( config.output_folder, "preprocessing_info.yaml")
if not isfile( filename_params_file ):
    raise ValueError("No file with config parameters.")
file_params = open( filename_params_file, "a" )
yaml.dump( d, file_params, default_flow_style=False )
file_params.close()
print "\nConfig parameters about loading of spike features appended to " + filename_params_file
