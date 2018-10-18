# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:28:08 2016

@author: davide

This script generates a preprocessed dataset for subsequent analysis.

"""

from __future__ import division
import numpy as np
import pylab
import h5py
import os
from os.path import join, isdir
from shutil import copyfile

from fklab.io import neuralynx as nlx
from fklab.utilities import yaml
from fklab.segments import Segment as Seg
from falcon.io.deserialization import load_file

from linearization import create_linearization_matrix

import config_file as config

if config.sleep_only and not config.not_prerun:
    raise ValueError("Config options not set correctly!")

data = {}
data["dataset ID"] = config.path_to_dataset[-19:]
if data["dataset ID"][-1] == '/':
    data["dataset ID"] = config.path_to_dataset[-20:-1]

with open( join( config.path_to_dataset, "info.yaml" ), 'r' ) as f_yaml:
    info = yaml.load( f_yaml )
    
n_epochs = len(info["epochs"])

# look for epoch id
for epoch_id in range(n_epochs+1):
    if epoch_id == n_epochs:
        raise ValueError("Epoch name " + config.epoch_name + " not found in info.yaml file.")
    if info["epochs"][epoch_id]['id'] == config.epoch_name: break
if epoch_id == n_epochs:
    raise ValueError("Invalid epoch id!")
# epoch id found

starttime_nlx = info["epochs"][epoch_id]['time'][0]
stoptime_nlx = info["epochs"][epoch_id]['time'][1]  
        
if config.not_prerun:
    falcon_data, falcon_header = load_file( config.path_to_estimated_behavior )
    starttime_falcon = falcon_data['hardware_ts'][0]/1e6
    stoptime_falcon = falcon_data['hardware_ts'][-1]/1e6
else:
    starttime_falcon = -1
    stoptime_falcon = 2**64*1e6
starttime = max( starttime_nlx, starttime_falcon )
stoptime = min( stoptime_nlx, stoptime_falcon )

analyzed_online = Seg( [starttime, stoptime] )
duration = stoptime - starttime
print "Selected epoch lasted {0} minutes" .format(duration/60) 
        
lin_matrix, pixel_to_cm = create_linearization_matrix( config.path_to_environmentfile,\
    config.arm_length_cm, config.extra_distance, config.out_of_track_value )   

if not config.sleep_only:
    
    path_to_preprocessed_positions = join( config.path_to_dataset, "epochs",\
        config.epoch_name, "position.hdf5" )
    f_localize = h5py.File( path_to_preprocessed_positions, "r" )
    sel = analyzed_online.contains( f_localize['time'])[0]
    selected_positions = np.vstack( (f_localize['position'][:,0][sel], f_localize['position'][:,1][sel]) ).T
    selected_time = f_localize['time'][sel]
    selected_velocity = f_localize['velocity'][sel]
    n_pos = len(selected_positions)
    linear_positions = np.ones( n_pos )
    for i in range(n_pos):
        if not np.any( np.isnan( selected_positions[i] ) ):
            coo = np.int32( np.round( selected_positions[i] ) )
            linear_positions[i] = lin_matrix[coo[0], coo[1]] / pixel_to_cm
        else:
            linear_positions[i] = np.NaN
            
if not isdir(config.output_folder):
    os.mkdir( config.output_folder )
    print "Output folder did not exist and was created."
f = h5py.File( join( config.output_folder, data["dataset ID"] + ".hdf5"), "w" )
dset = f.create_dataset( "general/start_time", data=np.array([starttime]) )
dset = f.create_dataset( "general/stop_time", data=np.array([stoptime]) )
if not config.sleep_only:
    dset = f.create_dataset( "behavior/linear_position", data=linear_positions )
    dset = f.create_dataset( "behavior/time", data=selected_time )
    dset = f.create_dataset( "behavior/speed", data=np.abs( selected_velocity )/pixel_to_cm )
    dset = f.create_dataset( "behavior/linearization_matrix", data=lin_matrix )
else:
    no_data = np.full( 1, np.NaN )
    dset = f.create_dataset( "behavior/linear_position", data=no_data )
    dset = f.create_dataset( "behavior/time", data=no_data )
    dset = f.create_dataset( "behavior/speed", data=no_data )
    dset = f.create_dataset( "behavior/linearization_matrix", data=no_data )
    print "\nYou have marked this dataset as a SLEEP ONLY dataset"
f.close()

print "\nBehavior dataset created"

# make a copy as backup (won't be directly overwritten by ephys dataset)
copyfile( join( config.output_folder, data["dataset ID"] + ".hdf5"),\
    join( config.output_folder, data["dataset ID"] + "_behavior.hdf5") )    
    
# save pre-processing info
config_filename = join( config.output_folder, "preprocessing_info.yaml")
f_config = open( config_filename, "w" )  
 
with  open( config.path_to_environmentfile, 'r') as f_env:
    info_env = yaml.load(f_env)
n_tracks = len( info_env[info_env.keys()[0]]["shapes"].keys() )

yaml.dump( {"DatasetID" : data["dataset ID"]}, f_config, default_flow_style=False )
generals = {\
    "ConversionFactorPixelsToCm" : np.float( np.round( pixel_to_cm, decimals=3) ),\
    "ArmLength_cm" : config.arm_length_cm,\
    "NumTracks" : n_tracks,\
    "AddedDistance_pixel" : config.extra_distance,\
    "SleepOnly" : config.sleep_only }
yaml.dump( generals, f_config, default_flow_style=False )

if not config.sleep_only:
    path_to_preprocessed_positions_settings = join( config.path_to_dataset, "epochs",\
        info["epochs"][epoch_id]["id"], "position.yaml" )
    copyfile( path_to_preprocessed_positions_settings,\
        join( config.output_folder, "position.yaml") ) 
    with open( join( config.output_folder, "position.yaml"), 'r') as f_pos_yaml:
        pos_yaml = yaml.load( f_pos_yaml )
    if data["dataset ID"] not in pos_yaml["source"]["path"]:
        raise ValueError("Datasets ID does not math the one in position.yaml")

f_config.close()

print "Config parameters were saved in " + config_filename

if config.plot_linearization_matrix and not config.sleep_only:
    pylab.figure(2)
    pylab.imshow( lin_matrix.T )

pylab.show()