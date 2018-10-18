#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:39:53 2016

@author: davide
"""

from os.path import join
import h5py
from fklab.ui.localize.tracker import NlxSingleTargetTracker

from preprocess_behavior_localize_config import source_filename, output_path,\
    overwrite_options_file


tracker = NlxSingleTargetTracker()

tracker.set_source( source_filename )

tracker.set_behavior_options( velocity_smooth=0.5 )

print tracker._options

options_file = join( output_path, "position.yaml" )

tracker.save_options( options_file, overwrite=overwrite_options_file )

tracker.compute_behavior()

filename= join( output_path, "position.hdf5" )

fid = h5py.File(filename,'w')

behav = tracker.behavior

targets = tracker.corrected_target_coordinates

fid.create_dataset("time", data=behav['time'])

grp = fid.create_group("diodes")

if len(targets)>0:
    grp.create_dataset("diode1", data=targets[0])

if len(targets)>1:
    grp.create_dataset("diode2", data=targets[1])

fid.create_dataset("position", data=behav['position'])

fid.create_dataset("velocity", data=behav['velocity'])

fid.create_dataset("head_direction", data=behav['head_direction'])

fid.close()

print "\nFinished"