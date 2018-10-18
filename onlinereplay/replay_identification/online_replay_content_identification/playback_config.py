#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:15:38 2017

@author: davide
"""
import numpy as np
import os

path_to_likelihood_synch =\
"/home/davide/Dropbox/LabPapers/OnlineReplay/Data/playback/dataset_2017_08_24/sleep/sink_synch.0_synchronizer.estimates.0.bin"
path_to_prerun_mua =\
"/home/davide/Dropbox/LabPapers/OnlineReplay/Data/playback/dataset_2017_08_24/mua.npy"
path_to_grid =\
"/home/davide/Dropbox/LabPapers/OnlineReplay/Data/playback/dataset_2017_08_24/grid.npy"

half_integration_window = 6

path_to_reference =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/sleep/reference_performance_meter.p"

if os.path.basename(os.path.dirname(path_to_reference)) == "sleep":
    mua_thr_experiment = 2.5
    pkns_thr_experiment = 0.5
else:
    assert( os.path.basename(os.path.dirname(path_to_reference)) == "postrun" )
    mua_thr_experiment = 3.
    pkns_thr_experiment = 0.65

mua_params = ( pkns_thr_experiment, np.arange( 0, 12.25, 0.25 ) )
peakiness_params = ( mua_thr_experiment, np.arange( 0, 1.05, 0.05 ) )
