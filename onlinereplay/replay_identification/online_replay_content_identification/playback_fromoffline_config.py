#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:15:38 2017

@author: davide
"""
import numpy as np
import os

path_to_posteriors =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/sleep/posteriors_entire_dataset/posteriors_20ms.npy"
path_to_posteriors_times =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/sleep/posteriors_entire_dataset/posterior_times_20ms.npy"
path_to_nspikes =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/sleep/posteriors_entire_dataset/nspikes_20ms.npy"
path_to_prerun_mua =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/prerun/mua.npy"
path_to_grid =\
"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/grid.npy"

history = 3
bin_size = 0.02

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
