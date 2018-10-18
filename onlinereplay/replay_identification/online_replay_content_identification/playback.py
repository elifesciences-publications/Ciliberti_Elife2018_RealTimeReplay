#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:15:00 2017

@author: davide
"""
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle

from simulator import OnlineReplaySimulator
from support_replay_analysis.plot import plot_param_performance_dependency as pppd

import playback_config as config


sim = OnlineReplaySimulator( config.path_to_likelihood_synch,\
                            config.path_to_prerun_mua, config.path_to_grid )
sim.compute_variables( config.half_integration_window )

pmj = pickle.load( open( config.path_to_reference, 'r' ) )

mua_performance = []
peakiness_performance = []

pkns_mua_range = config.mua_params[0]
mua_pkns_range = config.peakiness_params[0]

print "Playing back session with different MUA thresholds..."
for mua_std_thr in config.mua_params[1]:
    mua_performance.append( sim.compute_performance(\
            pmj, [mua_std_thr, pkns_mua_range, pkns_mua_range] ) )
    
print "Playing back session with different peakiness thresholds..."
for pkns_thr in config.peakiness_params[1]:
    peakiness_performance.append( sim.compute_performance(\
          pmj, [mua_pkns_range, pkns_thr, pkns_thr],  lockout=.075 ) )


# PLOTTING
xticks = np.arange( 0, config.mua_params[1].max()+1, 2 )
xlabel = "MUA threshold [S.D.]"
pppd( config.mua_params[1], xlabel, xticks, mua_performance, fign=1 )

xticks = np.linspace( 0, 1, num=6 )
xlabel = "sharpness"
pppd( config.peakiness_params[1], xlabel, xticks, peakiness_performance, fign=2 )

plt.show()

