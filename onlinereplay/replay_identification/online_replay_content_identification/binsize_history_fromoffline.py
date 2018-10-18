#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:15:00 2017

@author: davide
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from os.path import join

from simulator import OnlineReplaySimulator
from support_replay_analysis import tools

import binsize_history_fromoffline_config as config


bin_sizes_ms = [5, 10, 15, 20]
history_values = [2, 3, 4]
mua_range = np.arange( 0, 12.25, 0.25 )
pkns_range = np.arange( 0, 1.05, 0.05 )

pmj = tools.load_reference_replay( config.path_to_reference )

mcc_param_space = np.zeros( (len(bin_sizes_ms), len(history_values)) )
ocr_param_space = np.zeros_like( mcc_param_space )
mrl_param_space = np.zeros_like( mcc_param_space )
ica_param_space = np.zeros_like( mcc_param_space )


for i,bs in enumerate(bin_sizes_ms):
    
    suff = str(bs) + "ms.npy"
    path_to_posteriors = join( config.path_to_posterior_data, "posteriors_" + suff )
    path_to_posteriors_times = join( config.path_to_posterior_data, "posterior_times_" + suff )
    path_to_nspikes = join( config.path_to_posterior_data, "nspikes_" + suff )
    
    for j,h in enumerate(history_values):
        
        sim = OnlineReplaySimulator.from_offline_posteriors(\
            path_to_posteriors, path_to_posteriors_times, path_to_nspikes,\
            config.path_to_prerun_mua, config.path_to_grid, h=h, bin_size=bs/1e3 )
        sim.compute_variables( config.half_integration_window )
        performance = []
        print "Computing performance matrices bin_size {0} ms, history {1} bins" .format(bs, h)
        performance = tools.performance_mua_pnks( sim, pmj, mua_range, pkns_range )
        mcc_matrix = np.array( [[pf['Matthews correlation coefficient']\
                         for pf in perf] for perf in performance] )
        ocr_matrix = np.array( [[pf['out_of_candidate_rate']\
                         for pf in perf] for perf in performance] )
        mrl_matrix = np.array( [[pf['median relative latency']\
                         for pf in perf] for perf in performance] )
        ica_matrix = np.array( [[pf['inter_content_accuracy']\
                         for pf in perf] for perf in performance] )
        argmax = tools.argmax( mcc_matrix )
        mcc_param_space[i,j] = mcc_matrix[argmax]
        ocr_param_space[i,j] = ocr_matrix[argmax]
        mrl_param_space[i,j] = mrl_matrix[argmax]
        ica_param_space[i,j] = ica_matrix[argmax]


# PLOTTING
X, Y = np.meshgrid( np.array([0]+bin_sizes_ms)+2.5, np.array([1]+history_values)+0.5 )
plt.subplot(221)
plt.pcolormesh( X, Y, mcc_param_space.T )
plt.xticks( bin_sizes_ms )
plt.yticks( history_values )
plt.xlabel( "bin size [ms]" )
plt.ylabel( "history [#bins]" )
cbar = plt.colorbar()
cbar.set_label("Matthews correlation")

plt.subplot(222)
plt.pcolormesh( X, Y, ocr_param_space.T*60)
plt.xticks( bin_sizes_ms )
plt.yticks( history_values )
plt.xlabel( "bin size [ms]" )
plt.ylabel( "history [#bins]" )
cbar = plt.colorbar()
cbar.set_label("non-burst detection rate [$min^{-1}$]")

plt.subplot(223)
plt.pcolormesh( X, Y, mrl_param_space.T )
plt.xticks( bin_sizes_ms )
plt.yticks( history_values )
plt.xlabel( "bin size [ms]" )
plt.ylabel( "history [#bins]" )
cbar = plt.colorbar()
cbar.set_label("median relative latency")

plt.subplot(224)
plt.pcolormesh( X, Y, ica_param_space.T, vmin=0, vmax=1 )
plt.xticks( bin_sizes_ms )
plt.yticks( history_values )
plt.xlabel( "bin size [ms]" )
plt.ylabel( "history [#bins]" )
cbar = plt.colorbar()
cbar.set_label("content accuracy")

plt.show()

