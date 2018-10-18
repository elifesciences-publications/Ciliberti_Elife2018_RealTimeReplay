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

import playback_fromoffline_config as config


sim = OnlineReplaySimulator.from_offline_posteriors( config.path_to_posteriors,\
                                                     config.path_to_posteriors_times,\
                                                     config.path_to_nspikes,\
                                                     config.path_to_prerun_mua,\
                                                     config.path_to_grid,\
                                                     h=config.history,
                                                     bin_size=config.bin_size )
sim.compute_variables( config.half_integration_window )

pmj = pickle.load( open( config.path_to_reference, 'r' ) )
# add fields in case they don't exist in the loaded version of the perfomnace meter object
pmj.tp_semi = 0
pmj.fp_semi = 0
pmj.fn_semi = 0
pmj.tn_semi = 0
pmj.two_semi = 0

performance = []

mua_range = np.arange( 0, 12.25, 0.25 )
pkns_range = np.arange( 0, 1.05, 0.05 )

print "Computing performance matrices ..."
for mua_std_thr in mua_range:
    performance.append( [] )
    for pkns_thr in pkns_range:
        performance[-1].append( sim.compute_performance(\
                   pmj, [mua_std_thr, pkns_thr, pkns_thr], lockout=.075 ) )

mcc_matrix = np.array( [[pf['Matthews correlation coefficient'] for pf in perf]\
                        for perf in performance])

ocr_matrix = np.array( [[pf['out_of_candidate_rate'] for pf in perf]\
                        for perf in performance])
    
ca_matrix = np.array( [[pf['inter_content_accuracy'] for pf in perf]\
                        for perf in performance])
    
mrl_matrix = np.array( [[pf['median relative latency'] for pf in perf]\
                        for perf in performance])

# PLOTTING
extent = [mua_range[0], mua_range[-1], pkns_range[0],pkns_range[-1]]
plt.subplot(221)
plt.title('Matthew correlation')
plt.imshow( mcc_matrix, aspect='auto', extent=extent )
plt.xlabel('MUA threshold [S.D.]')
plt.ylabel('Sharpness threshold')
plt.colorbar()
plt.subplot(222)
plt.title('Out of candidate rate')
plt.imshow( ocr_matrix, aspect='auto', extent=extent )
plt.xlabel('MUA threshold [S.D.]')
plt.ylabel('Sharpness threshold')
plt.colorbar()
plt.subplot(223)
plt.title('Content accuracy')
plt.imshow( ca_matrix, aspect='auto', extent=extent )
plt.xlabel('MUA threshold [S.D.]')
plt.ylabel('Sharpness threshold')
plt.colorbar()
plt.subplot(224)
plt.title('median relative latency')
plt.imshow( mrl_matrix, aspect='auto', extent=extent )
plt.xlabel('MUA threshold [S.D.]')
plt.ylabel('Sharpness threshold')
plt.colorbar()
plt.show()

