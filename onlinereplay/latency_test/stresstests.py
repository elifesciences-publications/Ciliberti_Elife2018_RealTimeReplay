#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:10:51 2017

@author: davide
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from fklab.io import neuralynx as nlx
import seaborn as sns

import stresstests_config as config

START = 'START_OF_STREAM'
DETECTION = 'TEST_REPLAY'
BIN_SIZE_MS = 10
N_BINS_TEST_REPLAY = 10
N_LATENCIES = 3000
cores = (4, 32)
tetrodes = (16, 24, 32)  
spikes_per_bin = (0, 2, 4, 6, 8, 10)

test_replay_duration = BIN_SIZE_MS * N_BINS_TEST_REPLAY / 1000 # in seconds
added_latencies = np.zeros( (len(tetrodes), len(spikes_per_bin), N_LATENCIES, len(cores)) )

for c,n_cores in enumerate(cores):
    for i,n_tt in enumerate(tetrodes):
        for j,n_spb in enumerate(spikes_per_bin):
            filepath = join( config.root, str(n_cores) + "core",\
                            str(n_tt) + 'tt' + str(n_spb) + 'spb', "Events.nev" )
            f = nlx.NlxOpen( filepath )
            all_events  = f.readdata(start=f.starttime, stop=f.endtime)
            start_of_stream_time = all_events.time[all_events.eventstring == START][0]
            end_events_time = all_events.time[all_events.eventstring == DETECTION][:N_LATENCIES]
            stop_time = end_events_time[-1] - test_replay_duration
            start_events_time = np.arange(start_of_stream_time, stop_time, test_replay_duration)[:N_LATENCIES]
            n_test_replay_events = len(end_events_time)
            latencies = end_events_time - start_events_time
            added_latencies[i,j,:,c] = (latencies - test_replay_duration)*1e3 # in ms

fig = plt.figure( figsize=(10, 10))
sns.reset_defaults()
colors = sns.color_palette( "cubehelix", n_colors=len(tetrodes ) )
bin_size = BIN_SIZE_MS/1e3
plt.subplot(221)
plt.title("QUADCORE")
for i in range(len(tetrodes)):
    plt.plot( np.array(spikes_per_bin)/bin_size, np.percentile( added_latencies[i,:,:,0], 50, axis=1),\
             label=str(tetrodes[i]) + " tetrodes", color=colors[i])
plt.ylabel( "median added latency [ms]")
plt.xlabel( "spikes/s/tetrode")
plt.ylim( [0,3] )
plt.xlim( np.array([spikes_per_bin[0], spikes_per_bin[-1]])/(BIN_SIZE_MS/1e3))
plt.legend( loc='best' )
plt.subplot(223)
p = 99
for i in range(len(tetrodes)):
    plt.semilogy( np.array(spikes_per_bin)/bin_size, np.percentile( added_latencies[i,:,:,0], p, axis=1),\
             label=str(tetrodes[i]) + " tetrodes", color=colors[i])
plt.ylabel( "added latency, " + str(p)+ "th percentile  [ms]")
plt.xlabel( "spikes/s/tetrode" )
plt.legend( loc='best' )
plt.ylim( ymin=0.1, ymax=1e3 )
plt.xlim( np.array([spikes_per_bin[0], spikes_per_bin[-1]])/(BIN_SIZE_MS/1e3))
plt.subplot(222)
plt.title("32-CORE")
for i in range(len(tetrodes)):
    plt.plot( np.array(spikes_per_bin)/bin_size, np.percentile( added_latencies[i,:,:,1], 50, axis=1),\
             label=str(tetrodes[i]) + " tetrodes", color=colors[i])
plt.ylabel( "median added latency [ms]")
plt.xlabel( "spikes/s/tetrode")
plt.ylim( [0,3] )
plt.xlim( np.array([spikes_per_bin[0], spikes_per_bin[-1]])/(BIN_SIZE_MS/1e3))
plt.legend( loc='best' )
plt.subplot(224)
for i in range(len(tetrodes)):
    plt.semilogy( np.array(spikes_per_bin)/bin_size, np.percentile( added_latencies[i,:,:,1], p, axis=1),\
             label=str(tetrodes[i]) + " tetrodes", color=colors[i])
plt.ylabel( "added latency, " + str(p)+ "th percentile  [ms]")
plt.xlabel( "spikes/s/tetrode" )
plt.legend( loc='best' )
plt.ylim( ymin=0.1, ymax=1e3 )
plt.xlim( np.array([spikes_per_bin[0], spikes_per_bin[-1]])/(BIN_SIZE_MS/1e3))

plt.show()
