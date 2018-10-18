#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:10:51 2017

@author: davide
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from fklab.io import neuralynx as nlx

from support_replay_analysis.plot import configure_latency_plots

import overestimated_added_latency_config as config

PATH_OFFSET = 32

def replace_char(str1, char, new_char):
    new_string = ""
    for c in str1:
        if c == char:
            c = new_char
        new_string += c
    return new_string
        

f = nlx.NlxOpen(config.event_file_path)
all_events  = f.readdata(start=f.starttime, stop=f.endtime)
start_of_stream_time = all_events.time[all_events.eventstring == config.START][0]
end_events_time = all_events.time[all_events.eventstring == config.DETECTION]

test_replay_duration =\
    config.time_bin_duration * config.n_time_bins_test_replay / 1000 # in seconds
stop_time = end_events_time[-1] - test_replay_duration
start_events_time = np.arange(start_of_stream_time, stop_time, test_replay_duration)
n_test_replay_events = len(end_events_time)
latencies = end_events_time - start_events_time
added_latencies = (latencies - test_replay_duration) * 1000 # in ms

print("\n")
print "Min added latency = \t{:.4f} ms" .format(added_latencies.min())
print "median latency = \t{:.4f} ms" .format(np.median(added_latencies))
print "75th percentile = \t{:.4f} ms" .format(np.percentile(added_latencies, 75))
print "99th percentile = \t{:.4f} ms" .format(np.percentile(added_latencies, 99))
print "99.9th percentile = \t{:.4f} ms" .format(np.percentile(added_latencies, 99.9))
print "Max added latency = \t{:.4f} ms" .format(added_latencies.max())
print("Number of replay-like events = \t " + str(n_test_replay_events))

fig = plt.figure()
ax = plt.subplot( 111 )
fs = 20
configure_latency_plots( ax, [added_latencies], ["added latency"], config.nbins,
    fs=fs, colors=['k'], show_median=False )
plt.xlabel( "added latency [ms]", fontsize=fs )
plt.xlim( xmax=20 )
ax.xaxis.set_ticks( np.int32([0, 5, 10, 15, 20]) )
plt.vlines( 3, 0, 1, linestyles='dashed', alpha=0.25, color='k' )
ax.legend().set_visible(False)
plt.tight_layout()

plt.show()
