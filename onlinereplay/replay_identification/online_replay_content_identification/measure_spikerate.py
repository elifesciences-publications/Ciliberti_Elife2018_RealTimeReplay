#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:53:06 2018

@author: davide
"""
from __future__ import division
import pickle
from matplotlib import pyplot as plt
from falcon.io import load_file

data, header =\
load_file('/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/output/postrun/20170824_110102/sink_synch/sink_synch.0_synchronizer.estimates.0.bin')

cand = pickle.load(open('/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_postrun/candidate_events.p', 'r'))
m = cand.contains(data['hardware_ts']/1e6)[0]
avg_spike_rate_tt = data['n_spikes'][m]/(data['time_bin_ms'][m]/1e3)/14
plt.hist(avg_spike_rate_tt, bins=30, density=True)
plt.xlabel('spike rate per tetrode [spike/s]')

plt.show()