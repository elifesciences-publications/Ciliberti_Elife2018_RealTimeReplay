# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:12:27 2015

@author: davide
"""
from os.path import join

file_path = "/home/davide/Data/ReplayDisruption/dataset_2018_03_06/2018-03-06_08-38-44/falcon/output/delay/20180306_102402"
prefix = "sink_spikes"
intermediate = ".0_spikedetector"
suffix = ".spikes.0.bin"
tt_ids = range(1, 50)
npy_amplitude_file = join( file_path, "all_spike_amplitudes.npy" )
