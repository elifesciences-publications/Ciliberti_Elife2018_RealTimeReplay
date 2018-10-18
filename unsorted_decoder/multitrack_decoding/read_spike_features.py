# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:31:59 2015

@author: davide
"""
import numpy as np
from support_tools import extract_spike_amplitudes
import config_file as config


print "Extracting spike amplitides from ", config.path_to_spikes
spf, spf_dim, start, stop = extract_spike_amplitudes(config.path_to_spikes,\
    24, min_ampl=0, perc=0)

np.save( open( config.spike_features_filename, "w"), spf)
np.save( open( config.start_stop_times_filename, "w"), (start, stop) )

print( "\nSpike features max dimension is: " + str(spf_dim) )
print( "\nSpike features were saved in " + config.spike_features_filename )
print( "\nFirst spike meaured at time: {0}" .format(start) )
print( "\nLast spike meaured at time: {0}" .format(stop) )