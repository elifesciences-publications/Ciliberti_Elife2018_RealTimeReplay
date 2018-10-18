# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:58:45 2015

@author: davide
"""

from __future__ import division
import numpy as np
from os.path import join

from multitrack_decoding import support_tools

import compute_mua_config as config

# load spike data files
spike_features = np.load( config.spike_features_filename )

mua, mua_t = support_tools.compute_mua( spike_features,\
               bin_size=config.bin_size_raw_mua, smooth_bw=config.smooth_mua_bw )

np.save( open( join( config.mua_filepath, "mua.npy" ), "w"), mua )
np.save( open( join( config.mua_filepath, "mua_t.npy" ), "w"), mua_t )
print( "MUA data was saved in: " + config.mua_filepath )