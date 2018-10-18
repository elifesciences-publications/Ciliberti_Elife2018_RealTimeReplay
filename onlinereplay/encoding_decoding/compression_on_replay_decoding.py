#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:57:22 2017

@author: davide
"""

from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import seaborn
from os.path import join

import compression_on_replay_decoding_config as config


def extract_content( grid_ind, content_nodes ):
    '''
    '''
    for c,cn in enumerate(content_nodes):
        if grid_ind >= cn[0] and grid_ind <= cn[1]:
            return c
    return -1




content_nodes  = np.load( config.path_to_content_nodes )

pos_0 = np.load( join( config.root_path, "posteriors_candidate_events_0.0.npy" ) )

n_posteriors = sum( [ cand.shape[0] for cand in pos_0] )

max_pos = np.zeros( (len(config.thresholds), n_posteriors) )
max_pos_all = np.zeros( n_posteriors )
content_mismatch = np.zeros( len(config.thresholds) )
content_mismatch_low = np.zeros( len(config.thresholds) )
content_mismatch_high = np.zeros( len(config.thresholds) )
weighted_flip_rate = np.zeros( len(config.thresholds) )


for i, thr in enumerate(config.thresholds):
    pos_thr = np.load( join( config.root_path, "posteriors_candidate_events_" + str(thr) + ".npy" ) )
    p_ind = 0
    for cand, ref in zip(pos_thr, pos_0):
        for pc, pr in zip(cand, ref):
            if extract_content(np.argmax(pc), content_nodes) != extract_content(np.argmax(pr), content_nodes):
                content_mismatch[i] += 1
                weighted_flip_rate[i] += np.max(pr)
                max_pos[i, p_ind] = np.max(pr)
                if max_pos[i, p_ind] < config.map_thr:
                    content_mismatch_low[i] += 1
                else:
                    content_mismatch_high[i] += 1
            else:
                max_pos[i, p_ind] = np.NaN
            if i==0:
                # compute it only once
                max_pos_all[p_ind] = pr.max()
            p_ind += 1


nonweighted_flip_rate = content_mismatch / n_posteriors
nonweighted_flip_rate_low = content_mismatch_low / n_posteriors
nonweighted_flip_rate_high = content_mismatch_high / n_posteriors    

plt.figure(1)
plt.plot( config.thresholds, nonweighted_flip_rate_low, 'o', color='black',\
         label="P_MAP < " + str(config.map_thr) )
plt.plot( config.thresholds, nonweighted_flip_rate_high, 'o', color='gray',\
         label="P_MAP > " + str(config.map_thr) )
plt.legend( loc="top left" )
plt.xlabel( "compression threshold" )
plt.ylabel( "change fraction" )
plt.ylim( [0, 1] )
plt.xticks( config.thresholds )
plt.yticks( [0, .5, 1] )


