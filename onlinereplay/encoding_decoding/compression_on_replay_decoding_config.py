#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 11:22:12 2018

@author: davide
"""

import numpy as np

root_path =\
"/home/davide/Dropbox/LabPapers/OnlineReplay/Data/post-analysis/dataset_2017_08_23/sleep/comparisons"

path_to_content_nodes =\
"/home/davide/Dropbox/LabPapers/OnlineReplay/Data/environments/dataset_2017_08_23/content_nodes.npy"

thresholds = np.linspace( 1, 5, 5 )

map_thr = 0.05