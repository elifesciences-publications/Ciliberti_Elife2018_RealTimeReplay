#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:41:46 2017

@author: davide
"""


preprocessed_root = "/home/davide/Dropbox/LabPapers/OnlineReplay/Data/pre-processed"
dataset_month_year = "2017_08_"
days = (23, 24, 25)
epoch = ( "sleep", "postrun" )

replay_string = "detection"

timewindow_ms = 200

speed_thr = 5 # cm/s

plot = False

save_to_yaml = True
path_to_yaml_file = "/home/davide/Dropbox/Davide/Replay_Analysis/variables.yaml"