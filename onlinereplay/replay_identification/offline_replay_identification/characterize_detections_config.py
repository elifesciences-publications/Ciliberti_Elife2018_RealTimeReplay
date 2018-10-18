#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:32:17 2017

@author: davide
"""

from os.path import join


analysis_root = "/home/davide/Dropbox/Davide/Replay_Analysis"

#preprocessed_root = "/home/davide/Data/RealTimeReplayDetection/FK005/PreprocessedDatasets"
#dataset_month_year = "2016_01_"
#days = ( 24,25, 26 )

preprocessed_root = "/home/davide/Dropbox/LabPapers/OnlineReplay/Data/pre-processed"
dataset_month_year = "2017_08_"
days = ( 23, )

epoch = ( "sleep", )

#path_to_bias_score = "/home/davide/Dropbox/Davide/Replay_Analysis/content_measures/bias_score_3000s_800sh.npy" 
#online_dets_playback_filepath =\
#"/home/davide/Dropbox/Davide/Replay_Analysis/playback/online_detections_playback.npy"

MIN_LF_SCORE = 0.1
MIN_BIAS_SCORE = 3.0
MIN_DURATION_JOINT = 0.1
MIN_SEQUENCE_SCORE = -33.0

save_cm = False
path_to_cm = join( analysis_root, "dataset_" + dataset_month_year +\
         "%02d"%days[0], epoch[0], "confusion_matrix_" +\
         str(days[0]) + "_" + epoch[0] + ".npy" )
path_to_cmo = join( analysis_root, "dataset_" + dataset_month_year +\
         "%02d"%days[0], epoch[0], "confusion_matrix_object_" +\
         str(days[0]) + "_" + epoch[0] + ".npy" )
path_to_tp = join( analysis_root, "dataset_" + dataset_month_year +\
         "%02d"%days[0], epoch[0], "tp_" + str(days[0]) + "_" + epoch[0] + ".npy" )

random_cm_day = None #days[0] # 23, 24, 25, None
random_cm_epoch = epoch[0] # "sleep", "postrun" 
n_random_cm = 2000
save_random_cm = False
if random_cm_day is not None:
    path_to_random_cm = join( analysis_root, "dataset_" + dataset_month_year +\
         "%02d"%random_cm_day, random_cm_epoch, "random_confusion_matrix_objects" +\
         str(random_cm_day) + "_" + random_cm_epoch )

save_reference = False
path_to_reference = join( "/home/davide/Dropbox/Davide/Replay_Analysis",\
     "dataset_" + dataset_month_year + str(days[0]), epoch[0],\
     "reference_performance_meter.p" )

save_pandas_frame = False
path_to_pandas_frame = join( analysis_root, "characterization.csv" )

nbins = 500 # for cumulative distribution

plot = False

export_yaml_variables = False
yaml_filepath = "/home/davide/Dropbox/Davide/Replay_Analysis/variables.yaml"
