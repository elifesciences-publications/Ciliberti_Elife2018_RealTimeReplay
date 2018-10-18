# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:51:25 2016

@author: davide
"""
import os

path_to_dataset =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/postrun/hack/2017-08-24_09-36-44.hdf5"

path_to_encoding_model =\
"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_prerun/model_2.0_threshold"

bin_size_replay = 0.015 # s

save_posteriors = True
path_to_posteriors_dump = os.path.join(\
   "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/postrun/posteriors_entire_dataset/",\
   "posteriors_" + str(int(bin_size_replay*1e3)) + "ms.npy" )

save_posterior_times = True
path_to_posteriors_times_dump = os.path.join(\
   "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/postrun/posteriors_entire_dataset/",\
   "posterior_times_" + str(int(bin_size_replay*1e3)) + "ms.npy" )

save_nspikes = True
path_nspikes_dump = os.path.join(\
   "/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/postrun/posteriors_entire_dataset/",\
   "nspikes_" + str(int(bin_size_replay*1e3)) + "ms.npy" )