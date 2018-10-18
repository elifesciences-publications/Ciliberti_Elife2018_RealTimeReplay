
path_to_preprocessed_dataset_folder =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_25_prerun"
use_falcon_spikes = False
#path_to_falcon_spikes =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/output/prerun/20170825_101924/all_spike_amplitudes.npy"


# data selection options
min_run_duration = 1. # in sec
n_partitions = 1 # should be at least 2 for proper validation, must be even
partition_method = 'block' # 'sequence' or 'block'
partition_index = 0 # must be <= n_partitions/2 when n_partitions >2
fold = 1 # 0 or 1 (generally use 1)
min_n_encoding_spikes = 1
run_speed = 8.5

max_training_time = None # in minutes or None

# encoding - decoding options
bin_size_run = 0.2 # in seconds

# path_to_experimental_grid points to the grid used during online analysis,
# enter None if not available
#path_to_experimental_grid =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/encodingmodel_precomputations/23_08_2017/environment/grid.npy"
#path_to_experimental_grid =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/grid.npy"
path_to_experimental_grid =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/encodingmodel_precomputations/25_08_2017/environment/grid.npy"
grid_element_size_cm = 4 # will be used to generate a grid in case of no
# experimental grid provided
behav_bw_cm = 8.0
spf_bw_mV = 0.03
compression_threshold = 0.8
offset = 1e-9 # Hz

# plotting options
plot_train_test = False
plot_dec_behav = False
plot_dec_error = False
plot_occupancy = False
use_corrected_positions = True

# saving options
save_model = True
#path_to_encoding_model = join( dirname(path_to_preprocessed_dataset) ,\
#    "model" )
path_to_encoding_model =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_25_prerun/model_"\
+ str(compression_threshold) + "_threshold"
#path_to_encoding_model=\
#"/home/davide/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_25_prerun/comparisons/model_0.0_threshold"
save_decoded_behavior = False
path_to_decoded_behavior=\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_25/prerun/decoded_behavior_" +\
str(compression_threshold) + "_" + str(int(bin_size_run*1e3)) + "ms.npy"
save_decoding_errors = False
path_to_decoding_errors =\
"/home/davide/Dropbox/Davide/Replay_Analysis/dataset_2017_08_" + path_to_preprocessed_dataset_folder[-9:-7] +\
"/prerun/errors_with_experimental_compression_threshold/run_decoding_errors_" + str(compression_threshold) + ".npy"
