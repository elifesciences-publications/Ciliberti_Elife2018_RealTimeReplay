
path_to_preprocessed_dataset_folder =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/PreprocessedDatasets/dataset_2017_08_24_prerun"

# data selection options
min_run_duration = 1. # in sec
n_partitions = 2 # should be at least 2 for proper validation, must be even
partition_method = 'block' # 'sequence' or 'block'
partition_index = 0 # must be <= n_partitions/2 when n_partitions >2
fold = 1 # 0 or 1 (generally use 1)
min_n_encoding_spikes = 1
run_speed = 8.5

n_removed = 0

path_to_random_indices =\
"/home/fklab/Dropbox/Davide/Replay_Analysis/tetrode_degradation_detectability/combo_indices_randomized.npy"
start_random_index = 0
n_max_bootstrap = 50

# encoding - decoding options
bin_size_run = 0.2 # in seconds

# path_to_experimental_grid points to the grid used during online analysis,
# enter None if not available
#path_to_experimental_grid =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_23/2017-08-23_09-42-01/falcon/encodingmodel_precomputations/23_08_2017/environment/grid.npy"
path_to_experimental_grid =\
"/mnt/data3/Data/RealTimeReplayDetection/FM009/dataset_2017_08_24/2017-08-24_09-36-44/falcon/encodingmodel_precomputations/24_08_2017/environment/grid.npy"
#path_to_experimental_grid =\
#"/home/davide/Data/RealTimeReplayDetection/FM009/dataset_2017_08_25/2017-08-25_09-50-43/falcon/encodingmodel_precomputations/25_08_2017/environment/grid.npy"
grid_element_size_cm = 4 # will be used to generate a grid in case of no
# experimental grid provided
behav_bw_cm = 8.0
spf_bw_mV = 0.03
compression_threshold = 2.
offset = 1e-9 # Hz

# saving options
save_decoding_errors = True
path_to_decoding_errors =\
"/mnt/data2/Dropbox/Davide/Replay_Analysis/dataset_2017_08_24/prerun/errors_with_experimental_compression_threshold/tetrode_degradation"
