import numpy as np

path_to_dataset =\
"/home/davide/Data/ReplayDisruption/dataset_2018_03_08/2018-03-08_08-51-19"
epoch_name = "delay"
not_prerun = True # is it a presleep epoch? all the params relatated to behavior will be ignored if True
sleep_only = True

# used for non-prerun sessions (sleep and postrun) to sub-select the epoch during which the online
# decoding was active. Not for pre-run as no falcon data is saved in full mode
path_to_estimated_behavior =\
"/home/davide/Data/ReplayDisruption/dataset_2018_03_08/2018-03-08_08-51-19/falcon/output/delay/20180308_100751/sink_behavior/sink_behavior.0_behavior_estimator.behavior.0.bin"

# linearization matrix
path_to_environmentfile =\
"/home/davide/Data/ReplayDisruption/dataset_2018_03_08/2018-03-08_08-51-19/falcon/encodingmodel_precomputations/08_03_2018/environment/env.yaml"
arm_length_cm = 110
extra_distance = 400
out_of_track_value = np.NaN
plot_linearization_matrix = True

# specify destination
output_folder =\
"/home/davide/Data/ReplayDisruption/PreprocessedDatasets/dataset_2018_03_08_" + epoch_name

# specify params for extraction of spike features from .ntt files
min_ampl = 0
perc = 0
min_spike_rate = 0 # spikes/s
tt_exclude = [4, 6, 7, 8, 10, 13, 14, 15, 17, 22, 23]