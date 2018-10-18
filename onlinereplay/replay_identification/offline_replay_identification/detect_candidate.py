# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 19:46:18 2016

@author: davide
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from os.path import dirname, join
import cPickle as pickle
from pandas import ewma

from fklab.segments import Segment as seg
from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
import fklab.signals.kernelsmoothing as ks
from fklab.signals.basic_algorithms import detect_mountains
from fklab.utilities import yaml
from fklab.io import neuralynx as nlx
from fklab.plot import plot_signals

import config_file as config


def detrend( x, span ):
    
    fwd = ewma( x, span=span ) # take EWMA in fwd direction
    bwd = ewma( x[::-1], span=span ) # take EWMA in bwd direction
    c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
    trend = np.mean( c, axis=0 ) # average
    return x - trend, trend


print "Loading and checking pre-processed data"
behavior, ephys, start_stop = read_dataset( config.path_to_preprocessed_dataset ) 
check_ephys_dataset( ephys, nlow=0 )
n_tt_dataset = len(ephys)
n_active_channels = [ np.shape( ephys[key]["spike_amplitudes"][1] )[0]\
    for key in ephys.keys() ]
duration_ephys = start_stop[1] - start_stop[0]

print "Extracting STOP segments"
stop = seg.fromlogical( behavior["speed"] < config.stopping_speed_cm, x=behavior["time"] )

print "Computing MUA"
all_spike_times = np.empty(0)
n_stop_spikes = np.uint64(0)
for tt in [ ephys[key] for key in ephys.keys() ]:
    all_spike_times = np.concatenate( ( all_spike_times, tt['spike_times']) )
    n_stop_spikes += np.sum( len(tt['spike_times']) )
all_spike_times.sort()

n_mua = np.uint( duration_ephys / config.bin_size_mua )
mua_bins = np.linspace( start_stop[0]+config.bin_size_mua,\
    start_stop[1]+config.bin_size_mua, n_mua )
histogram, mua_t = np.histogram( all_spike_times, bins=mua_bins, density=False )
mua_t = mua_t[:-1] # remove last element so that the two returns have same # elements

if config.smooth_mua_bw > 0:
    smoothed_histogram = ks.Smoother( kernel=ks.GaussianKernel(\
        bandwidth=config.smooth_mua_bw ) )( histogram, delta=config.bin_size_mua )
else:
    smoothed_histogram = histogram
mua_w_trend = smoothed_histogram / config.bin_size_mua
if config.detrend:
    mua, trend = detrend( mua_w_trend, config.span_ewma )
else:
    mua = mua_w_trend
    trend = np.zeros( len(mua) )
mean_firing_rate = n_stop_spikes / np.sum( duration_ephys )
#assert( np.isclose( mean_firing_rate, mua.mean()) )

# detect CANDIDATE replay events occurring inside and outside the maze
print "\nDetecting candidate events"   
mask_stop = stop.contains( mua_t )[0] 
if len(stop) == 0:
    assert( np.all(np.isnan(behavior['linear_position'])) )
    mask_stop = np.ones( len(mua), dtype='bool' ) # all mua for sleep sessions
mua_stop = mua[ mask_stop ]
# use stats from stop periods only
# mua stop used only for determining the thresholds
high_thr = mua_stop.mean() + config.mua_thr_std_H * mua_stop.std()
low_thr = mua_stop.mean() + config.mua_thr_std_L * mua_stop.std()

candidate = detect_mountains( mua, x=mua_t, high=(high_thr), low=(low_thr ) )
candidate = candidate[ candidate.duration > config.min_duration_cand ]
candidate.ijoin( gap=config.allowable_gap )

n_candidate_events = len(candidate)
#nbins_per_cand_event = np.uint( candidate.duration / config.bin_size_replay )
print "\n{0} candidate events have been detected in the entire dataset with ".format(\
    n_candidate_events) + " average duration of {0:0.2f} ms".format(\
    np.mean( candidate.duration * 1e3) )
    
if config.save_events:
    #TODO: insert None to span_ewma_detrend in case detrend is disabled
    pickle.dump( candidate, open( join( dirname(\
        config.path_to_preprocessed_dataset), "candidate_events.p"), 'w') )
    with open( join( dirname(config.path_to_preprocessed_dataset),\
    "candidate_detection_params.yaml" ), 'w' ) as yaml_file:
        d = {'bin_size_mua' : config.bin_size_mua,\
            'smooth_mua_bw': config.smooth_mua_bw,\
            'mua_thr_std_H' : config.mua_thr_std_H,\
            'mua_thr_std_L' : config.mua_thr_std_L,\
            'min_duration_cand' : config.min_duration_cand,\
            'max_allowable_gap' : config.allowable_gap,\
            'span_ewma_detrend' : config.span_ewma,\
            'max_speed' : config.stopping_speed_cm }
        yaml.dump( d, yaml_file, default_flow_style=False )
    print "\nCandidate events and config parameters saved inside " + dirname(\
        config.path_to_preprocessed_dataset )


if config.plot_mua:
    f = nlx.NlxOpen( config.path_to_online_detections )
    all_events  = f.readdata( start=f.starttime, stop=f.endtime )
    replay_ttl_times = all_events.time[all_events.eventstring == config.detection_string]
    print "Plotting MUA"
    plt.figure(1)
    mua_t0 = 0
    plt.plot( mua_t - mua_t0, mua_w_trend, label="MUA", color='blue' )
    t_start_plot = candidate.start - mua_t0
    t_stop_plot = candidate.stop - mua_t0
    for t1, t2 in zip(t_start_plot, t_stop_plot):
        plt.fill( [t1, t2, t2, t1], [0, 0, mua_w_trend.max(), mua_w_trend.max()],\
            'b', alpha=0.2, edgecolor='r' )
    plot_signals( mua_t - mua_t0, trend+low_thr, label="low threshold",\
        linewidth=2, color="green" )
    plot_signals( mua_t - mua_t0, trend+high_thr, label="high threshold",\
        linewidth=2, color="red" )
    plot_signals( mua_t - mua_t0, trend, label="trend", linewidth=2, color="gray")
    plt.vlines( replay_ttl_times, 0, 3000, label="online detections", linewidth=2,\
        color='black')
    plt.legend( loc="best" )
    plt.ylabel( "spikes/s" )
    plt.xlabel( "time [s]" )
    plt.figure(2)
    plt.hist( candidate.duration * 1e3, bins=20 )
    plt.title( "durations of detected candidate events [ms] " )
    
if config.save_mua:
    np.save( open( config.mua_filename, 'w' ), mua_w_trend )
    np.save( open( config.mua_times_filename, 'w'), mua_t )

plt.show()
