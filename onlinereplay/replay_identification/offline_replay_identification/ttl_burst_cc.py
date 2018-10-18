#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:26:58 2017

@author: davide
"""

from __future__ import division
import numpy as np
from os import listdir
from os.path import join, isfile, dirname
import cPickle as pickle
import natsort
import seaborn
from matplotlib import pyplot as plt
from fklab.io import neuralynx as nlx
from fklab.io.preprocessed_data import read_dataset
import support_replay_analysis.tools as tools
from neuronpy.util.spiketrain import coincident_spikes_correlogram
from fklab.utilities import yaml
from support_replay_analysis.tools import to_yaml

import ttl_burst_cc_config as config

len_cc = config.timewindow_ms*2+1
cc_start = np.zeros( len_cc )
cc_stop = np.zeros( len_cc )

n_just_outside = 0
n_ttl = 0
n_ttl_sleep = 0
n_ttl_postrun = 0
n_in_burst = 0
n_in_burst_sleep = 0
n_in_burst_postrun = 0

speed_non_burst = np.empty(0)
speed_burst = np.empty(0)

for d in config.days:  
    
    for e in config.epoch:      
        
        dirpath = join( config.preprocessed_root, 'dataset_'+config.dataset_month_year +\
                       str(d) + '_' + e +'/' )
        filelist = natsort.natsorted( [f for f in listdir(dirpath) if isfile(join(dirpath, f))] )
        path_to_preprocessed_dataset = join( dirpath, filelist[0] )
        path_to_nlx_eventdata = join( dirpath, "Events.nev" )
        
        # read online detection
        behav, ephys, start_stop = read_dataset( path_to_preprocessed_dataset ) 
        f_nlx_events = nlx.NlxOpen( path_to_nlx_eventdata )
        all_events  = f_nlx_events.readdata( start=start_stop[0], stop=start_stop[1] )
        if config.replay_string not in np.unique( all_events.eventstring ):
            raise ValueError("Replay event string " + config.replay_string + " not found")
        replay_ttl_times = all_events.time[all_events.eventstring == config.replay_string]
        
        candidate = pickle.load( open( join( dirname( path_to_preprocessed_dataset ),\
                                            "candidate_events.p"), 'r') )
        
        in_burst_mask = candidate.contains( replay_ttl_times )[0]
        n_in_burst += sum(in_burst_mask)
        
        if e == "postrun":
            n_ttl_postrun += len(replay_ttl_times)
            t_ttl_outside = replay_ttl_times[~in_burst_mask]
            t_ttl_inside = replay_ttl_times[in_burst_mask]
            speed_non_burst = np.hstack( (speed_non_burst, np.interp( t_ttl_outside,\
                                 behav['time'], behav['speed']) ) )
            speed_burst = np.hstack( (speed_burst, np.interp( t_ttl_inside,\
                                 behav['time'], behav['speed']) ) )
            n_in_burst_postrun += sum(in_burst_mask)
            
        if e == "sleep":
            n_ttl_sleep += len(replay_ttl_times)
            n_in_burst_sleep += sum(in_burst_mask)
        
        deltas = np.array([ tools.nearest_diff(candidate.stop, t_ttl)\
                    for t_ttl in replay_ttl_times])
        mask = candidate.contains( [replay_ttl_times[ttl_ind]\
                    for ttl_ind in np.nonzero( deltas < 0.02 ) ])[0]
        n_just_outside += sum(~mask)
        n_ttl += len(replay_ttl_times)
        
        
        cc_start += coincident_spikes_correlogram( replay_ttl_times*1e3,\
                  candidate.start*1e3, timewindow=config.timewindow_ms )
        
        cc_stop += coincident_spikes_correlogram( replay_ttl_times*1e3, candidate.stop*1e3,\
                    timewindow=config.timewindow_ms )

non_burst_values =  speed_non_burst[np.isfinite(speed_non_burst)]
in_burst_values =  speed_burst[np.isfinite(speed_burst)]


if config.save_to_yaml:
    
    n_in_burst = n_in_burst_sleep + n_in_burst_postrun
    
    yaml_variables = { "online_hits" : {
        "description" : "online generated triggers in relation to offline defined bursts",
        "all" : {
            "count" : n_ttl,
            "bursts" : {
                "count" : int(n_in_burst),
                "percent" : to_yaml( n_in_burst/n_ttl*100, u="%" ) } },
        "rest" : {
            "count" : n_ttl_sleep,
            "bursts" : {
                "count" : int(n_in_burst_sleep),
                "percent" : to_yaml( n_in_burst_sleep/n_ttl_sleep*100, u="%" ) } },
        "run2" : {
            "count" : n_ttl_postrun,
            "bursts" : {
                "count" : int(n_in_burst_postrun),
                "percent" : to_yaml( n_in_burst_postrun/n_ttl_postrun*100, u="%" ),
                "percent_at_low_speed" : to_yaml( sum(in_burst_values < config.speed_thr)/len(in_burst_values)*100, u='%' ) },
            "non_bursts" :{
                "percent_at_high_speed" : to_yaml( sum(non_burst_values > config.speed_thr)/len(non_burst_values)*100, u='%' )  }}
        } }
    
    with open( config.path_to_yaml_file, "a" ) as f:
        
        yaml.dump( yaml_variables, f, default_flow_style=False )
        
    print "YAML variables file appended to " + config.path_to_yaml_file


if config.plot:
    plt.figure(1)
    t = np.linspace(-config.timewindow_ms, config.timewindow_ms, num=len_cc)   
    plt.subplot(211)     
    plt.plot( t , cc_start )
    plt.xlabel( "time [ms]" )
    plt.ylabel( "average counts aross epochs" )
    plt.title( "Probability of online detection after burst onset" )
    plt.subplot(212)     
    plt.plot( t , cc_stop )
    plt.xlabel( "time [ms]" )
    plt.ylabel( "average counts aross epochs" )
    plt.title( "Probability of online detection after burst offset" )
    
    plt.figure(2)
    print "{0:.1f}% of ttl-associated speed values higher than 5 cm/s" .format(\
           sum(non_burst_values>config.speed_thr)/len(non_burst_values)*100)
    seaborn.kdeplot( non_burst_values, bw=.5)
    h, _, _ = plt.hist( non_burst_values, bins=50, normed=True )
    plt.vlines( np.nanpercentile(non_burst_values, [50]), 0, h.max() )
    plt.xlabel( "speed [m/s]" )
    plt.ylabel( "probability" )
    
    plt.figure(3)
    print "{0:.1f}% of ttl-associated speed values lower than 5 cm/s" .format(\
           sum(in_burst_values<config.speed_thr)/len(in_burst_values)*100)
    seaborn.kdeplot( in_burst_values, bw=.5)
    h, _, _ = plt.hist( in_burst_values, bins=50, normed=True )
    plt.vlines( np.nanpercentile( in_burst_values, [50]), 0, h.max() )
    plt.xlabel( "speed [m/s]" )
    plt.ylabel( "probability" )
    
    plt.show()
