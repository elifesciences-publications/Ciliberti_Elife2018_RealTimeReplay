# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:45:46 2015

@author: davide
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import join, dirname, pardir, abspath
import time
import natsort
import os
import os.path as op
import cPickle as pickle

from falcon.io.deserialization import load_file

from fklab.io import neuralynx as nlx
from fklab.utilities import yaml
from fklab.io.preprocessed_data import read_dataset

from replay_identification.detection_performance import PerformanceMeter
import support_replay_analysis.tools as tools

import config_file_valid_with_decoding as config


unknown_content_str = "replay_unknown_content" # sent out when a candidate event was detected online
BIN_SIZE_REPLAY = 0.01

path_simulators =\
"/home/davide/Dropbox/Davide/Replay_Analysis/variables_online_replayidentification/line_fitting_reference"
MUA_THR = 2.5
paths = natsort.natsorted( os.listdir(path_simulators) )
paths_to_simulators = [ pa for pa in paths if "simulator" in pa]
for j in range(3):
    i=j*2
    paths_to_simulators[i], paths_to_simulators[i+1] = paths_to_simulators[i+1], paths_to_simulators[i]
simulators = [ pickle.load( open( op.join(path_simulators, sim_path), 'r') )\
              for sim_path in paths_to_simulators ]
print "\nSimulators loaded"

characterization = pickle.load( open(
    "/home/davide/Dropbox/Davide/Replay_Analysis/event_characteristics.p", 'r') )
session_indices = [0] + np.cumsum(
    [ len(sim.candidate_events) for sim in simulators ]).tolist()
reference_content = np.copy( np.array( characterization.get( "arm max bias" ) ) )
mask_undetected = np.copy( np.logical_not(characterization.get( "has_content" )) )
reference_content[mask_undetected] = -1

# load preprocessed data info
with open( join( dirname( config.path_to_preprocessed_dataset ),\
"preprocessing_info.yaml" ), 'r' ) as prepr_f:
    info_preprocess = yaml.load(prepr_f)
    
# load preprocess data
_, _, start_stop = read_dataset( config.path_to_preprocessed_dataset )

# load grid, content nodes and MUA
print "\nI'm loading the grid, the content nodes and the MUA"
xgrid_vector_pixel = np.load( join( config.path_to_environment, "grid.npy" ) )
xgrid_vector_cm = xgrid_vector_pixel / info_preprocess["ConversionFactorPixelsToCm"]
grid = xgrid_vector_cm.reshape( (len(xgrid_vector_cm),1) )
content_nodes = np.load( join ( config.path_to_environment, "content_nodes.npy" ) )
n_contents = len(content_nodes)

print "I'm loading the posteriors of the candidate events"
posteriors_candidate = np.load( join( dirname( config.path_to_preprocessed_dataset ),\
    "posteriors_candidate_events.npy") )
candidate = pickle.load( open( join( dirname( config.path_to_preprocessed_dataset ),\
    "candidate_events.p"), 'r') )
n_candidate_events = len(posteriors_candidate)
assert( n_candidate_events == len(candidate) )
correct_posteriors = all([ np.isclose( 1, np.mean( np.sum(posteriors_candidate[i],\
    axis=1) ) ) for i in range(n_candidate_events) ])
if not correct_posteriors:
    raise ValueError("At least one posterior did not sum to 1")

# extract Falcon detections by reading the EventData serialized by the ReplayIdentifier
print "I'm extracting the detection bins that were serialized by the Replay Identifier"
session_path = listdir(abspath( join( config.path_to_environment, pardir,\
    pardir, pardir, "output") ) )[0]
path_to_eventdata = abspath( join( config.path_to_environment, pardir,\
    pardir, pardir, "output", session_path, "sink_replay/sink_replay.0_replayidentifier.events.0.yaml" ) )
event_data, event_data_header = load_file( path_to_eventdata )
assert( event_data_header['format'] == 'FULL' )
extension = path_to_eventdata[-4:]
# detection times receive an add bin_size_replay time to offset the detection time of 1 bin
# because the timestamp is the starting timestamp of the likelihood
if extension == 'yaml':
    detection_times_replay_identifier = np.array([\
    [el['data']['hardware_ts']/1e6 + BIN_SIZE_REPLAY, int(el['data']['event'][-1])]\
        for el in event_data if el['data']['event'][10:] != unknown_content_str[10:] ])
detected_content_replay_identifier = np.uint( detection_times_replay_identifier[:, 1])
detection_times_replay_identifier = np.delete( detection_times_replay_identifier, 1, 1).ravel()

_, _, start_stop = read_dataset( config.path_to_preprocessed_dataset ) 
f_nlx_events = nlx.NlxOpen( config.path_to_nlx_eventdata )
all_events  = f_nlx_events.readdata( start=start_stop[0], stop=start_stop[1] )
if config.replay_string not in np.unique( all_events.eventstring ):
    raise ValueError("Replay event string " + config.replay_string + " not found")
replay_ttl_times = all_events.time[all_events.eventstring == config.replay_string]

falcon_added_latencies_ms = np.array([ tools.nearest_diff(detection_times_replay_identifier, v)\
    for v in replay_ttl_times ]) * 1e3
print("\nMax added latency = {0:.2f} ms") .format(falcon_added_latencies_ms.max())
n_ttl = len(replay_ttl_times)
ret_contains = candidate.contains( replay_ttl_times )
replay_ttl_to_candidate = tools.seg_indices( ret_contains[0], ret_contains[1] )
latencies_ms_per_candidate = [ 1e3*(replay_ttl_times[el[0]] - candidate[i].start)[0]\
    if el[0]!=-1 else np.NaN for i, el in enumerate(ret_contains[2])]
ttl_content = np.array( [ detected_content_replay_identifier[\
    np.argmin( np.abs(detection_times_replay_identifier - ttl_time ) )] - 1\
    for ttl_time  in replay_ttl_times ], dtype='int' )    

n_detections_in_candidate = sum( candidate.contains( replay_ttl_times )[1] )
print("\n{0:.2f} % of the online detected events with content happened during CANDIDATE events"\
    .format(n_detections_in_candidate / n_ttl * 100))

grid_element_length = np.mean( np.diff( grid.ravel() ) )
new_content_nodes, equalized_content_nodes = tools.fix_content_nodes(\
    content_nodes, equalize=True,\
    arm_length_grid_units=np.int(info_preprocess["ArmLength_cm"]/grid_element_length+1 ) ) # remove meaningless pos
adjusted_grid = grid[:new_content_nodes[-1, -1]+1] 
assert( np.all(new_content_nodes >= 0))

# prepare plot event posteriors
f = 1
max_n_plots_per_figure = config.n_vertical_plots * config.n_horiz_plots
max_n_plots = min(config.max_n_events, n_candidate_events)

# parameters validation online detection
performance = PerformanceMeter( n_contents )

idx_unique_ttl = 0


# prepare plotting
if config.plot_event_posteriors:     
    print "\nI'm drawing the posteriors of the selected candidate events using {0} windows"\
        .format( int(max_n_plots/max_n_plots_per_figure)+1 )
    import seaborn
    seaborn.reset_orig()
    ylimit = adjusted_grid[-1] + adjusted_grid[0]
    fontsize = 16

        
for j in range(max_n_plots):#loop over events  
    
    if j == 0:
        t0 = time.time()
        
    if j == 3:
        t1 = time.time()
        duration_3_cycles = t1 - t0
        print "Execution will take about {0:.1f} more minutes" .format(\
            max_n_plots/3*duration_3_cycles/60 )
    
    if config.plot_event_posteriors:
        if (j%max_n_plots_per_figure) == 0:            
            plt.figure(f)
            f += 1
        ax = plt.subplot( config.n_vertical_plots, config.n_horiz_plots,\
            (j%max_n_plots_per_figure) + 1 )
    
    posteriors = posteriors_candidate[j]            
    posteriors = tools.adjust_posteriors( posteriors, equalized_content_nodes )    
        
    ref_content = reference_content[j]
    linefit_score = characterization.get( "linefit_score" )[j]
    bias_score = characterization.get( "bZ" )[j]
        
    duration = np.shape(posteriors)[0] * BIN_SIZE_REPLAY * 1e3
    duration = np.round( duration, decimals=1 )
    
    if config.plot_event_posteriors:
        
        plt.imshow( posteriors.T, cmap=plt.cm.gray_r, aspect='auto', origin='lower',\
            interpolation='none', extent=[0, duration, 0, ylimit] ) 
        plt.autoscale(False)
            
        plt.hlines( adjusted_grid[new_content_nodes[1:, 0]], 0, duration,\
            color='brown' )
            
        plt.xlim( 0, duration )
        plt.ylim( 0, ylimit )
        
        plt.ylabel('position [cm]', fontsize=fontsize )
        plt.xlabel('time into burst [ms]', fontsize=fontsize )
        
        plt.xticks( [0, duration], fontsize=fontsize )
        plt.yticks( [0, int( np.round(adjusted_grid.max()) )],\
            fontsize=fontsize )
        plt.tick_params( left='off', right='off' )
        
        line, score_per_arm, ref_content, time_vector, time_bins, pos_vector, posteriors_1content =\
            tools.find_line( posteriors, BIN_SIZE_REPLAY, adjusted_grid,\
                            new_content_nodes )
           
        plt.plot( time_bins, line, 'm--', linewidth=3 )
        
        if config.plot_color_bar:
            cbar = plt.colorbar( label="probability", format="%.2f", shrink=.5 )
            cbar_ticks = np.array( [posteriors.min(), posteriors.max()] )
            cbar.set_ticks( cbar_ticks ) 
            cbar.ax.tick_params( labelsize=fontsize/3*2 )
            cbar.set_label( label="probability", size=fontsize/3*2 )  

    assert( not np.isnan( ref_content ) )
    
    arm_labels_positions = 1/n_contents/2 * np.arange(1,6,2)
    if config.plot_event_posteriors:
        for k, al_pos in enumerate(arm_labels_positions):
            plt.text( -0.15, al_pos, "Arm"+str(k+1), color='k', transform=ax.transAxes,\
                fontsize=10 ) 

    text1 = "linefit_score=" + str(np.round(characterization.get("linefit_score"),\
        decimals=3) ) + "; bias Z-score=" + str( np.round(
        characterization.get("bZ")[j], decimals=1 ) )
    text2 = "c=" + str(ref_content+1)
    if ref_content > -1:
        if config.plot_event_posteriors:
            plt.text( .35, 1.03, str(j), color='red', transform=ax.transAxes )
            plt.text( .45, 1.03, text1, color='k', transform=ax.transAxes )
            plt.text( 0.1, 1.03, text2, color='k', transform=ax.transAxes )
            
        offline_detection = ref_content 
    else:
        if config.plot_event_posteriors:
            plt.text( .35, 1.03, str(j), color='blue', transform=ax.transAxes )
            plt.text( .45, 1.03, text1, color='k', transform=ax.transAxes )
            plt.text( 0.1, 1.03, text2 + " n.s.", color='k', transform=ax.transAxes )
        offline_detection = -1

       
    online_detection = -1
    if not np.isnan( latencies_ms_per_candidate[j] ):
        temp = ttl_content[np.array(replay_ttl_to_candidate) == j]
        online_content = temp[0] # exclude secondary detections of the same event
        if config.plot_event_posteriors:
            plt.vlines( latencies_ms_per_candidate[j],\
                adjusted_grid[new_content_nodes[online_content, 0]],\
                adjusted_grid[new_content_nodes[online_content, 1]], color='red',\
                linewidth=2.5 )
        online_detection = online_content

    performance.update( offline_detection, online_detection, latencies_ms_per_candidate[j],\
        candidate[j].duration[0]*1e3, only_tp_latencies=config.only_tp_latencies )
    
        
performance.print_results()
performance.plot_latency_result( f+1 )

plt.show()
