# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:45:46 2015

@author: davide
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import join, dirname, isdir
import time
import cPickle as pickle

from falcon.io.deserialization import load_file

from fklab.io import neuralynx as nlx
from fklab.utilities import yaml
from fklab.io.preprocessed_data import read_dataset
from fklab.segments.segment import Segment as Seg

import support_replay_analysis.tools as tools

import config_file_extract_measurements as config

unknown_content_str = "replay_unknown_content" # sent out when a candidate event was detected online
undefined_content_str = "replay_undefined" # when x peak falls outside the content ranges 
BIN_SIZE_REPLAY = 0.01

if config.stop_after_plot and config.save_measurements:
    raise ValueError("Some of the measurements might not be calculated")

# load preprocessed data info
with open( join( dirname( config.path_to_preprocessed_dataset ),\
"preprocessing_info.yaml" ), 'r' ) as prepr_f:
    info_preprocess = yaml.load( prepr_f )
    
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

# read Falcon detections by reading the EventData serialized by the ReplayIdentifier
print "I'm extracting the detection bins that were serialized by the Replay Identifier"
event_data, event_data_header = load_file( config.path_to_eventdata )
assert( event_data_header['format'] == 'FULL' )
extension = config.path_to_eventdata[-4:]
# detection times receive an add bin_size_replay time to offset the detection time of 1 bin
# because the timestamp is the starting timestamp of the likelihood
if extension == 'yaml':
    detection_times_replay_identifier = np.array([\
    [el['data']['hardware_ts']/1e6 + BIN_SIZE_REPLAY, int(el['data']['event'][-1])]\
        for el in event_data if (el['data']['event'][10:] != unknown_content_str[10:]\
                 and el['data']['event'][10:] != undefined_content_str[10:]) ])
detected_content_replay_identifier = np.uint( detection_times_replay_identifier[:, 1])
detection_times_replay_identifier = np.delete( detection_times_replay_identifier, 1, 1).ravel()
sel_mask = Seg( start_stop ).contains( detection_times_replay_identifier )[0]
detection_times_replay_identifier = detection_times_replay_identifier[sel_mask]
detected_content_replay_identifier = detected_content_replay_identifier[sel_mask]

# read online detection
_, _, start_stop = read_dataset( config.path_to_preprocessed_dataset ) 
f_nlx_events = nlx.NlxOpen( config.path_to_nlx_eventdata )
all_events  = f_nlx_events.readdata( start=start_stop[0], stop=start_stop[1] )
if config.replay_string not in np.unique( all_events.eventstring ):
    raise ValueError("Replay event string " + config.replay_string + " not found")
replay_ttl_times = all_events.time[all_events.eventstring == config.replay_string]

# compute added latencies
falcon_added_latencies = np.array([ tools.nearest_diff(detection_times_replay_identifier, v)\
    for v in replay_ttl_times ])
print("\nMax added latency = {0:.2f} ms") .format(falcon_added_latencies.max()*1e3)
n_ttl = len(replay_ttl_times)
ret_contains = candidate.contains( replay_ttl_times )
replay_ttl_to_candidate = tools.seg_indices( ret_contains[0], ret_contains[1] )
# NAN for candidate events with no detection
absolute_latencies = np.array( [ (replay_ttl_times[el[0]] - candidate[i].start)[0]\
    if el[0]!=-1 else np.NaN for i, el in enumerate(ret_contains[2])] )
ttl_content = np.array( [ detected_content_replay_identifier[\
    np.argmin( np.abs(detection_times_replay_identifier - ttl_time ) )] - 1\
    for ttl_time  in replay_ttl_times ], dtype='int' )    

n_detections_in_candidate = sum( candidate.contains( replay_ttl_times )[1] )
print("\n{0:.2f} % of the online detected events with content happened during CANDIDATE events"\
    .format(n_detections_in_candidate / n_ttl * 100))

outside_detections = ttl_content[~candidate.contains( replay_ttl_times )[0]]
print "# false detection arm1: {0}" .format(np.sum(outside_detections == 0))
print "# false detection arm2: {0}" .format(np.sum(outside_detections == 1))
print "# false detection arm3: {0}" .format(np.sum(outside_detections == 2))

# adjust content nodes and grid to compensate for the extra linear distance added
grid_element_length = np.mean( np.diff( grid.ravel() ) )
new_content_nodes, equalized_content_nodes = tools.fix_content_nodes(\
    content_nodes, equalize=True,\
    arm_length_grid_units=np.int(info_preprocess["ArmLength_cm"]/grid_element_length+1 ) ) 
adjusted_grid = grid[:new_content_nodes[-1, -1]+1] 
assert( np.all(new_content_nodes >= 0) )

# prepare the storing of the measurements
linefit_scores = np.full( (n_candidate_events), np.NaN )
linefit_scores_1 = np.full( (n_candidate_events), np.NaN )
linefit_scores_2 = np.full( (n_candidate_events), np.NaN )
if config.compute_linefit_score_against_shuffles:
    linefit_vs_shuffling_scores = np.full( (n_candidate_events), np.NaN )
online_contents = np.full( n_candidate_events, np.NaN, dtype='int' )
b_scores = np.full( n_candidate_events, np.NaN )
b_scores_1 = np.full( n_candidate_events, np.NaN )
b_scores_2 = np.full( n_candidate_events, np.NaN )
bias_pvalues = np.full( n_candidate_events, np.NaN )
bias_pvalues_1 = np.full( n_candidate_events, np.NaN )
bias_pvalues_2 = np.full( n_candidate_events, np.NaN )
b_observed = np.full( n_candidate_events, np.NaN )
b_observed_1 = np.full( n_candidate_events, np.NaN )
b_observed_2 = np.full( n_candidate_events, np.NaN )
arm_max_bias = np.full( n_candidate_events, np.NaN )
arm_max_bias_1 = np.full( n_candidate_events, np.NaN )
arm_max_bias_2 = np.full( n_candidate_events, np.NaN )
relative_latencies = np.full( n_candidate_events, np.NaN )
slopes = np.full( (n_candidate_events), np.NaN )
slopes_1 = np.full( (n_candidate_events), np.NaN )
slopes_2 = np.full( (n_candidate_events), np.NaN )
r_observed = np.full( n_candidate_events, np.NaN )
r_observed_1 = np.full( n_candidate_events, np.NaN )
r_observed_2 = np.full( n_candidate_events, np.NaN )
r_scores = np.full( n_candidate_events, np.NaN )
r_scores_1 = np.full( n_candidate_events, np.NaN )
r_scores_2 = np.full( n_candidate_events, np.NaN )

# prepare plotting of event posteriors
if config.plot_event_posteriors:   
    max_n_plots_per_figure = config.n_vertical_plots * config.n_horiz_plots
    max_n_plots = min(config.max_n_events, n_candidate_events)
    print "\nI'm drawing the posteriors of the selected candidate events using {0} windows"\
        .format( int(max_n_plots/max_n_plots_per_figure)+1 )
    f = 1
    import seaborn
    seaborn.reset_orig()
    ylimit = adjusted_grid[-1] + adjusted_grid[0]
    fontsize = 12

# plot events and compute params
t0 = time.time()
for j in range(n_candidate_events):#loop over events  
        
    if j == 3:
        t1 = time.time()
        duration_3_cycles = t1 - t0
        print "Execution will take no more than about {0:.1f} minutes" .format(\
            n_candidate_events/3*duration_3_cycles/60 )
        
    plot_posteriors = config.plot_event_posteriors and j < max_n_plots
    
    if plot_posteriors:
        if (j%max_n_plots_per_figure) == 0:            
            plt.figure(f)
            f += 1
        ax = plt.subplot( config.n_vertical_plots, config.n_horiz_plots,\
            (j%max_n_plots_per_figure) + 1 )
    
    # prepare event posteriors 
    posteriors = posteriors_candidate[j]            
    posteriors = tools.adjust_posteriors( posteriors, equalized_content_nodes )
    
    # create two partitions    
    duration_bins = np.shape( posteriors )[0] 
    duration = duration_bins * BIN_SIZE_REPLAY * 1e3
    duration = np.round( duration, decimals=1 )
    half_duration_bins = int(duration_bins/2)
    if duration_bins%2 == 0:    
        posteriors_1 = posteriors[:half_duration_bins, :]
        posteriors_2 = posteriors[half_duration_bins:, :]
    else:
        posteriors_1 = posteriors[:half_duration_bins+1, :]
        posteriors_2 = posteriors[half_duration_bins:, :]
    
    # full event 
    b_scores[j], bias_pvalues[j], b_observed[j], arm_max_bias[j] = tools.compute_bias_score(
        posteriors, new_content_nodes, M=config.n_shuffles )
        
    line, score_per_arm, _, time_vector, time_bins, pos_vector, posteriors_single_content =\
        tools.find_line( posteriors, BIN_SIZE_REPLAY, adjusted_grid, new_content_nodes, content_sel=arm_max_bias[j]  )
    slopes[j] = (line[-1] - line[0]) / (time_vector[-1] - time_vector[0])
    linefit_scores[j] = score_per_arm[arm_max_bias[j]]
    line_plot = line
    time_bins_plot = time_bins
    
    norm_posterior_single_content =\
        posteriors_single_content / posteriors_single_content.sum(axis=1)[:,np.newaxis]
    corr = tools.Correlation( norm_posterior_single_content, time_vector, pos_vector )
    r_observed[j] = corr.correlation_resampling( r="pearson" )
    r_scores[j] = corr.sequence_score( r="pearson", r_observed=r_observed[j],\
            M=config.n_shuffles, N=2000 )
    
    # first half
    b_scores_1[j], bias_pvalues_1[j], b_observed_1[j], arm_max_bias_1[j] =\
        tools.compute_bias_score( posteriors_1, new_content_nodes, M=config.n_shuffles )
    
    line, score_per_arm, _, time_vector, time_bins, pos_vector, posteriors_single_content =\
        tools.find_line( posteriors_1, BIN_SIZE_REPLAY, adjusted_grid, new_content_nodes )
    slopes_1[j] = (line[-1] - line[0]) / (time_vector[-1] - time_vector[0])
    linefit_scores_1[j] = score_per_arm[arm_max_bias_1[j]]
    
    norm_posterior_single_content =\
        posteriors_single_content / posteriors_single_content.sum(axis=1)[:,np.newaxis]
    corr = tools.Correlation( norm_posterior_single_content, time_vector, pos_vector )
    r_observed_1[j] = corr.correlation_resampling( r="pearson" )
    r_scores_1[j] = corr.sequence_score( r="pearson", r_observed=r_observed_1[j],\
            M=config.n_shuffles, N=2000 )
    
    # second half
    b_scores_2[j], bias_pvalues_2[j], b_observed_2[j], arm_max_bias_2[j] =\
        tools.compute_bias_score( posteriors_2, new_content_nodes, M=config.n_shuffles )
    
    line, score_per_arm, _, time_vector, time_bins, pos_vector, posteriors_single_content =\
        tools.find_line( posteriors_2, BIN_SIZE_REPLAY, adjusted_grid, new_content_nodes )
    slopes_2[j] = (line[-1] - line[0]) / (time_vector[-1] - time_vector[0])
    linefit_scores_2[j] = score_per_arm[arm_max_bias_2[j]] 
    
    norm_posterior_single_content =\
        posteriors_single_content / posteriors_single_content.sum(axis=1)[:,np.newaxis]
    corr = tools.Correlation( norm_posterior_single_content, time_vector, pos_vector )
    r_observed_2[j] = corr.correlation_resampling( r="pearson" )
    r_scores_2[j] = corr.sequence_score( r="pearson", r_observed=r_observed_2[j],\
            M=config.n_shuffles, N=2000 )
    
    if config.compute_linefit_score_against_shuffles:
        linefit_vs_shuffling_scores[j], _, _ = tools.compute_linefit_Zscore_againstshuffle(\
              posteriors, time_vector, adjusted_grid, new_content_nodes, arm_max_bias_1[j],\
              n_shuffles=config.n_shuffles, num_cores=config.n_cores )
    
    if plot_posteriors:
        
        plt.imshow( posteriors.T, cmap=plt.cm.gray_r, aspect='auto', origin='lower',\
            interpolation='none', extent=[0, duration, 0, ylimit] ) 
        plt.autoscale(False)
        plt.hlines( adjusted_grid[new_content_nodes[1:, 0]], 0, duration,\
            color='black', linestyle="dotted" )
        plt.xlim( 0, duration )
        plt.ylim( 0, ylimit )
        plt.ylabel('position [cm]', fontsize=fontsize )
        plt.xlabel('time into burst [ms]', fontsize=fontsize )
        plt.xticks( [0, duration], fontsize=fontsize )
        plt.yticks( [0, int( np.round(adjusted_grid.max()) )],\
            fontsize=fontsize )
        plt.tick_params( left='off', right='off' )
        if config.plot_line:
            plt.plot( time_bins_plot, line_plot, 'm--', linewidth=2 )
        if config.plot_color_bar:
            cbar = plt.colorbar( label="probability", format="%.2f", shrink=.5 )
            cbar_ticks = np.array( [posteriors.min(), posteriors.max()] )
            cbar.set_ticks( cbar_ticks ) 
            cbar.ax.tick_params( labelsize=fontsize/3*2 )
            cbar.set_label( label="probability", size=fontsize/3*2 )  
            
    if not plot_posteriors and config.stop_after_plot: break
    
    arm_labels_positions = 1/n_contents/2 * np.arange(1,6,2)
    if plot_posteriors:
        for k, al_pos in enumerate(arm_labels_positions):
            plt.text( -0.05, al_pos, str(k+1), color='k', transform=ax.transAxes,\
                fontsize=10 ) 
        text1 = str(j)
        text2 = "lfs=" + str( np.round( linefit_scores[j], decimals=2 ) ) + "," +\
            str( np.round( linefit_scores_1[j], decimals=2 ) ) + "," +\
            str( np.round( linefit_scores_2[j], decimals=2 ) )
        text3 = " bZ=" + str( np.round( b_scores[j], decimals=1 ) ) + "," +\
            str( np.round( b_scores_1[j], decimals=1 ) ) + "," +\
            str( np.round( b_scores_2[j], decimals=1 ) )
        text4 = " rZ=" + str( np.round( r_scores[j], decimals=1 ) ) + "," +\
            str( np.round( r_scores_1[j], decimals=1 ) ) + "," +\
            str( np.round( r_scores_2[j], decimals=1 ) )
        plt.text( 0, 1.03, text1, color='blue', transform=ax.transAxes )
        plt.text( .1, 1.03, text2, color='k', transform=ax.transAxes, fontsize=6 )
        plt.text( .5, 1.03, text3, color='k', transform=ax.transAxes, fontsize=6 )
        plt.text( .9, 1.03, text4, color='k', transform=ax.transAxes, fontsize=6 )
       
    relative_latencies[j] = absolute_latencies[j] / candidate[j].duration[0]
        
    online_detection = -1
    if not np.isnan( absolute_latencies[j] ):
        temp = ttl_content[np.array(replay_ttl_to_candidate) == j]
        online_content = temp[0] # exclude secondary detections of the same event
        if plot_posteriors:
            plt.vlines( absolute_latencies[j]*1e3,\
                adjusted_grid[new_content_nodes[online_content, 0]],\
                adjusted_grid[new_content_nodes[online_content, 1]], color='red',\
                linewidth=2.5 )
        online_detection = online_content
                         
    online_contents[j] = online_detection                
    
        
if config.save_measurements:
    
    if not isdir( config.save_measurements_dump ):
        os.mkdir( config.save_measurements_dump )
        print "\nFolder created (" + config.save_measurements_dump + ")"
    
    arm_max_bias_filename = join( config.save_measurements_dump, "arm_max_bias.npy" )
    np.save( open( arm_max_bias_filename, 'w'), np.int8(arm_max_bias) )
    arm_max_bias_filename_1 = join( config.save_measurements_dump, "arm_max_bias_1.npy" )
    np.save( open( arm_max_bias_filename_1, 'w'), np.int8(arm_max_bias_1) )
    arm_max_bias_filename_2 = join( config.save_measurements_dump, "arm_max_bias_2.npy" )
    np.save( open( arm_max_bias_filename_2, 'w'), np.int8(arm_max_bias_2) )
    print "\nArm max bias saved in " + arm_max_bias_filename
    print "Arm max bias first segment saved in " + arm_max_bias_filename_1
    print "Arm max bias last segment saved in " + arm_max_bias_filename_2
    
    bias_measures_filename = join( config.save_measurements_dump, "bias_measures.npy" )
    np.save( open( bias_measures_filename, 'w'), b_observed )
    bias_measures_filename_1 = join( config.save_measurements_dump, "bias_measures_1.npy" )
    np.save( open( bias_measures_filename_1, 'w'), b_observed_1 )
    bias_measures_filename_2 = join( config.save_measurements_dump, "bias_measures_2.npy" )
    np.save( open( bias_measures_filename_2, 'w'), b_observed_2 )
    print "\nBias measures saved in " + bias_measures_filename
    print "Bias measures first segment saved in " + bias_measures_filename_1
    print "Bias measures last segment saved in " + bias_measures_filename_2
    
    bias_scores_filename = join( config.save_measurements_dump,\
                    "bias_scores_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_scores_filename, 'w'), b_scores )
    bias_scores_1_filename = join( config.save_measurements_dump,\
                     "bias_scores_1_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_scores_1_filename, 'w'), b_scores_1 )
    bias_scores_2_filename = join( config.save_measurements_dump,\
                      "bias_scores_2_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_scores_2_filename, 'w'), b_scores_2 )
    print "\nBias scores saved in " + bias_scores_filename
    print "Bias scores of first segment saved in " + bias_scores_1_filename
    print "Bias scores of last segment saved in " + bias_scores_2_filename
    
    bias_pvalues_filename = join( config.save_measurements_dump,\
                    "bias_pvalues_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_pvalues_filename, 'w'), bias_pvalues )
    bias_pvalues_1_filename = join( config.save_measurements_dump,\
                     "bias_pvalues_1_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_pvalues_1_filename, 'w'), bias_pvalues_1 )
    bias_pvalues_2_filename = join( config.save_measurements_dump,\
                      "bias_pvalues_2_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( bias_pvalues_2_filename, 'w'), bias_pvalues_2 )
    print "\nBias pvalues saved in " + bias_pvalues_filename
    print "Bias pvalues of first segment saved in " + bias_pvalues_1_filename
    print "Bias pvalues of last segment saved in " + bias_pvalues_2_filename
    
    linefit_scores_filename = join( config.save_measurements_dump,\
                       "linefit_scores.npy" )
    np.save( open( linefit_scores_filename, 'w'), linefit_scores )
    linefit_scores_1_filename = join( config.save_measurements_dump,\
                     "linefit_scores_1.npy" )
    linefit_scores_2_filename = join( config.save_measurements_dump,\
                     "linefit_scores_2.npy" )
    np.save( open( linefit_scores_1_filename, 'w'), linefit_scores_1 )
    np.save( open( linefit_scores_2_filename, 'w'), linefit_scores_2 )
    print "\nLine-fit scores saved in " + linefit_scores_filename
    print "Line-fit scores first segment saved in " + linefit_scores_1_filename
    print "Line-fit scores last segment saved in " + linefit_scores_2_filename
    
    slopes_filename = join( config.save_measurements_dump, "slopes.npy" )
    np.save( open( slopes_filename, 'w'), slopes )
    slopes_1_filename = join( config.save_measurements_dump, "slopes_1.npy" )
    np.save( open( slopes_1_filename, 'w'), slopes_1 )
    slopes_2_filename = join( config.save_measurements_dump, "slopes_2.npy" )
    np.save( open( slopes_2_filename, 'w'), slopes_2 )
    print "\nSlopes first segment saved in " + slopes_1_filename
    print "Slopes last segment saved in " + slopes_2_filename
    print "Slopes saved in " + slopes_filename
    
    correlation_measures_filename = join( config.save_measurements_dump, "r_pearson.npy" )
    np.save( open( correlation_measures_filename, 'w'), r_observed )
    correlation_measures_filename_1 = join( config.save_measurements_dump, "r_pearson_1.npy" )
    np.save( open( correlation_measures_filename_1, 'w'), r_observed_1 )
    correlation_measures_filename_2 = join( config.save_measurements_dump, "r_pearson_2.npy" )
    np.save( open( correlation_measures_filename_2, 'w'), r_observed_2 )
    print "\nBias measures saved in " + correlation_measures_filename
    print "Correlation measures first segment saved in " + correlation_measures_filename_1
    print "Correlation measures last segment saved in " + correlation_measures_filename_2
    
    correlation_scores_filename = join( config.save_measurements_dump,\
                    "sequence_scores_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( correlation_scores_filename, 'w'), r_scores )
    correlation_scores_1_filename = join( config.save_measurements_dump,\
                     "sequence_scores_1_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( correlation_scores_1_filename, 'w'), r_scores_1 )
    correlation_scores_2_filename = join( config.save_measurements_dump,\
                      "sequence_scores_2_" + str(config.n_shuffles) + "sh.npy" )
    np.save( open( correlation_scores_2_filename, 'w'), r_scores_2 )
    print "\Sequence scores saved in " + correlation_scores_filename
    print "Sequence scores of first segment saved in " + correlation_scores_1_filename
    print "Sequence scores of last segment saved in " + correlation_scores_2_filename
    
    if config.compute_linefit_score_against_shuffles:
        linefit_scores_against_null_filename = join( config.save_measurements_dump,\
            "linefit_scores_against_null" + str(config.n_shuffles) + "sh.npy" )
        np.save( open( linefit_scores_against_null_filename, 'w' ), linefit_vs_shuffling_scores )
        print "\nLine-fit Z-scores against null distribution were saved in " +\
            linefit_scores_against_null_filename
            
    if not isdir( config.latencies_filepath_dump ):
        os.mkdir( config.latencies_filepath_dump )
        print "\nFolder created (" + config.latencies_filepath_dump + ")"
    absolute_latencies_filename = join( config.latencies_filepath_dump, "absolute.npy" )
    np.save( open( absolute_latencies_filename, 'w' ), absolute_latencies )
    print "\nAbsolute latencies saved in " + absolute_latencies_filename
    
    relative_latencies_filename = join( config.latencies_filepath_dump, "relative.npy" )
    np.save( open( relative_latencies_filename, 'w' ), relative_latencies )
    print "\nRelative latencies saved in " + relative_latencies_filename
    
if config.save_added_latencies:
    added_latencies_filename = join( config.latencies_filepath_dump, "added.npy" )
    np.save( open( added_latencies_filename, 'w' ), falcon_added_latencies )
    print "\nAdded latencies saved in " + added_latencies_filename
    
if config.save_online_content:
    online_content_filename = join( config.online_content_filepath_dump, "online_content.npy" )
    np.save( open( online_content_filename, 'w'), online_contents )
    print "\nOnline content saved in " + online_content_filename

if config.save_final_content_nodes:
    filepath_final_content_nodes = join( config.content_nodes_filepath_dump,\
                        "final_content_nodes.npy" )
    np.save( open( filepath_final_content_nodes, 'w' ), new_content_nodes )
    print "\nFinal content nodes saved in " + filepath_final_content_nodes


plt.show()
