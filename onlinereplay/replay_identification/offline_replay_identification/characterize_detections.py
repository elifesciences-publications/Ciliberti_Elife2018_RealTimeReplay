#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:22:19 2017

@author: davide
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from os.path import join, dirname
from scipy.stats import cumfreq
from fklab.segments import Segment
from fklab.plot.plotting import noline_legend
import seaborn
import pandas as pd
import cPickle as pickle
from fklab.utilities import yaml

from support_replay_analysis.plot import configure_latency_plots
import support_replay_analysis.tools as tools
from support_replay_analysis.tools import to_yaml, iqr, ci90, range_
from replay_identification.detection_performance import PerformanceMeterJoint 

import characterize_detections_config as config


fs = 16
colors = seaborn.color_palette( "cubehelix", n_colors=4 )
loc = "upper left"
lw = 2


def plot_score_cumdistr( measurements, detected, xlabel, xticks=None, nbins=100, compute_random=False, n_shuffles=500, fign=1 ):
    '''
    '''
    global fs, colors, loc, lw
    
    plt.rc( 'legend',fontsize=fs )
    plt.rc( 'axes', linewidth=lw/2 )
    plt.rc( 'xtick.major', width=lw/2 )
    plt.rc( 'ytick.major', width=lw/2 )
    
    if compute_random:
        prob_online_detection = np.sum(detected) / len(detected)
        cum = np.zeros( (nbins, n_shuffles) )
        for n in range(n_shuffles):
            randomly_detected = np.random.choice( [False, True], size=(len(detected),),\
                             p=[1-prob_online_detection, prob_online_detection] )
            cum[:,n] = cumfreq( measurements[randomly_detected], nbins )[0]/np.sum(randomly_detected)
            
        random_median = np.median( cum, axis=1 )
        random_p5 = np.percentile( cum, 5, axis=1 )
        random_p95 = np.percentile( cum, 95, axis=1 )
    
    fig = plt.figure( fign )
    not_detected = np.logical_not( detected )
    ax = fig.add_subplot( 111 )
    x_det = np.linspace( np.min(measurements[detected]), np.max(measurements[detected]), num=nbins )
    x_ndet = np.linspace( np.min(measurements[not_detected]), np.max(measurements[not_detected]), num=nbins )
    ax.plot( x_det, cumfreq( measurements[detected], nbins )[0]/np.sum(detected),\
             label="w/ content detected online", color='red')
    ax.plot( x_ndet, cumfreq( measurements[not_detected], nbins )[0]/np.sum(not_detected),\
             label="w/o content detected online", color='blue')
    if compute_random:
        ax.plot( x_ndet, random_median, label="w/ randomly detected content", color='black')
        ax.fill_between( x_ndet, random_p5, random_p95, facecolor='gray', alpha=.15 )
    
    
    plt.xlabel( xlabel, fontsize= fs )
    plt.ylabel( "cumulative fraction", fontsize=fs )
    plt.xlim( xmin=0 )
    plt.ylim( ymin=0, ymax=1 )
    
    noline_legend( ax.legend( loc=loc, frameon=False ) )
        
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    ax.yaxis.set_ticks( [0, .5, 1] )
    ax.yaxis.set_ticklabels( [str(0), str(0.5), str(1)] )
    ax.tick_params( labelsize=fs, axis='both', which='major', pad=7.5  )
    if xticks is not None:
        ax.xaxis.set_ticks( xticks )
    ax.xaxis.set_tick_params( direction='in' )
    ax.yaxis.set_tick_params( direction='in' )

    median = np.median( measurements[detected] )
    plt.vlines( median, 0, .5, color='red', linestyles='dashed', alpha=.25 )
    plt.hlines( .5, ax.get_xlim()[0], median, color='red', linestyles='dashed', alpha=.25 )
    median = np.median( measurements[not_detected] )
    plt.vlines( median,0, .5, color='blue', linestyles='dashed', alpha=.25 )
    plt.hlines( .5, ax.get_xlim()[0], median, color='blue', linestyles='dashed', alpha=.25 )


def plot_absolute_latency_cumdistr( latencies, labels, nbins=100, fign=3 ):
    '''
    plot cumulative distributions of a set of labeled absolute latencies
    '''
    global fs, colors, loc
    
    plt.rc( 'legend',fontsize=fs )
    fig = plt.figure( fign )
    ax = fig.add_subplot( 111 )
    configure_latency_plots( ax, latencies, labels, nbins,\
            seaborn.color_palette( "cubehelix", n_colors=2 ), loc=loc, fs=fs )
    plt.xlabel( "latency from event onset [ms]", fontsize=fs )
    ax.xaxis.set_ticks( [0, 50, 100, 150, 200] )
    

def plot_relative_latency_cumdistr( latencies, labels, nbins=100, fign=3 ):
    '''
    plot cumulative distributions of a set of labeled relative latencies
    '''
    global fs, colors, loc
    
    plt.rc( 'legend',fontsize=fs )
    fig = plt.figure( fign )
    ax = fig.add_subplot( 111 )
    configure_latency_plots( ax, latencies, labels, nbins,\
        seaborn.color_palette( "cubehelix", n_colors=2 ), loc=loc, fs=fs )
    plt.xlabel( "latency relative to event duration [%]", fontsize=fs )
    ax.xaxis.set_ticks( np.linspace( 0, 100, num=5) )

arm_max_bias = np.empty(0, dtype=int )
arm_max_bias_1 = np.empty(0, dtype=int )
arm_max_bias_2 = np.empty(0, dtype=int )
bias = np.empty(0)
bias_1 = np.empty(0)
bias_2 = np.empty(0)
bZ = np.empty(0)
bZ_1 = np.empty(0)
bZ_2 = np.empty(0)
bP = np.empty(0)
bP_1 = np.empty(0)
bP_2 = np.empty(0)
lfs = np.empty(0)
lfs_1 = np.empty(0)
lfs_2 = np.empty(0)
lfZ = np.empty(0)
online_contents = np.empty(0)
absolute_latencies = np.empty(0)
relative_latencies = np.empty(0)
added_latencies = np.empty(0)
absolute_latencies_sleep = np.empty(0)
absolute_latencies_postrun = np.empty(0)
added_latencies_sleep = np.empty(0)
added_latencies_postrun = np.empty(0)
relative_latencies_sleep = np.empty(0)
relative_latencies_postrun = np.empty(0)
event_durations = np.empty(0)
rZ = np.empty(0)
rZ_1 = np.empty(0)
rZ_2 = np.empty(0)
r = np.empty(0)
r_1 = np.empty(0)
r_2 = np.empty(0)

lo_events = []

candidate_events = Segment([])
in_sleep = np.empty( 0, dtype='bool' )

for d in config.days:  
    
    for e in config.epoch:      
        
        dirpath = join( config.analysis_root, "dataset_" + config.dataset_month_year+"%02d"%d, e,\
                       "offline_posteriors" )
        
        arm_max_bias = np.hstack( ( arm_max_bias,  np.load( join( dirpath,\
                                                 "arm_max_bias.npy" ) ) ) )
        arm_max_bias_1 = np.hstack( ( arm_max_bias_1,  np.load( join( dirpath,\
                                                 "arm_max_bias_1.npy" ) ) ) )
        arm_max_bias_2 = np.hstack( ( arm_max_bias_2,  np.load( join( dirpath,\
                                                 "arm_max_bias_2.npy" ) ) ) )
        
        bias = np.hstack( ( bias,  np.load( join( dirpath, "bias_measures.npy" ) ) ) )
        bias_1 = np.hstack( ( bias_1,  np.load( join( dirpath, "bias_measures_1.npy" ) ) ) )
        bias_2 = np.hstack( ( bias_2,  np.load( join( dirpath, "bias_measures_2.npy" ) ) ) )
        
        bZ = np.hstack( ( bZ, np.load( join( dirpath, "bias_scores_2000sh.npy" ) ) ) )
        bZ_1 = np.hstack( ( bZ_1, np.load( join( dirpath, "bias_scores_1_2000sh.npy" ) ) ) )
        bZ_2 = np.hstack( ( bZ_2, np.load( join( dirpath, "bias_scores_2_2000sh.npy" ) ) ) )
        
        bP = np.hstack( ( bP, np.load( join( dirpath, "bias_pvalues_2000sh.npy" ) ) ) )
        bP_1 = np.hstack( ( bP_1, np.load( join( dirpath, "bias_pvalues_1_2000sh.npy" ) ) ) )
        bP_2 = np.hstack( ( bP_2, np.load( join( dirpath, "bias_pvalues_2_2000sh.npy" ) ) ) )
        
        lfs = np.hstack( ( lfs, np.load( join( dirpath, "linefit_scores.npy" ) ) ) )
        lfs_1 = np.hstack( ( lfs_1, np.load( join( dirpath, "linefit_scores_1.npy" ) ) ) )
        lfs_2 = np.hstack( ( lfs_2, np.load( join( dirpath, "linefit_scores_2.npy" ) ) ) )
        
        r = np.hstack( ( r, np.load( join( dirpath, "r_pearson.npy" ) ) ) )
        r_1 = np.hstack( ( r_1, np.load( join( dirpath, "r_pearson_1.npy" ) ) ) )
        r_2 = np.hstack( ( r_2, np.load( join( dirpath, "r_pearson_2.npy" ) ) ) )
        
        rZ = np.hstack( ( rZ, np.load( join( dirpath, "sequence_scores_2000sh.npy" ) ) ) )
        rZ_1 = np.hstack( ( rZ_1, np.load( join( dirpath, "sequence_scores_1_2000sh.npy" ) ) ) )
        rZ_2 = np.hstack( ( rZ_2, np.load( join( dirpath, "sequence_scores_2_2000sh.npy" ) ) ) )
        
        try:
            lfZ_temp = np.load( join( dirpath, "linefit_scores_against_null2000sh.npy" ) )
            # computed for sub-posterior of content of max bias
            lfZ = np.hstack( (lfZ, lfZ_temp) )
        except:
            print "Linefit scores against null not present!"
            lfZ = np.hstack( (lfZ, np.zeros_like(bZ)) )
            
        if e == "sleep":
            abs_lat_sleep = np.load( join( dirname( dirpath ), "latencies",\
                                          "absolute.npy" ))
            rel_lat_sleep = np.load( join( dirname( dirpath ), "latencies",\
                                          "relative.npy" )) 
            add_lat_sleep = np.load( join( dirname( dirpath ), "latencies",\
                                          "added.npy" ))
            absolute_latencies_sleep = np.hstack( (absolute_latencies_sleep,\
                 abs_lat_sleep ))
            relative_latencies_sleep = np.hstack( (relative_latencies_sleep,\
                 rel_lat_sleep ))
            added_latencies_sleep = np.hstack( (added_latencies_sleep,\
                 add_lat_sleep ))
            absolute_latencies = np.hstack( (absolute_latencies, abs_lat_sleep) )  
            relative_latencies = np.hstack( (relative_latencies, rel_lat_sleep) ) 
            added_latencies = np.hstack( (added_latencies, add_lat_sleep) )
            in_sleep = np.hstack( (in_sleep, np.ones_like( abs_lat_sleep, dtype='bool' ) ) )
            
        if e == "postrun":
            abs_lat_postrun = np.load( join( dirname( dirpath ), "latencies",\
                                          "absolute.npy" ))
            rel_lat_postrun = np.load( join( dirname( dirpath ), "latencies",\
                                          "relative.npy" )) 
            add_lat_postrun = np.load( join( dirname( dirpath ), "latencies",\
                                          "added.npy" )) 
            absolute_latencies_postrun = np.hstack( (absolute_latencies_postrun,\
                 abs_lat_postrun ))
            relative_latencies_postrun = np.hstack( (relative_latencies_postrun,\
                 rel_lat_postrun ))
            added_latencies_postrun = np.hstack( (added_latencies_postrun,\
                 add_lat_postrun ))
            absolute_latencies = np.hstack( (absolute_latencies, abs_lat_postrun) )  
            relative_latencies = np.hstack( (relative_latencies, rel_lat_postrun) )
            added_latencies = np.hstack( (added_latencies, add_lat_postrun) )
            in_sleep = np.hstack( (in_sleep, np.zeros_like( abs_lat_postrun, dtype='bool' ) ) )
            
        path_to_candidate = join( config.preprocessed_root, "dataset_" +\
                 config.dataset_month_year+"%02d"%d+"_"+e, "candidate_events.p" )
        candidate_events += pickle.load( open(path_to_candidate, 'r') )
        
        online_contents = np.hstack( (online_contents, np.load( join(\
             dirname(dirpath), "online_content.npy" ) ) ) )
        
        n_events_epoch = len(np.load( join( dirpath, "bias_scores_2000sh.npy" ) ))
        for n in range(n_events_epoch):
            lo_events.append( [n, e, d] )
        
# process imported values
n_events = len(online_contents)
assert( n_events == len(lfs) )
n_events_sleep = len( absolute_latencies_sleep )
n_events_postrun = len( absolute_latencies_postrun )
assert( n_events_sleep + n_events_postrun == n_events)
online_contents = np.int16( online_contents )

detected = online_contents != -1
not_detected = np.logical_not( detected )

arm_max_bias_matrix = np.array( [ arm_max_bias, arm_max_bias_1, arm_max_bias_2 ] ).T

bZ_matrix = np.array( [ bZ, bZ_1, bZ_2 ] ).T
sel_event_measures = np.argmax( bZ_matrix, axis=1 )
event_bZ = np.max( bZ_matrix, axis=1 )
assert( np.all(np.diag( bZ_matrix.take( sel_event_measures, axis=1) ) == event_bZ) )

r_matrix = np.abs( np.array( [ r, r_1, r_2 ] ).T )
sel_event_measures = np.argmax( r_matrix, axis=1 )
event_r = np.max( r_matrix, axis=1 )
assert( np.all(np.diag( r_matrix.take( sel_event_measures, axis=1) ) == event_r) )

rZ_matrix = np.array( [ rZ, rZ_1, rZ_2 ] ).T
sel_event_measures = np.argmax( rZ_matrix, axis=1 )
event_rZ = np.max( rZ_matrix, axis=1 )
assert( np.all(np.diag( rZ_matrix.take( sel_event_measures, axis=1) ) == event_rZ) )

bias_matrix = np.array( [bias, bias_1, bias_2] ).T
event_bias = np.diag( bias_matrix.take( sel_event_measures, axis=1 ) )

bP_matrix = np.array( [bP, bP_1, bP_2] ).T
event_bias_pvalue = np.diag( bP_matrix.take( sel_event_measures, axis=1 ) )

lfs_matrix = np.array([lfs, lfs_1, lfs_2]).T
event_lfs = np.diag( lfs_matrix.take( sel_event_measures, axis=1) )


# compute and print collective performance
pmj = PerformanceMeterJoint( 3, candidate_events, arm_max_bias_matrix, bZ_matrix, lfs_matrix, rZ_matrix)
pmj.set_reference_thresholds( min_bias_score=config.MIN_BIAS_SCORE, min_lf_score=config.MIN_LF_SCORE,\
         min_duration_joint=config.MIN_DURATION_JOINT, min_sequence_score=config.MIN_SEQUENCE_SCORE)        
tp, fp, fn, tn, cc, ic = pmj.compute_performance( online_contents, relative_latencies )

cm =  np.array( [[tp, fp.sum()], [fn.sum(), tn] ] )      
cmo = tools.ConfusionMatrixObject( cm )    
sens, spec, precision, accuracy, ll_ratio_pos, ll_ratio_neg = tools.compute_stats( cm )
yi = sens + spec - 1
inter_content_accuracy = cc.sum()/(cc.sum()+ic.sum())
dor, (dor_95ci_lower, dor_95ci_upper) = tools.compute_dor( cm )
rr, (rr_95ci_lower, rr_95ci_upper) = tools.compute_relative_risk( cm )

print pmj.two_semi

# create pandas frame
event_measures = {\
            "normalized bias" : event_bias,\
            "line-fit score" : event_lfs,\
            "bias-againstnull Z-score" : event_bZ,\
            "sequnce score" : event_rZ,\
            "Pearson correlation score" : event_r,\
            "latency absolute [ms]" : absolute_latencies*1e3,\
            "latency relative" : relative_latencies,\
            "event duration [ms]" : candidate_events.duration*1e3,\
            "bias P-value" : event_bias_pvalue,\
            "single content" : pmj.is_single,\
            "joint" : pmj.is_joint,\
            "online content" : online_contents }
event_measures.update( {"intra-epoch index":np.array(lo_events)[:, 0],\
    "epoch":np.array(lo_events)[:, 1], "session #":np.array(lo_events)[:, 2]})
dataframe = pd.DataFrame( event_measures )

assert sum(pmj.is_joint)+sum(pmj.is_single) == pmj.n_events_with_content
                                
print "\nLine-fit score min threshold: {0:.3f}" .format(config.MIN_LF_SCORE)
print "bZ min threshold: {0:.2f}" .format(config.MIN_BIAS_SCORE)
print "rZ min threshold: {0:.2f}" .format(config.MIN_SEQUENCE_SCORE)
print "min duration joint event: {0} ms" .format(int(config.MIN_DURATION_JOINT*1000))
print "\nSensitivity: {0:.2f} %" .format(sens*100)
print "Specificity: {0:.2f} %" .format(spec*100)
print "Youden's index: {0:.2f} %" .format(yi*100)
print "Diagnostic odds ratio: {0:.1f} [{1:.1f}, {2:.1f}]" .format(dor, dor_95ci_lower, dor_95ci_upper)
print "Risk ratio: {0:.1f} [{1:.1f}, {2:.1f}]" .format( rr, rr_95ci_lower, rr_95ci_upper)
print "# events with content: {0}" .format(pmj.n_events_with_content)
print "Precision (excluding non-candidate detections): {0:.2f} %" .format(precision*100)
print "Negative predictive value: {0:.2f} %" .format(cmo.npv*100 )
print "Markedness: {0:.2f} %" .format(cmo.markedness*100)
print "Matthew Correlation Coefficient: {0:.3f}" .format(cmo.mcc)
print "fraction events with content: {0:.2f} %" .format(100*pmj.n_events_with_content/n_events)
print "(of which {0:.2f} % are single and {1:.2f} % are joint events)" .format(\
       100*sum(pmj.is_single)/pmj.n_events_with_content, 100*sum(pmj.is_joint)/pmj.n_events_with_content)
print "Inter-content accuracy: {0:.2f} %" .format(inter_content_accuracy*100)
print "\nMedian absolute latency sleep: {0:.1f} ms" .format( np.nanmedian(absolute_latencies_sleep)*1e3)
print "Median absolute latency run: {0:.1f} ms" .format( np.nanmedian(absolute_latencies_postrun)*1e3)
print "Median relative latency sleep: {0:.1f} %" .format( np.nanmedian(relative_latencies_sleep)*1e2)
print "Median relative latency run: {0:.1f} %" .format( np.nanmedian(relative_latencies_postrun)*1e2)

print "CC", cc
print "IC", ic
print "FP", fp
print "FN", fn
print "TN", tn


if config.random_cm_day is not None:
    
    assert( len(config.days)==1 )
    assert( config.days[0] == config.random_cm_day )
    assert( len(config.epoch) == 1 )
    assert( config.epoch[0] == config.random_cm_epoch )
    
    random_cm = tools.create_cm_random( online_contents, pmj, relative_latencies, config.n_random_cm )
    random_cm_objects = np.array([tools.ConfusionMatrixObject( random_cm[:,:,i] )\
         for i in range(config.n_random_cm) ] )
    random_yi = np.array( [rcmo.yi for rcmo in random_cm_objects] )
    random_tpr = np.array( [rcmo.tpr for rcmo in random_cm_objects] )
    random_fpr = np.array( [rcmo.fpr for rcmo in random_cm_objects] )
    random_markedness = np.array( [rcmo.markedness for rcmo in random_cm_objects] )
    random_mcc = np.array( [rcmo.mcc for rcmo in random_cm_objects] )
    if config.save_random_cm:
        np.save( open(config.path_to_random_cm + ".npy", 'w'), random_cm_objects )
        print "Confusion matrix objects for random detections created and saved in " + config.path_to_random_cm
    
in_run = ~in_sleep
event_durations_sleep = absolute_latencies_sleep / relative_latencies_sleep
event_durations_postrun = absolute_latencies_postrun / relative_latencies_postrun
long_bursts_mask_all = candidate_events.duration > config.MIN_DURATION_JOINT
long_bursts_mask_sleep = event_durations_sleep > config.MIN_DURATION_JOINT
long_bursts_mask_postrun = event_durations_postrun > config.MIN_DURATION_JOINT    
has_replay = np.logical_or( pmj.is_single, pmj.is_joint )
long_replay_bursts_mask_all = np.logical_and( has_replay, long_bursts_mask_all )   
n_bursts_with_replay = has_replay.sum()  
has_replay_sleep = np.logical_and( has_replay, in_sleep )
has_replay_run = np.logical_and( has_replay, in_run )

# SAVING
if config.save_reference:
    filepath = pickle.dump( pmj, open( config.path_to_reference, 'w' ) )
    print "\nReference content saved in " + config.path_to_reference
if config.save_pandas_frame:
    dataframe.to_csv( config.path_to_pandas_frame )
    print "\nPandas frame saved in " + config.path_to_pandas_frame
if config.save_cm:
    assert( len(config.days)==1 )
    assert( len(config.epoch) == 1 )
    cmo = tools.ConfusionMatrixObject( cm )
    np.save( open( config.path_to_cm, 'w' ), cm )
    np.save( open( config.path_to_cmo, 'w' ), cmo )
    np.save( open( config.path_to_tp, 'w' ), (cc, ic) )
    print "\nConfusion matrix for single epoch saved in " + config.path_to_cm
    print "\nTrue positives for single epoch saved in " + config.path_to_tp
    print "\nConfusion matrix object for single epoch saved in " + config.path_to_cmo

if config.export_yaml_variables:

    with open( config.yaml_filepath, "w" ) as f_yaml_var:  

        yaml_variables = {\
        "reference_bursts" : {
               "description" : "set of population bursts as detected " +\
                               "offline, either classified as event " +\
                               "with replay content or without",
               "all" :
                   { "count" : n_events,
                     "duration" : {"median" : to_yaml( np.median( candidate_events.duration*1e3 ), u='ms' ),
                                   "iqr" : to_yaml( iqr( candidate_events.duration*1e3 ) ) },
                     "replay" : {"count" : int(n_bursts_with_replay),
                                 "percent" : to_yaml( n_bursts_with_replay/n_events*100, u='%'),
                                 "joint" :
                                         {"count" : int(pmj.is_joint.sum()),
                                          "percent" : to_yaml( pmj.is_joint.sum()/n_bursts_with_replay*100, u='%') }
                                 }
                    },
               "rest" :
                   { "count" : int(in_sleep.sum()),
                     "duration" : {"median" : to_yaml( np.median( candidate_events.duration[in_sleep] )*1e3, u='ms' ),
                                   "iqr" : to_yaml( iqr( candidate_events.duration[in_sleep]*1e3 ) ) },
                     "replay" : {"count" : int(has_replay_sleep.sum()),
                                 "percent" : to_yaml( has_replay_sleep.sum()/in_sleep.sum()*100, u='%'),
                                 "joint" :
                                         {"count" : int(np.sum( pmj.is_joint[in_sleep] )),
                                          "percent" : to_yaml( np.sum( pmj.is_joint[in_sleep] )/has_replay_sleep.sum(), u='%') }
                                 }
                    },
               "run2" :
                   { "count" : int(in_run.sum()),
                     "duration" : {"median" : to_yaml( np.median( candidate_events.duration[in_run] )*1e3, u='ms' ),
                                   "iqr" : to_yaml( iqr( candidate_events.duration[in_run]*1e3 ) ) },
                     "replay" : {"count" : int(has_replay_run.sum()),
                                 "percent" : to_yaml( has_replay_run.sum()/in_run.sum()*100, u='%'),
                                 "joint" :
                                         {"count" : int( np.sum( pmj.is_joint[in_run] )),
                                          "percent" : to_yaml( np.sum( pmj.is_joint[in_run] )/has_replay_run.sum()*100, u='%') }
                                 }
                   }
            },
        "latency" : {
               "added" : {
                   "description" : "measurement of the time between acquisition" +\
                           "of the data (10 ms bin) and the feedback trigger that " +\
                           "signals replay content determination.",
                   "median" : to_yaml( np.median( added_latencies ), u='ms' ),
                   "percentile95" : to_yaml( np.percentile( added_latencies, 95 ), u='ms' ) },
               "detection" : {
                   "all" : {
                       "all_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies*1e3 ) )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies )*100 )} },
                       "long_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[long_bursts_mask_all] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[long_bursts_mask_all] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[long_bursts_mask_all]*100 ), u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[long_bursts_mask_all] )*100 )} },
                       "replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[has_replay] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[has_replay] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[has_replay] )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[has_replay] )*100 )} },
                       "long_replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[long_replay_bursts_mask_all] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[long_replay_bursts_mask_all] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[long_replay_bursts_mask_all] )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[long_replay_bursts_mask_all] )*100 )} } },
                   "rest" : {
                       "all_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies_sleep )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies_sleep*1e3 ) )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies_sleep )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies_sleep )*100 )} },
                       "long_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies_sleep[long_bursts_mask_sleep] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies_sleep[long_bursts_mask_sleep] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies_sleep[long_bursts_mask_sleep]*100 ), u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies_sleep[long_bursts_mask_sleep] )*100 )} },
                       "replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[has_replay_sleep] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[has_replay_sleep] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[has_replay_sleep] )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[has_replay_sleep] )*100 )} },
                       "long_replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[long_replay_bursts_mask_all*in_sleep] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[long_replay_bursts_mask_all*in_sleep] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[long_replay_bursts_mask_all*in_sleep]  )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[long_replay_bursts_mask_all*in_sleep] )*100 )} } },
                   "run2" : {
                       "all_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies_postrun )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies_postrun*1e3 ) )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies_postrun )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies_postrun )*100 )} },
                       "long_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies_postrun[long_bursts_mask_postrun] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies_postrun[long_bursts_mask_postrun] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies_postrun[long_bursts_mask_postrun]*100 ), u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies_postrun[long_bursts_mask_postrun] )*100 )} },
                       "replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[has_replay_run] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[has_replay_run] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[has_replay_run] )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[has_replay_run] )*100 )} },
                       "long_replay_bursts" : {
                           "absolute" : {"median" : to_yaml( np.nanmedian( absolute_latencies[long_replay_bursts_mask_all*in_run] )*1e3, u='ms'),
                                         "ci90" : to_yaml( ci90( absolute_latencies[long_replay_bursts_mask_all*in_run] )*1e3 )},
                           "relative" : {"median" : to_yaml( np.nanmedian( relative_latencies[long_replay_bursts_mask_all*in_run]  )*100, u='%'),
                                         "ci90" : to_yaml( ci90( relative_latencies[long_replay_bursts_mask_all*in_sleep] )*100 )} } }
                       }
                }
        }
                         
        yaml.dump( yaml_variables, f_yaml_var, default_flow_style=False )
    print "YAML variables file created in: " + config.yaml_filepath


#### PLOTTING
if config.plot:
    seaborn.reset_orig()
    not_detected = np.logical_not( detected )
    n_detected = np.sum( detected )
    n_undetected = np.sum( not_detected )
    
    plot_score_cumdistr( event_lfs, detected, "line fit score [a.u.]",\
        xticks=[0, 0.1, 0.2, 0.3, 0.4], nbins=config.nbins, compute_random=True,\
        n_shuffles=1000, fign=4 )
    
    plot_score_cumdistr( event_bZ, detected, "bias score [a.u.]",\
        xticks=np.arange(-2, 9, 2, dtype='int'), nbins=config.nbins, fign=5,\
        compute_random=True, n_shuffles=1000 )
    
    plot_score_cumdistr( event_rZ, detected, "sequence score [a.u.]",\
        xticks=np.arange(-2, 9, 2, dtype='int'), nbins=config.nbins, fign=6,\
        compute_random=True, n_shuffles=1000 )
    
    plot_score_cumdistr( event_bias, detected, "observed bias [a.u.]",\
        xticks=[0, 0.2, 0.4, .6, .8], nbins=config.nbins, fign=7,\
        compute_random=True, n_shuffles=1000 )
    
    plot_score_cumdistr( event_r, detected, "Pearson correlation [a.u.]",\
        xticks=[0, 0.2, 0.4, .6, .8], nbins=config.nbins, fign=8,\
        compute_random=True, n_shuffles=1000 )
        
    plot_absolute_latency_cumdistr( [absolute_latencies_sleep*1e3, absolute_latencies_postrun*1e3],\
       config.epoch, nbins=config.nbins, fign=9 )
    plt.title( "all bursts" )
    plot_relative_latency_cumdistr( [100*relative_latencies_sleep, 100*relative_latencies_postrun],\
       config.epoch, nbins=config.nbins, fign=10 )
    plt.title( "all bursts" )
    
    plot_absolute_latency_cumdistr( [absolute_latencies_sleep[long_bursts_mask_sleep]*1e3, absolute_latencies_postrun[long_bursts_mask_postrun]*1e3],\
       config.epoch, nbins=config.nbins, fign=11 )
    plt.title( "bursts longer than 100 ms" )
    plot_relative_latency_cumdistr( [100*relative_latencies_sleep[long_bursts_mask_sleep], 100*relative_latencies_postrun[long_bursts_mask_postrun]],\
       config.epoch, nbins=config.nbins, fign=12 )
    plt.title( "bursts longer than 100 ms" )
    
    plt.show()
