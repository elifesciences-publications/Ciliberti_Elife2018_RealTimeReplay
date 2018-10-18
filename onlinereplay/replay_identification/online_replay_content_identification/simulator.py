#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:41:42 2017

@author: davide
"""
from __future__ import division
import numpy as np
from os.path import dirname, join
from joblib import Parallel, delayed

from falcon.io.deserialization import load_likelihood_data

from multitrack_decoding.support_tools import compute_mua

import support_replay_analysis.tools as tools


HISTORY = 3
BIN_SIZE = 0.01
    

class OnlineReplaySimulator():
    
    def __init__( self, path_to_likelihood_synch=None, path_to_prerun_mua=None,\
                 prerun_spikes=None, discard=None, path_to_grid=None,\
                 h=HISTORY, bin_size=BIN_SIZE ):
        '''
        Build simulator of online replay classification algorithm
        '''
        print "Initializing simulator ..."
        
        if path_to_likelihood_synch is not None:
            likelihood_data, grid_size, nbins, data, header = load_likelihood_data(\
                                           path_to_likelihood_synch )
        
        if path_to_prerun_mua is not None:
            self.mua_encoding = np.load( path_to_prerun_mua )
            self.spikes_encoding = None
        elif prerun_spikes is not None:
            self.spikes_encoding = prerun_spikes
            self.mua_encoding = compute_mua( prerun_spikes, discard=discard )[0]
        else:
            raise ValueError("Prerun MUA is undefined")
        
        if path_to_grid is not None:
            path_to_content_nodes = join( dirname( path_to_grid ), "content_nodes.npy" )
            self.grid = np.load( path_to_grid )
            self.content_nodes = np.load( path_to_content_nodes )
            self.n_contents = len(self.content_nodes)
            self.grid_size = len(self.grid)
        
        if path_to_likelihood_synch is not None:
            self.nspikes_per_bin = np.hstack( ( self._attachment, likelihood_data["n_spikes"] ) )
            log_likelihood = np.copy( likelihood_data['log_likelihood'] )
            maxs = np.reshape( np.max( log_likelihood, axis=1 ), (nbins, 1) )
            log_likelihood -= maxs
            likelihood = np.exp( log_likelihood )
            self.posteriors = likelihood / np.reshape( np.sum( likelihood, axis=1 ),\
                                                  (nbins,1) )
            self.posteriors_t = data['hardware_ts']/1e6 + bin_size
            self.nbins = nbins
        
        self.mua_avg = None
        self.online_contents = None
        self.peakiness_avg = None
        self.content_consistency = None
        
        self.history = h
        self._attachment = np.zeros( h-1 )
        self._attachment_bool = np.zeros( h-1, dtype='bool' )
        
        self.bin_size = bin_size
    
        self.online_detection_to_candidate = None
        self.online_detection_per_candidate = None
        
        self._variables_computed = False
        
        
    @classmethod
    def from_offline_posteriors( cls, path_to_posteriors, path_to_posteriors_t,\
            path_to_nspikes, path_to_prerun_mua, path_to_grid, h=HISTORY,\
            bin_size=BIN_SIZE ):
        '''
        '''
        cls_temp = cls( path_to_prerun_mua=path_to_prerun_mua, path_to_grid=path_to_grid,\
                       h=h, bin_size=bin_size )
        posteriors = np.load( path_to_posteriors )[0]
        posteriors_t = np.load( path_to_posteriors_t ) + bin_size # use ending times for latency measures
        assert( len(posteriors) == len(posteriors_t) )
        cls_temp.posteriors = posteriors
        cls_temp.posteriors_t = posteriors_t
        cls_temp.nbins = len(posteriors)
        cls_temp.nspikes_per_bin = np.load( path_to_nspikes )
        
        return cls_temp
    
    
    @classmethod
    def from_temp_posteriors( cls, posteriors, posteriors_t, nspikes,\
         path_to_grid, prerun_spikes, discard, h=HISTORY, bin_size=BIN_SIZE ):
        '''
        '''
        cls_temp = cls( prerun_spikes=prerun_spikes,\
                       discard=discard, path_to_grid=path_to_grid, h=h, bin_size=bin_size )
        cls_temp.posteriors = posteriors
        cls_temp.posteriors_t = posteriors_t + bin_size # use ending times for latency measures
        cls_temp.nbins = len(posteriors)
        cls_temp.nspikes_per_bin = nspikes
        
        return cls_temp
    
    
    def compute_variables( self, hiw ):
        '''
        Compute variable of online replay content classification algorithm
        '''
        print "computing MUA avg..."
        self.mua_avg = np.array( [np.mean( self.nspikes_per_bin[i:i+self.history] )\
                                  for i in range(self.nbins)] ) / self.bin_size
                
        print "computing peakiness/sharpness..."
        xpeak = np.uint( np.hstack( (self._attachment, np.argmax( self.posteriors, axis=1 )) ) ) # in grid indices
        win_half_low = [ min( hiw, g ) for g in range(self.grid_size) ] # in grid indices
        win_half_high = [ min( hiw, len(self.grid) - g - 1 )\
            for g in range(self.grid_size) ] # in grid indices
        self.peakiness = np.zeros( self.nbins )
        for b in range(self.nbins):
            x = xpeak[b+self.history-1]
            wh1, wh2 = (win_half_low[x], win_half_high[x])
            low = np.uint( x - wh1 )
            high = np.uint( x + wh2 + 1 )
            self.peakiness[b] = np.sum( self.posteriors[b, low:high] )
        assert( np.isclose( np.sum( self.posteriors ), self.nbins ) )
        peakiness_avg = np.array( [np.mean( self.peakiness[i:i+self.history] )\
                                   for i in range(self.nbins)] )
        
        self.peakiness_avg = np.hstack( (self._attachment, peakiness_avg[:-(self.history-1)] ) )
    
        print "Computing content consistency"
        self.online_contents = -1 * np.ones_like( xpeak, dtype='int16' )
        for content_idx, nodes in enumerate(self.content_nodes):
            m = np.logical_and( xpeak >= nodes[0], xpeak < nodes[1] )
            self.online_contents[m] = content_idx
        self.content_consistency = np.zeros( self.nbins, dtype='bool' )
        for i in range(self.nbins):
            self.content_consistency[i], _ = tools.consistent_content(\
                    self.online_contents[i:i+self.history], self.n_contents )
        
        self._variables_computed = True
        
        
    def compute_performance( self, performance_meter_joint, thresholds, lockout=0.1 ):
        '''
        thresholds = [ MUA_THR, PEAKINESS_THR, PEAKINESS_HISTORY_AVG_THR ]
        
        apply_lockout to filter out close-in-time non-burst detections like in experimental conditions
        
        For detections during burst (candidate) only the first one is considered
        independently of the lockout
        '''
        mua_thr = thresholds[0]
        peakiness_thr = thresholds[1]
        peakiness_hist_avg_thr = thresholds[2]
        
        mua_thr_replay = self.mua_encoding.mean() + mua_thr * self.mua_encoding.std()
    
        high_mua = self.mua_avg > mua_thr_replay
        high_pkns = self.peakiness > peakiness_thr
        high_pkns_havg = self.peakiness_avg > peakiness_hist_avg_thr
        
        detections = high_mua * high_pkns * high_pkns_havg * self.content_consistency
        online_detections_t = self.posteriors_t[detections]
        
        online_detections = np.full_like( detections, -1, dtype='int' )
        online_detections[detections] = self.online_contents[((self.history)-1):][detections]
        
        
        ret_contains = performance_meter_joint.candidate_events.contains(\
                                                         online_detections_t )
        
        nonburst_detections_t = online_detections_t[np.logical_not( ret_contains[0] )]
        lo_nonburst_detections_t = tools.apply_lockout_time_series(\
                                       nonburst_detections_t, lockout )
        
        n_outside = len( lo_nonburst_detections_t )
        online_detection_to_candidate = tools.seg_indices( ret_contains[0], ret_contains[1] )
        
        self.online_detection_to_candidate = online_detection_to_candidate
        
        absolute_latencies = np.full( len(performance_meter_joint.candidate_events), np.NaN )
        n_multiple_detections = np.full_like( absolute_latencies, 0 )
        
        online_detections_per_cand = np.full(\
             len(performance_meter_joint.candidate_events), -1, dtype='int' )
        for i, _ in enumerate(performance_meter_joint.candidate_events):
            if i in online_detection_to_candidate:
                indices = np.nonzero( np.array( online_detection_to_candidate ) == i )[0]
                absolute_latencies[i] = online_detections_t[indices[0]] +\
                    - performance_meter_joint.candidate_events[i].start
                n_multiple_detections[i] = len(indices) - 1
                online_detections_per_cand[i] = self.online_contents[((self.history)-1):][detections][indices[0]]
        
        self.online_detection_per_candidate = online_detections_per_cand
                
        relative_latencies = absolute_latencies / performance_meter_joint.candidate_events.duration
    
        tp, fp, fn, tn, cc, ic = performance_meter_joint.compute_performance(\
                                 online_detections_per_cand, relative_latencies )
        
        cm =  np.array( [[tp, fp.sum()], [fn.sum(), tn] ] ) 
        sensitivity, specificity, precision, _, _, _ = tools.compute_stats( cm )
        if tp != 0:
            inter_content_accuracy = cc.sum()/(cc.sum()+ic.sum())    
        else:
            inter_content_accuracy = np.NaN
            
        cmo = tools.ConfusionMatrixObject( cm )
        
        performances = {}
        performances["confusion matrix"] = cm
        performances["median relative latency"] = np.nanmedian( relative_latencies )
        performances["median absolute latency"] = np.nanmedian( absolute_latencies )
        performances["sensitivity"] = sensitivity
        performances["specificity"] = specificity
        performances["precision"] = precision
        performances["Matthews correlation coefficient"] = cmo.mcc
        performances["inter_content_accuracy"] = inter_content_accuracy
        performances["out_of_candidate_rate"] =\
            n_outside/(self.posteriors_t[-1]-self.posteriors_t[0])

        return performances, mua_thr_replay
        
    
    def epoch( self, bin_size=0.01 ):
        
        return len(self.mua_avg) * bin_size
    
    
    def compute_performance_multiple( self, thresholds_matrix, history=3 ):
        '''
        TO BE TESTED
        '''
        
        self.compute_peakiness_avg( history )
        performances, n_outside_multiple = Parallel(n_jobs=-1)( delayed(\
           self.compute_performance) (thr for thr in thresholds_matrix), use_cache=False )
        
        return performances, n_outside_multiple
    
    