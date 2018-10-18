# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:36:42 2016

@author: davide
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
        

class PerformanceMeterJoint():
    
    def __init__( self, n_contents, candidate_events, arm_max_bias_matrix, bZ_matrix,\
                 lfs_matrix, rZ_matrix, min_bias_score=None, min_lf_score=None,\
                 min_duration_joint=None, min_sequence_score=None ):
        
        self.n_contents = n_contents
        self.n_events = np.shape(arm_max_bias_matrix)[0]
        self.candidate_events = candidate_events
        assert( len(candidate_events) == self.n_events )
        assert( np.shape(arm_max_bias_matrix) ==\
               np.shape(bZ_matrix) == np.shape(lfs_matrix) == np.shape(rZ_matrix) )
        self.arm_max_bias = arm_max_bias_matrix[:, 0]
        self.arm_max_bias_1 = arm_max_bias_matrix[:, 1]
        self.arm_max_bias_2 = arm_max_bias_matrix[:, 2]
        self.bZ = bZ_matrix[:, 0]
        self.bZ_1 = bZ_matrix[:, 1]
        self.bZ_2 = bZ_matrix[:, 2]
        self.lfs = lfs_matrix[:, 0]
        self.lfs_1 = lfs_matrix[:, 1]
        self.lfs_2 = lfs_matrix[:, 2]
        self.rZ = rZ_matrix[:, 0]
        self.rZ_1 = rZ_matrix[:, 1]
        self.rZ_2 = rZ_matrix[:, 2]
        
        self.MIN_BIAS_SCORE = min_bias_score
        self.MIN_LF_SCORE = min_lf_score
        self.MIN_DURATION_JOINT = min_duration_joint
        self.MIN_SEQUENCE_SCORE = min_sequence_score
        
        self.is_single = np.empty( self.n_events, dtype='bool' )
        self.is_joint = np.empty( self.n_events, dtype='bool' )
        self.n_events_with_content = 0
        
        self.tp_semi = 0
        self.fp_semi = 0
        self.fn_semi = 0
        self.tn_semi = 0
        self.two_semi = 0
        
    
    def set_reference_thresholds( self, min_bias_score=3., min_lf_score=0.1,\
             min_duration_joint=0.1, min_sequence_score=-33. ):
        '''
        '''
        self.MIN_BIAS_SCORE = min_bias_score
        self.MIN_LF_SCORE = min_lf_score
        self.MIN_DURATION_JOINT = min_duration_joint
        self.MIN_SEQUENCE_SCORE = min_sequence_score
    
        
    def compute_performance( self, online_contents, relative_latencies ):
        '''
        '''
        
        assert( len(online_contents) == len(relative_latencies) == len(self.candidate_events) )
        
        tp = 0
        fp = np.zeros( self.n_contents, dtype='int' )
        fn = np.zeros( self.n_contents, dtype='int' )
        tn = 0
        cc = np.zeros( self.n_contents, dtype='int' ) # correct content classification
        ic = np.zeros( self.n_contents, dtype='int' ) # incorrect content classification
        self.n_events_with_content = 0
        
        event_durations = self.candidate_events.duration
        
        self.tp_semi = 0
        self.fp_semi = 0
        self.fn_semi = 0
        self.tn_semi = 0
        self.two_semi = 0
        
        for i in range(self.n_events):
            
            single_content = self.bZ[i] > self.MIN_BIAS_SCORE and self.lfs[i] > self.MIN_LF_SCORE\
                and (event_durations[i] < self.MIN_DURATION_JOINT or\
                     self.bZ[i] > self.bZ_1[i] and self.bZ[i] > self.bZ_2[i])\
                and self.rZ[i] > self.MIN_SEQUENCE_SCORE
            content_in_first_seg = self.bZ_1[i] > self.MIN_BIAS_SCORE\
                and self.lfs_1[i] > self.MIN_LF_SCORE and self.rZ_1[i] > self.MIN_SEQUENCE_SCORE\
                and event_durations[i] > self.MIN_DURATION_JOINT
            content_in_second_seg = self.bZ_2[i] > self.MIN_BIAS_SCORE\
                and self.lfs_2[i] > self.MIN_LF_SCORE and self.rZ_2[i] > self.MIN_SEQUENCE_SCORE\
                and event_durations[i] > self.MIN_DURATION_JOINT
          
            self.is_single[i] = single_content
            self.is_joint[i] = ~single_content and (content_in_first_seg or content_in_second_seg)
            
            
            if single_content:
                
                self.n_events_with_content += 1
                
                if online_contents[i] == -1:
                    fn[self.arm_max_bias[i]] += 1
                else:
                    tp += 1
                    if online_contents[i] == self.arm_max_bias[i]:
                        cc[online_contents[i]] += 1
                    else:
                        ic[online_contents[i]] += 1    
        
            elif content_in_first_seg or content_in_second_seg:
                
                self.n_events_with_content += 1
                    
                if online_contents[i] == -1:
                    fn[self.arm_max_bias[i]] += 1
                    self.fn_semi += 1
                    
                elif content_in_first_seg and relative_latencies[i]<0.5:
                    # second segment doesn't matter; cases 1) and 6)
                    tp += 1
                    if online_contents[i] == self.arm_max_bias_1[i]:
                        cc[online_contents[i]] += 1
                    else:
                        ic[online_contents[i]] += 1
                    self.tp_semi += 1
                        
                elif content_in_first_seg and content_in_second_seg and relative_latencies[i]>0.5:
                    # case 2)
                    fn[self.arm_max_bias[i]] += 1
                    tp += 1
                    if online_contents[i] == self.arm_max_bias_2[i]:
                        cc[online_contents[i]] += 1
                    else:
                        ic[online_contents[i]] += 1
                    self.fn_semi += 1
                    self.tp_semi += 1
                    self.two_semi += 1
                    
                elif ~content_in_first_seg and content_in_second_seg and relative_latencies[i]<0.5:
                    # case 3)
                    fp[online_contents[i]] += 1
                    self.fp_semi += 1
                    
                elif ~content_in_first_seg and content_in_second_seg and relative_latencies[i]>0.5:
                    # case 4)
                    tn += 1
                    tp += 1
                    if online_contents[i] == self.arm_max_bias_2[i]:
                        cc[online_contents[i]] += 1
                    else:
                        ic[online_contents[i]] += 1
                    self.tp_semi += 1
                    self.tn_semi += 1
                    self.two_semi += 1
                        
                elif content_in_first_seg and ~content_in_second_seg and relative_latencies[i]>0.5:
                    # case 5)
                    fn[self.arm_max_bias[i]] += 1
                    fp[online_contents[i]] += 1
                    self.fn_semi += 1
                    self.fp_semi += 1
                    self.two_semi += 1
                    
            else:
                
                if online_contents[i] == -1:
                    tn += 1
                else:
                    fp[online_contents[i]] += 1
                    
                    
        return tp, fp, fn, tn, cc, ic


def sum_performance_meters( performance_meters ):
    """
    Returns a single performance meter, sum of the listed individual meters.
    Assumes that contents with same indices are related, but this assumption
    is not used for the combined statistics that are printed and plotted
    """ 
    n_contents = performance_meters[0].n_contents

    collective_meter = PerformanceMeter( n_contents )
    
    for pm in performance_meters:
        
        if pm.n_contents != n_contents:
            raise ValueError("Not all meters have the same number of contents")
        
        collective_meter.true_positives += pm.true_positives
        collective_meter._n_true_negatives += pm._n_true_negatives
        collective_meter.false_positives += pm.false_positives
        collective_meter._n_false_negatives += pm._n_false_negatives
        collective_meter.inter_content_errors += pm.inter_content_errors
        
        collective_meter.ref_events_with_content += pm.ref_events_with_content
        collective_meter.ref_n_events_no_content += pm.ref_n_events_no_content
        collective_meter.online_events_with_content += pm.online_events_with_content
        collective_meter.online_n_events_no_content += pm.online_n_events_no_content
    
        collective_meter._absolute_latencies += pm._absolute_latencies
        collective_meter._relative_latencies += pm._relative_latencies
        
        collective_meter.offline_detections += pm.offline_detections
        collective_meter.online_detections += pm.online_detections
        
    return collective_meter
    
    
class PerformanceMeter():
    
    def __init__( self, n_contents ):
        
        self.n_contents = n_contents
        
        self.true_positives = np.zeros( n_contents, dtype='int' )
        self._n_true_negatives = 0
        self.false_positives = np.zeros( n_contents, dtype='int' )
        self._n_false_negatives = 0
        self.inter_content_errors = np.zeros( n_contents, dtype='int' )
        
        self.ref_events_with_content = np.zeros( n_contents, dtype='int' )
        self.ref_n_events_no_content = 0
        self.online_events_with_content = np.zeros( n_contents, dtype='int' )
        self.online_n_events_no_content = 0

        self._absolute_latencies = []
        self._relative_latencies = []
        
        self.offline_detections = []
        self.online_detections = []
        
        
    def update( self, offline, online, latency, duration, only_tp_latencies=False ):
        
        assert( offline >= -1 and offline < self.n_contents  )
        assert( online >= -2 and online < self.n_contents  )
        
        self.offline_detections.append( offline )
        self.online_detections.append( online )
        
        if offline == online != -1:
            content = offline
            self.true_positives[content] += 1
            self.ref_events_with_content[offline] += 1
            self.online_events_with_content[online] += 1
            self._absolute_latencies.append( latency )
            self._relative_latencies.append( latency/duration )
            
        elif offline == online == -1:
            self._n_true_negatives += 1
            self.ref_n_events_no_content += 1
            self.online_n_events_no_content += 1
            
        elif offline == -1 and online in range(self.n_contents):
            self.false_positives[online] += 1
            self.ref_n_events_no_content += 1
            self.online_events_with_content[online] += 1
            if not only_tp_latencies:
                self._absolute_latencies.append( latency )
                self._relative_latencies.append( latency/duration )
            
        elif offline in range(self.n_contents) and online == -1:
            self._n_false_negatives += 1
            self.ref_events_with_content[offline] += 1
            self.online_n_events_no_content += 1
            
        elif offline in range(self.n_contents) and online in range(self.n_contents):
            self.ref_events_with_content[offline] += 1
            self.online_events_with_content[online] += 1
            self.true_positives[online] += 1
            if offline != online:
                self.inter_content_errors[online] += 1
    
    
    def n_true_positives( self ):
        
        return np.sum( self.true_positives )
            
            
    def n_false_positives( self ):
        
        return np.sum( self.false_positives )  
        
        
    def n_false_negatives( self ):
        
        return self._n_false_negatives


    def n_true_negatives( self ):
        
        return self._n_true_negatives


    def sensitivity( self ):
        
        num = self.n_true_positives()
        den = self.n_true_positives() + self.n_false_negatives()
        assert( den>0 )
        assert( den == ( sum(self.ref_events_with_content)))

        return num / den
        
        
    def specificity( self ):
        
        num = self.n_true_negatives()
        den =  self.n_true_negatives() + self.n_false_positives()
        assert( den>0 )
        assert( den == self.ref_n_events_no_content )        
        
        return num / den
        
        
    def inter_content_accuracy( self ):
        """
        given a true positive (replay with content), what's the probability
        that the content is correctly identified
        """
        inter_content_accuracy = ( self.n_true_positives() -\
            np.sum( self.inter_content_errors ) ) / self.n_true_positives()
        assert( inter_content_accuracy >= 0 or np.isnan(inter_content_accuracy) )
                    
        return inter_content_accuracy

        
    def absolute_latencies( self ):
        
        return np.array( self._absolute_latencies )
        
        
    def relative_latencies( self ):
        
        return np.array( self._relative_latencies )
        
        
    def youden_index( self ):
        
        return self.sensitivity() + self.specificity() - 1
        
        
    def print_results( self ):
        
        assert( np.sum( self.ref_events_with_content ) > 0 )
        
        print "\nSensitivity: {0:.2f} %" .format( self.sensitivity() * 100 )
        print "Specificity: {0:.2f} %" .format( self.specificity() * 100 )
        print "Inter-content accuracy: {0:.2f} %" .format(\
            self.inter_content_accuracy() * 100 )
        print "Youden's index: {0:.2f} %" .format( self.youden_index() * 100 )
        print "\nMedian absolute latency: {0:.2f} ms" .format(\
            np.median( self.absolute_latencies() ) )
        print "Median relative latency: {0:.2f} %" .format(\
            np.median( self.relative_latencies() ) * 100 )
        print "\n# reference events with content: {0}" .format( np.sum(\
            self.ref_events_with_content ) )
            
            
    def plot_latency_result( self, f ):
        
        import seaborn
        seaborn.set_context('poster')
        
        f += 1
        fig = plt.figure( num=f, figsize=(9,6) )
        fig.canvas.set_window_title( "latencies" )
        
        fs_labels = 50
        fs_ticks = 40
        fs_text = 50
        lw_median = 8
        
        max_plot_abs_latency = 300
        
        # plot distribution
        ax = plt.subplot(211)
        seaborn.kdeplot( self.absolute_latencies(), shade=True, gridsize=800, bw=3.5,\
            color='gray' )
        plt.xlabel( 'absolute latency [ms]',  fontsize=fs_labels )
        plt.ylabel( 'probability', fontsize=fs_labels, visible=False )
        ylim = plt.ylim()[1]
        plt.vlines(  np.median(self.absolute_latencies()), 0, ylim, color='r', \
            label='median', linewidth=lw_median )
        plt.text( np.median(self.absolute_latencies())-5, ylim,\
            str( np.round( np.median(self.absolute_latencies()), decimals=2 ))  + " ms" ,\
            color='r', fontsize=fs_text, weight='bold')
        xt = np.linspace(0, 300, 5, dtype='int')
        plt.xticks( xt, [str(h) for h in xt], fontsize=fs_ticks )
        plt.yticks( [0, ylim/2, ylim], fontsize=fs_ticks, visible=False )
        plt.xlim( 0, 300 )
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params( which=u'both',top='off', right='off', left='off', bottom='on', length=0 )
        
        ax = plt.subplot(212)
        seaborn.kdeplot( self.relative_latencies()*100, shade=True, gridsize=800, bw=2.5,\
            color='gray')
        plt.xlabel( 'relative latency [%]',  fontsize=fs_labels )
        plt.ylabel( 'probability', fontsize=fs_labels, visible=False )
        plt.xlim( 0, 100 )
        ylim = plt.ylim()[1]
        plt.vlines(  np.median(self.relative_latencies())*100, 0, plt.ylim()[1], color='r',\
            label='median', linewidth=lw_median )
        median_value = np.round( np.median( self.relative_latencies() )*100,\
            decimals=2 )
        plt.text( np.median(self.relative_latencies())*100-2.5, ylim, str(median_value) +" %",\
            color='r', fontsize=fs_text, weight='bold' )
        xt = np.linspace(0, 100, 5, dtype='int')
        plt.xticks( xt, [str(h) for h in xt], fontsize=fs_ticks )
        plt.yticks( [0, ylim/2, ylim], fontsize=fs_ticks, visible=False )
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params( axis='y', which='both',length=0 )
        ax.tick_params( axis='x', which='both', top='off',length=0 )
        
        # plot histogram
        f += 1
        fig = plt.figure( num=f, figsize=(9,6) )
        fig.canvas.set_window_title( "latencies (histogram)" )
        plt.subplot(211)
        plt.hist( self.absolute_latencies(), bins=20, label='absolute latencies' )
        plt.xlabel( 'latency [ms]',  fontsize=20 )
        plt.ylabel( 'counts', fontsize=20, visible=True )
        ylim = plt.ylim()[1]
        plt.vlines(  np.median( self.absolute_latencies() ), 0, ylim*.9, color='r', \
            label='median latency value' )
        plt.text( np.median(self.absolute_latencies() - 2.5 ), ylim*.9,\
            str(np.round( np.median( self.absolute_latencies() ), decimals=2 ) ),\
            color='r', fontsize=20, weight='bold' )
        xt = np.arange( 0, max_plot_abs_latency + 20, 20 )
        plt.xticks( xt, [str(h) for h in xt], fontsize=20 )
        plt.yticks( [0, ylim/2, ylim], fontsize=20 )
        plt.xlim( 0, max_plot_abs_latency )
        plt.legend( loc="best", fontsize=20 )
        
        plt.subplot(212)
        plt.hist( self.relative_latencies()*100, bins=20, label='relative latencies',\
            color='green')
        plt.xlabel( '% of total duration',  fontsize=20 )
        plt.ylabel( 'counts', fontsize=20, visible=True )
        plt.xlim( 0, 100 )
        ylim = plt.ylim()[1]
        plt.vlines(  np.median(self.relative_latencies())*100, 0, plt.ylim()[1],\
            color='r', label='median latency value' )
        median_value = np.round( np.median( self.relative_latencies() )*100,\
            decimals=2 )
        plt.text( np.median(self.relative_latencies())*100-2.5, ylim,\
            str(np.round( median_value, decimals=2 )), color='r', fontsize=20, weight='bold' )
        xt = np.arange(0, 110, 10 )
        plt.xticks( xt, [str(h) for h in xt], fontsize=20 )
        plt.yticks( [0, ylim/2, ylim], fontsize=20 )
        plt.legend( loc="best", fontsize=20 )
