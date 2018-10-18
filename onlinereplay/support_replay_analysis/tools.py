# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:55:12 2016

@author: davide
"""
from __future__ import division
import numpy as np
import scipy as sp
import math
import natsort
import cPickle as pickle
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

from fklab.utilities import yaml
from fklab.geometry import shapes # needed for opening enviornment yaml files
from fklab.utilities import randomize

from os.path import join, isfile

from replay_identification.detection_performance import PerformanceMeter

def seg_indices( isinseg, ninseg ):
    """
    
    """
    j = 0
    indices = []
    ninseg_cpy = ninseg.copy()
    for i in range(len(isinseg)):
        if isinseg[i]:
            while ninseg_cpy[j] == 0:
                j += 1
            ninseg_cpy[j] -= 1
            indices.append( j )
        else:
            indices.append( np.NaN )
    assert( len(indices) == len(isinseg) )
    
    return indices


class Bounds():
    
    def __init__( self, lower, upper ):
        self.lower = lower
        self.upper = upper
        
    def is_inside( self, value ):
        return (np.logical_and( value > self.lower, value < self.upper ) )
    
    
class ContentDetector():
    
    def __init__( self, content_nodes, grid ):
        self.grid = grid
        self.n_contents = len(content_nodes)
        self.bounds = [ Bounds( grid[cn[0]], grid[cn[1]] ) for cn in content_nodes ]
    
    def detect_content( self, line ):
        """
        Given a set of aligned position values, returns the content id of the line
        
        Returns
        --------
        content : int or NaN, if NaN no content was detect, otherwise ID (from 0)
        of the corresponding content of the line
        """
        
        if len(line) < 3:
            raise ValueError("Line does not have enough values")
            
        for c, bds in enumerate(self.bounds):
            if np.bool( np.median( bds.is_inside( line ) ) ):
                return c
        
        return np.NaN
        

def nearest_diff( array, value ):
    '''
    Returns the absolute difference between value and the nearest element of the array
    '''    
    return np.abs(array-value).min()
    

def compute_pixel_to_cm( environment_filepath, arm_length_cm ):
    """
    
    """
    with open( environment_filepath, 'r') as f_yaml:
        info = yaml.load( f_yaml )
    
    assert( len(info.keys()) == 1 )
    maze_s = info.keys()[0]
    arm_length_pixel = np.array( [\
        sp.spatial.distance.euclidean( info[maze_s]["shapes"][arm_s]['shape'].vertices[0],\
        info['RAM3']["shapes"][arm_s]['shape'].vertices[1])\
        for arm_s in info[maze_s]["shapes"].keys()] )
            
    return np.mean( arm_length_pixel ) / arm_length_cm


def adjust_posteriors( posteriors, content_nodes ):
    """
    Remove meaningless linear positions added for mere linearization purposes
    """    
#    assert( posteriors.shape[1] > posteriors.shape[0] )
    sel = np.hstack( [ np.arange(cn[0], cn[1]) for cn in content_nodes ] )
    new_posterios = posteriors[:, sel]
    renormalized_new_posteriors =\
        new_posterios / np.sum(new_posterios, axis=1)[:, np.newaxis]
        
    return renormalized_new_posteriors
    
    
def fix_content_nodes( content_nodes, equalize=False, arm_length_grid_units=50 ):
    """
    """
    n_contents = len(content_nodes)
    fixed_content_nodes = np.empty_like( content_nodes, dtype='int' )
    nodes = content_nodes
    
    if equalize:
        equalized_content_nodes = np.zeros_like( content_nodes, dtype='int' )
        equalized_content_nodes[0, 1] = arm_length_grid_units
        for i in range(1, n_contents-1):
            to_be_removed = np.int( ( np.diff( content_nodes[i] ) - arm_length_grid_units ) / 2 )
            equalized_content_nodes[i, 0] = content_nodes[i, 0] + to_be_removed 
            equalized_content_nodes[i, 1] = content_nodes[i, 1] - to_be_removed 
        
        equalized_content_nodes[-1, 0] = content_nodes[-1, 0] +\
            ( np.diff( content_nodes[-1] ) - arm_length_grid_units )
        equalized_content_nodes[-1, 1] = content_nodes[-1, -1]
    
        nodes = equalized_content_nodes
    else:
        equalized_content_nodes = None
    
    new_limits = np.cumsum( np.diff( nodes) )
    fixed_content_nodes[0] = nodes[0] 
    for i in range(1, n_contents):
        fixed_content_nodes[i] = np.array([ new_limits[i-1], new_limits[i] ])    
    
    return fixed_content_nodes, equalized_content_nodes
    

def compute_duration( posteriors, bin_size_replay ):
    """
    """
    duration = np.shape(posteriors)[0] * bin_size_replay * 1e3
    
    return np.round( duration, decimals=1 )
    

def fit_replay_line( posteriors, time_vector=None, pos_vector=None ):
    """
    Fit line in posteriors, performs smoothing before fitting
    """
    from fklab import radon
    radon_transform = radon.RadonTransform( interpolation='Linear',\
        constraint='X', oversampling=4, pad='Median' ) 

    smooth_posteriors = radon.local_integrate( posteriors, n=3, axis=0 ) 
    
    return radon_transform.fit_line( smooth_posteriors, x=time_vector,\
        y=pos_vector )
    
    
def find_line( posteriors, bin_size_replay, grid, content_nodes, content_sel=None ):
    """
    Find best fitting line among the subgroups of posteriors that correspond
    to each different arm
    
    returns scores as 1d array with a score per content if content_sel is None
    """
    
    duration = compute_duration( posteriors, bin_size_replay )
    time_bins = np.arange( 0, duration+bin_size_replay*1e3, bin_size_replay*1e3 )
    time_vector = (time_bins[:-1] + time_bins[1:])/2.
    
    n_contents = len(content_nodes >1)
    assert( n_contents > 0 )
    pos_vectors = [ grid[0:np.diff(content_nodes)[c][0]] for c in range(n_contents) ]
    slopes = np.full( n_contents, np.NaN )
    intercepts = np.full( n_contents, np.NaN )
    scores = np.full( n_contents, np.NaN )
    
    for c in range(n_contents):
        
        if c == content_sel or content_sel is None:
            
            posteriors_single_content =\
                posteriors[:, content_nodes[c, 0] : content_nodes[c, 1]]
            
            (slopes[c], intercepts[c]), scores[c], _, _ = fit_replay_line(\
                posteriors_single_content, time_vector=time_vector,\
                pos_vector=pos_vectors[c].ravel() )
        
    if content_sel == None:
        content = int(np.argmax( scores ))
    else:
        content = int(content_sel)
        
    offset = (grid[content_nodes[content, 0]] - grid[0])[0]
    
    line = time_bins * slopes[content] + intercepts[content] + offset
        
    assert (np.all(np.logical_not( np.isnan(line) )))        
        
    return ( line, scores, content, time_vector, time_bins, pos_vectors[content].ravel(),\
        posteriors[:, content_nodes[content, 0] : content_nodes[content, 1]] )
    

def compute_score_random( posteriors, time_vector, pos_vector ):
    """
    Compute the score of the best fitted line over a randomized dataset of
    posteriors
    
    """
  
    random_posteriors, _ = randomize( posteriors, axis=1, method='roll' )
        
    _, random_scores, _, _ = fit_replay_line( random_posteriors, time_vector,\
        pos_vector )
             
    return random_scores
    
    
def compute_replay_significance( posteriors, score, time_vector, pos_vector,\
n_shuffles, num_cores=7 ):
    """
    Perform N randomization and compute score of best fitting line, then
    compute Monte-Carlo p-value using multi-processing
    """
    
    random_scores = Parallel( n_jobs=num_cores )( delayed( compute_score_random )\
        ( posteriors, time_vector, pos_vector ) for k in range(n_shuffles) )

    return ( np.sum( random_scores >= score ) + 1. ) / ( n_shuffles+1 )


def compute_linefit_Zscore_againstshuffle( posteriors, time_vector, pos_vector,\
content_nodes, content, score=None, n_shuffles=500, num_cores=7 ):
    """
    compute score of best fitting line, then
    compute Z-score of the obtained value against a Monte-Carlo shuffle distribution,
    content must be provided to indicate which part of the posterior to use
    for the randomization
    """

    n_contents = len(content_nodes)
    assert( content < n_contents )
    assert( content >= 0 )
    scores = np.zeros( n_contents )
    
    for c in range(n_contents):
        
        posteriors_single_content =\
            posteriors[:, content_nodes[c, 0] : content_nodes[c, 1]]
        select_content = np.zeros( len(pos_vector), dtype='bool' )
        select_content[content_nodes[c, 0]:content_nodes[c, 1]] = True
        (_, _), scores[c], _, _ = fit_replay_line( posteriors_single_content,\
            time_vector=time_vector, pos_vector=pos_vector[select_content] )
        
    select_content = np.zeros( len(pos_vector), dtype='bool' )
    select_content[content_nodes[content, 0]:content_nodes[content, 1]] = True
    
    random_linefit_scores = Parallel( n_jobs=num_cores )( delayed( compute_score_random )\
         ( posteriors[:, content_nodes[content, 0] : content_nodes[content, 1]],\
         time_vector, pos_vector[select_content] ) for k in range(n_shuffles) )
    
    linefit_score_observed = scores[content]
    if score is not None: assert( linefit_score_observed == score )
    linefit_score_againstnull =\
    ( linefit_score_observed - np.mean( random_linefit_scores ) ) / np.std( random_linefit_scores )
    
    return linefit_score_againstnull, content, linefit_score_observed


def is_consistent_content( contents, ncontents ):
    '''
        contents = array of content indices [-1, ncontents-1]
        ncontents = number of potential contents
        
    '''
    assert( min(contents) >= -1 )
    assert( max(contents) <= (ncontents-1) )
    for c in range(ncontents):
        if all(contents == c):
            return (True, c)
    return (False, -1)


def apply_lockout( detections, bin_size, lockout_ms, lockout_value=-1 ):
    '''
    simulate lockout period
    detections: array of bool or int
    '''
    nbins_lockout = np.int( lockout_ms/1e3/bin_size )
    n_bins = len(detections)
    i = 0
    result = np.copy( detections )
    locked_out = False
    if detections.dtype == np.int:
        locked_out = lockout_value
    while i < n_bins-nbins_lockout:
        if detections[i] != -1:
            result[i+1:i+1+nbins_lockout] = locked_out
            i += nbins_lockout
        else:
            i += 1
    return result


def apply_lockout_time_series( times, lockout ):
    '''
    times: list of timestamps in s of online detections
    lockout: lockout time in s
    '''
    if len(times) <= 1:
        return times
    else:
        lo_times = []
        lo_times.append( times[0] )
        last_valid_time = times[0]
        for i,t in enumerate( times ):
            if t - last_valid_time > lockout:
                lo_times.append( t )
                last_valid_time = t
        
        return np.array( lo_times )

            

def compute_performance( ref_detections, candidate_events, online_detections, online_detections_t, n_contents ):
    '''
    Compute sensitivity, specificity, inter-content accuracy and latencies of a
    set of online detections 
    
    online_detection: 1D array with 1st row being an array of int with content
    (negative values = no content)
    online_detections_t: 1D array with the timestamp (in s) of each detection
    
    ref_detections: 1d array of int indicating the reference content of each
    candidate event
    
    Returns
    ----------------
    
    '''
    import support_replay_analysis.tools as tools
    
    assert( len(online_detections) == len(online_detections_t))
    
    performance = PerformanceMeter( n_contents )
    n_outside = np.sum( np.logical_not( candidate_events.contains( online_detections_t )[0] ) )

    n_multiple_detections = np.zeros( len(candidate_events), dtype='int' )
    
    ret_contains = candidate_events.contains( online_detections_t )
    online_detection_to_candidate = tools.seg_indices( ret_contains[0], ret_contains[1] )
    
    no_latency = -1e18
    no_duration = 0
    
    for i, _ in enumerate(candidate_events):
        
        if i in online_detection_to_candidate:

            indices = np.nonzero( np.array( online_detection_to_candidate ) == i )[0]
            latency = online_detections_t[indices[0]] - candidate_events[i].start
            performance.update( ref_detections[i], online_detections[indices[0]],\
                               latency*1e3, candidate_events[i].duration*1e3 )
            n_multiple_detections[i] = len(indices) - 1
        else:
            performance.update( ref_detections[i], -1, no_latency, no_duration )
            
    return performance, n_outside, n_multiple_detections        



def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    return dcor


def compute_bias( posteriors, content_nodes, arm=None ):
    '''
    posteriors: nxm matrix,
        n = # time bins
        m = # position bins
    content_nodes: k x 2 matrix,
        k = # contents
        content_nodes[i,0] = index 1st position bin content i
        content_noeds[i,1] = index last postion bin content i
    '''
    
    n_time_bins = np.shape( posteriors )[0]
    n_contents = len(content_nodes)
    
    b = [np.sum( posteriors[:, content_nodes[i][0]:content_nodes[i][1]] )/n_time_bins for i in range(n_contents)]
    if arm == None:
        bias = np.max(b)
    else:
        bias = b[arm]
    f = 1/n_contents
    bias_norm = (bias-f)/(1-f)
    if arm == None:
        ret_arm = int( np.argmax( b ) )
    else:
        ret_arm = arm
    
    return b, bias_norm, ret_arm


def compute_bias_significance_score( posterior, content_nodes, M=1000 ):
    '''
    bias: n x m array, n = # contents (arms), m = # time bins
    Compute bias significance score as 1/MonteCarlo_P-value 
    '''
    
    bias, bias_observed, arm_max_bias = compute_bias( posterior, content_nodes )
    random_bias = np.full( (M,), np.NaN )
    for k in range(M):
        randomized_posterior, _ = randomize( posterior, axis=1, method='roll' )
        _, random_bias[k], _ = compute_bias( randomized_posterior, content_nodes, arm=arm_max_bias )
    significance_score = 1/(np.sum(random_bias > bias_observed)/M)
    if significance_score == np.inf:
        significance_score = M
    
    return bias, bias_observed, significance_score, arm_max_bias, random_bias


def compute_bias_score( posterior, content_nodes, bias=None, M=5000, arm=None ):
    '''
    bias: n x m array, n = # contents (arms), m = # time bins
    Compute bias score bZ as:
        bZ = ( b(observed) - mean[b(null)] ) / S.D.(b(null))
    '''
    
    if bias is None:
        b, bias_norm, arm_max_bias = compute_bias( posterior, content_nodes, arm=arm )
    b_observed = bias_norm
    random_bias = np.full( (M,), np.NaN )
    for k in range(M):
        randomized_posterior, _ = randomize( posterior, axis=1, method='roll' )
        random_bias[k] = compute_bias( randomized_posterior, content_nodes, arm=arm_max_bias )[1]
    b_score = ( b_observed - np.mean(random_bias) ) / np.std( random_bias )
    p_value = np.sum( random_bias > b_observed )/M
    
    return b_score, p_value, b_observed, arm_max_bias


def lin_regression( posteriors, time_bins, pos_bins, N=3000, pos_mask=None ):
    '''
    Compute linear regression of event after resampling each posterior bin with
    weighted distribution
        
    pos_mask = boolean 1d array to select a subset of position over which
    linear regression with resampling should be computed
    '''
    
    n_time_bins = len(time_bins)
    n_pos_bins = len(pos_bins)
    
    assert( np.shape(posteriors) == (time_bins,pos_bins) )
    
    x = np.zeros( N * n_time_bins )
    y = np.zeros_like( x )
    
    include = np.ones( n_time_bins*N, dtype='bool' )
    
    for i in range(n_time_bins):
        x[i*N:(i+1)*N] = time_bins[i]
        y[i*N:(i+1)*N] = np.random.choice( pos_bins, size=N, p=posteriors[i,:] )
    
    if pos_mask is not None:
        assert( len(pos_mask) == n_pos_bins )
        include = np.in1d( y, pos_bins[pos_mask] )
        
    x = x[include]
    y = y[include]
    
    n_points = np.sum(include)

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress( x, y )
    
    return slope, intercept, r_value, p_value, std_err, n_points


class Correlation():
    
    def __init__( self, posterior, time_vector, position_vector ):
        
        self.posterior = posterior

        self.n_time_bins = np.shape( posterior )[0]
        self.n_pos_bins = np.shape( posterior )[1]
        
        self.time_vector = time_vector
        self.position_vector = position_vector
        
        self.m_t = np.average( self.time_vector, weights=np.sum(posterior, axis=1) )
        self.m_pos = np.average( self.position_vector, weights=np.sum(posterior, axis=0) )
        
        self.norm_factor = np.sum( posterior )
        
        self.x = None
        self.y = None


    def weighted_covariance( self ):
        '''
        Computed weighted Pearson covariance
        '''
        num = 0
        for i in range(self.n_time_bins):
            for j in range(self.n_pos_bins):
                num += self.posterior[i,j] * ( self.time_vector[i] - self.m_t ) * (\
                                     self.position_vector[j] - self.m_pos )
        
        return num/self.norm_factor
    
    
    def weighted_covariance_t( self ):
        
        num = 0
        for i in range(self.n_time_bins):
            for j in range(self.n_pos_bins):
                num += self.posterior[i,j] * ( self.time_vector[i] - self.m_t )**2
        
        return num/self.norm_factor
    
    
    def weighted_covariance_pos( self ):

        num = 0
        for i in range(self.n_time_bins):
            for j in range(self.n_pos_bins):
                num += self.posterior[i,j] * ( self.position_vector[j] - self.m_pos )**2
        
        return num/self.norm_factor
    
    
    def weighted_correlation( self ):
        '''
        Computed weighted Pearson correlation
        '''
        den = np.sqrt( self.weighted_covariance_t() * self.weighted_covariance_pos() )
        
        return self.weighted_covariance() / den 
    
    
    def distance_correlation_bs( self, N=1000 ):
        '''
        Compute distance correlation over entire posterior
        using bootstrapping. N correlation values are returned
        '''
        distribution = np.array([ distcorr( self.time_vector, [ np.random.choice(\
                self.position_vector, p=self.posterior[i, :] )\
                for i in range(self.n_time_bins) ] ) for n in range(N) ])
    
        return np.median(distribution), distribution
        
    
    def correlation_resampling( self, N=1000, r='pearson', pos_mask=None ):
        '''
        Compute correlation after resampling the probability ditribution associated
        with each posterior. Correlation can be computed as:
            pearson, spearman or distance
            
        pos_mask = boolean 1d array to select a subset of position over which
        correlation with resampling should be computed
        '''
        x = np.zeros( N * self.n_time_bins )
        y = np.zeros_like( x )
        
        include = np.ones( self.n_time_bins*N, dtype='bool' )
        
        for i in range(self.n_time_bins):
            x[i*N:(i+1)*N] = self.time_vector[i]
            y[i*N:(i+1)*N] = np.random.choice( self.position_vector, size=N,\
                  p=self.posterior[i,:] )
        
        if pos_mask is not None:
            assert( len(pos_mask) == self.n_pos_bins )
            include = np.in1d( y, self.position_vector[pos_mask] )
            
        x = x[include]
        y = y[include]
            
        self.x = x
        self.y = y

        if r=="pearson":
            return sp.stats.pearsonr( x, y )[0]
        elif r=="spearman":
            return sp.stats.spearmanr( x, y )[0]
        elif r=="distance":
            return distcorr( x, y )
        else:
            raise ValueError("Correlation method not understood")    
            
            
    def sequence_score( self, r_observed=None, r="pearson", pos_mask=None, M=500, N=1000  ):
        '''
        rZ = ( |r(observed)| - |mean[r(null)]| ) / S.D.(|r(null)|)
        Grosmark&Buzsaki(2016)
        
        N = # samples distribution resampling
        M = # shuffles null distribution
        '''
        if r_observed is None:
            r_observed = self.correlation_resampling( N=N, r=r, pos_mask=pos_mask )
        random_correlations = np.full( (M,), np.NaN )
        for k in range(M):
            random_data, _ = randomize( self.posterior, axis=1, method='roll' )
            random_correlations[k] = Correlation(\
                random_data, self.time_vector, self.position_vector ).correlation_resampling(\
                             r=r, pos_mask=pos_mask )
#        for k in range(int(M/2)):
#            random_data, _ = randomize( self.posterior, axis=1, method='shuffle' )
#            random_correlations[int(M/2)+k] = Correlation(\
#                random_data, self.time_vector, self.position_vector ).correlation_resampling(\
#                             r=r, pos_mask=pos_mask )
#        
        return ( np.abs(r_observed) - np.abs( np.mean( random_correlations ) ) ) / np.std(\
                     np.abs(random_correlations) )
    
    
    def bias( self, content_nodes ):
        '''
        Compute bias versus each arm defined by a set of grid nodes
        
        Returns:
            1d array of N bias measures, N=number of contents
        '''
        return compute_bias( self.posterior, content_nodes )
        

    def bias_score( self, content_nodes, M=5000 ):
        '''
        Compute bias score bZ as:
            bZ = ( |b(observed)| - |mean[b(null)]| ) / S.D.(|b(null)|)
        '''
        bias = self.bias( content_nodes )
        arm_max_bias = np.argmax( bias )
        b_observed = bias[arm_max_bias]
        random_bias = np.full( (M,), np.NaN )
        for k in range(M):
            randomized_posterior, _ = randomize( self.posterior, axis=1, method='roll' )
            random_bias[k] = np.max( compute_bias( randomized_posterior, content_nodes ) )
        b_score = ( b_observed - np.mean(random_bias) ) / np.std( random_bias )
        
        return b_score, arm_max_bias
        
    
    def linefit_againstnull_score( self, content_nodes, M=5000, num_cores=7 ):
        '''
        
        '''
        n_contents = len(content_nodes)
        scores = np.zeros( n_contents )
    
        for c in range(n_contents):
            
            posteriors_single_content =\
                self.posterior[:, content_nodes[c, 0] : content_nodes[c, 1]]
            select_content = np.zeros( len(self.position_vector), dtype='bool' )
            select_content[content_nodes[c, 0]:content_nodes[c, 1]] = True
            (_, _), scores[c], _, _ = fit_replay_line(\
                posteriors_single_content, time_vector=self.time_vector,\
                pos_vector=self.position_vector[select_content] )
            
        content = np.argmax( scores )
        select_content = np.zeros( len(self.position_vector), dtype='bool' )
        select_content[content_nodes[content, 0]:content_nodes[content, 1]] = True
        
        random_linefit_scores = Parallel( n_jobs=num_cores )( delayed( compute_score_random )\
             ( self.posterior[:, content_nodes[content, 0] : content_nodes[content, 1]],\
             self.time_vector, self.position_vector[select_content] ) for k in range(M) )
        
        linefit_score_observed = scores[content]
        linefit_score_againstnull =\
        ( linefit_score_observed - np.mean( random_linefit_scores ) ) / np.std( random_linefit_scores )
        
        return linefit_score_againstnull, content, linefit_score_observed
 
    
def boundaries( linear_pos_cm, added_distance_cm, n_tracks ):
    """
    Returns high and low boundaries for rescaling the linear positions
    (to correct for the added distance introduced for decoding purposes)
    """
    
    low_boundaries = np.array( [ b*added_distance_cm for b in range(n_tracks) ] )
    low_boundaries *= .99
    
    high_boundaries = [ np.nanpercentile( linear_pos_cm[linear_pos_cm < lb], 99 )\
        for lb in low_boundaries[1:] ]
    high_boundaries.append( np.nanmax( linear_pos_cm ) )
    high_boundaries = np.array( high_boundaries )
    high_boundaries *= 1.01
    
    return low_boundaries, high_boundaries
    

def remove_added_distance( linear_pos_cm, added_distance_cm, n_tracks,\
low_boundaries, high_boundaries ):
    """
    Return linear positon values without the added distance artifically introduced
    between two tracks for decoding purposes.
    
    Params
    ------
    linear_pos_cm :  1d array of linearized position values
    added_distance_cm : scalar
    n_tracks : integer
    """
    linear_pos_output = linear_pos_cm.copy() 
    
    diff_boundaries = low_boundaries[1:] - high_boundaries[:-1]
    
    cm_diff_boundaries = np.cumsum( diff_boundaries )
    
    # values from 1st track are not modified
    for t in range(1, n_tracks):
        sel = np.logical_and( linear_pos_output >= low_boundaries[t],\
            linear_pos_output <= high_boundaries[t] )
        linear_pos_output[sel] = linear_pos_output[sel] - cm_diff_boundaries[t-1]
            
    return linear_pos_output    


def compute_confusion_matrix( condition, test ):
    '''
    Takes two boolean arrays. Returns a 2x2 matrix\n
    TP FP\n
    FN TN
    \n\ne.g. condition: has_content
        test : detected_online
    '''
    tp = np.sum( np.logical_and( test, condition ) )
    fp = np.sum( np.logical_and( test, ~condition ) )
    tn = np.sum( np.logical_and( ~test, ~condition ) )
    fn = np.sum( np.logical_and( ~test, condition ) )
    
    return np.array( [[tp, fp], [fn, tn] ] )


def compute_relative_risk( confusion_matrix ):
    '''
    '''
    tp, fp, fn, tn = tuple( confusion_matrix.ravel() )
    rr = (tp/(tp+fp)) / (fn/(fn+tn))
    se_lnrr = np.sqrt( 1/tp + 1/fn - 1/(tp+fp) - 1/(fn+tn) )
    rr_95ci_lower = np.exp( np.log(rr) - 1.96 * se_lnrr )
    rr_95ci_upper = np.exp( np.log(rr) + 1.96 * se_lnrr )
    
    return rr, (rr_95ci_lower, rr_95ci_upper)


def compute_dor( confusion_matrix ):
    '''
    '''
    tp, fp, fn, tn = tuple( confusion_matrix.ravel() )
    dor = (tp/fn)/(fp/tn)
    se_lndor = np.sqrt( 1/tp + 1/tn + 1/fp + 1/fn )
    dor_95ci_upper = np.exp( np.log(dor) + 1.96 * se_lndor )
    dor_95ci_lower = np.exp( np.log(dor) - 1.96 * se_lndor )
    
    return dor, (dor_95ci_lower, dor_95ci_upper)


def compute_stats( confusion_matrix ):
    '''
    '''
    tp, fp, fn, tn = tuple( confusion_matrix.ravel() )
    sens = tp/(tp + fn)
    spec = tn/(fp + tn)
    ppv = tp/(tp+fp) # precision / positive predicitive value
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    ll_ratio_pos = sens/(1-spec) 
    ll_ratio_neg = (1-sens)/spec
    
    return sens, spec, ppv, accuracy, ll_ratio_pos, ll_ratio_neg

    
def extract_preprocessed_dataset_fn( folder_name ):
    '''
    '''
    from os.path import join, isfile
    from os import listdir
    
    if folder_name[-1] == '/':
        folder_name = folder_name[:-1]
    
    l = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and len(f)==24 and f[-4:]=='hdf5' ]
    assert(len(l)==1)
    
    return join( folder_name, l[0] )


def shuffle_in_unison(a, b):
    '''
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def create_cm_random( online_contents, pmj, relative_latencies, N ):
    '''
    '''
    
    random_cm = np.zeros( (2, 2, N), dtype=int )
    
    for i in range(N):
        
        online_contents_sh = online_contents.copy()
        relative_latencies_sh = relative_latencies.copy()
        shuffle_in_unison( online_contents_sh, relative_latencies_sh )
        tp, fp, fn, tn, _, _ = pmj.compute_performance(\
            online_contents_sh, relative_latencies_sh )
        random_cm[:,:,i] = np.array( [[tp, fp.sum()], [fn.sum(), tn] ] ) 
        
    return random_cm
    

class ConfusionMatrixObject:
    '''
    '''
    def __init__(self, cm ):
        self.matrix = cm
        self.tp, self.fp, self.fn, self.tn = tuple( cm.ravel() )
        self.sensitivity = np.true_divide( self.tp, (self.tp + self.fn) )
        self.tpr = self.sensitivity
        self.specificity = np.true_divide( self.tn, (self.tn + self.fp) )
        self.fpr = 1 - self.specificity
        self.yi = self.sensitivity + self.specificity - 1
        self.informedness = self.yi
        self.precision = np.true_divide( self.tp, (self.tp + self.fp) )
        self.ppv = self.precision
        self.false_discovery_rate = 1 - self.ppv
        self.npv = np.true_divide( self.tn, (self.tn + self.fn) )
        self.for_ = 1 - self.npv
        self.markedness = self.ppv + self.npv - 1
        self._mcc = np.sqrt( self.informedness * self.markedness )
        a = self.tp+self.fp
        if a==0: a = 1
        b = self.tp+self.fn
        if b==0: b=1
        c = self.tn+self.fp
        if c==0: c=1
        d = self.tn+self.fn
        if d==0: d=1
        self.mcc = (self.tp*self.tn - self.fp*self.fn)/np.sqrt(a*b*c*d)
        if not (np.isclose(np.abs(self.mcc),self._mcc) or (self.mcc==0 and np.isnan(self._mcc)) ):
            print self.mcc
            print'\t'
            print self._mcc
        

def to_yaml( fl, d=1, u=None ):
    '''
    Converts float to string with d decimals and unit u.
    This formatting is needed for marked-down doc that import floats from yaml.
    Recognizes intervals and generates [xx.x, xx.x] string.
    No unit for intervals.
    '''
    if np.shape(fl) == ():
        out_fl = "{0:.{1:d}f}" .format( fl, d )
        if u is not None and u is not '%':
            out_fl += ' ' + u
        if u is '%':
            out_fl += u
        return out_fl
    elif np.shape(fl)[0] == 2:
        a = "{0:.{1:d}f}" .format( fl[0], d )
        b = "{0:.{1:d}f}" .format( fl[1], d )
        return '[' + a + ', ' + b + ']'
    else:
        return None


def iqr( values, nan_aware=True ):
    '''
    '''
    if nan_aware:
        a = np.nanpercentile( values, 25 )
        b = np.nanpercentile( values, 75 )
    else:
        a = np.percentile( values, 25 )
        b = np.percentile( values, 75 )
    return np.array( [a, b] )


def range_( values, nan_aware=True):
    '''
    '''
    if nan_aware:
        a = np.nanmin( values )
        b = np.nanmax( values )
    else:
        a = np.min( values )
        b = np.max( values )
    return np.array( [a, b] )


def ci90( values, nan_aware=True ):
    '''
    '''
    if nan_aware:
        a = np.nanpercentile( values, 5 )
        b = np.nanpercentile( values, 95 )
    else:
        a = np.percentile( values, 5 )
        b = np.percentile( values, 95 )
    return np.array( [a, b] )


def generate_test_spikes( path_to_online_data, path_to_output ):
    '''
    generate test spikes in unraveled format from online spikes assembled from
    a dictionary listing tetrodes ('id', 'value')
    '''
    online_spikes = np.load( path_to_online_data )

    for tt in online_spikes:
        
        filename_out = join( path_to_output, path_to_output )
        np.save( open( join(filename_out, tt['id'] + '.npy'), 'w'),\
                  tt['value'].ravel() )

 
def bhattacharyya(a, b):
    """
    Bhattacharyya distance between distributions (lists of floats).
    """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")

    return sum((math.sqrt(u * w) for u, w in zip(a, b)))


def extract_encoding_tt_ids( partial_path_to_encoding_model, n_tt_max=40 ):
    """
    """
    TT_ids = []
    j = 0
    for i in range(n_tt_max):
        tt_id = "TT" + str( i+1 )
        filename_mixture = join( partial_path_to_encoding_model, tt_id + "_mixture.dat" )
        if isfile( filename_mixture ):
            TT_ids.append( tt_id )
            j += 1
    
    return TT_ids
    
    
def extract_mismatch_tt_ids( small, large ):
    """
    """
    assert( len(large) > len(small) )
    assert( natsort.natsorted(small) == small )
    assert( natsort.natsorted(large) == large )
    mismatches = []
    for i, el in enumerate(large):
        if el not in small:
            mismatches.append( i )
    return mismatches
    

def win_halves( grid_size, hiw ):
    '''
    '''
    win_half_low = [ min( hiw, g ) for g in range(grid_size) ] # in grid indices
    win_half_high = [ min( hiw, grid_size - g - 1 ) for g in range(grid_size) ] # in grid indices
    
    return win_half_low, win_half_high

    
def compute_sharpness( posteriors, grid_size, win_half_low, win_half_high, history=3 ):
    '''
    compute sharpness as in online algorithm, first (history-1) posteriors used
    as initial conditions
    '''
    xpeak = np.uint( np.argmax( posteriors, axis=1 )) # in grid indices
    nbins = len(posteriors)
    peakiness = np.zeros( nbins )
    for b in range(nbins):
        x = xpeak[b]
        wh1, wh2 = (win_half_low[x], win_half_high[x])
        low = np.uint( x - wh1 )
        high = np.uint( x + wh2 + 1 )
        peakiness[b] = np.sum( posteriors[b, low:high] )
    peakiness_avg = np.array( [np.mean( peakiness[i-history:i] )\
                               for i in range(history,nbins+1)] )
    
    return peakiness[history-1:], peakiness_avg


def compute_content_consistency( posteriors, grid_size, content_nodes, history=3 ):
    '''
    
    '''
    xpeak = np.uint( np.argmax( posteriors, axis=1 )) # in grid indices
    nbins = len(posteriors)
    n_contents = len(content_nodes)
    online_contents = -1 * np.ones_like( xpeak, dtype='int16' )
    for content_idx, nodes in enumerate(content_nodes):
        m = np.logical_and( xpeak >= nodes[0], xpeak < nodes[1] )
        online_contents[m] = content_idx
    content_consistency = np.zeros( nbins-(history-1), dtype='bool' )
    for i in range(history,nbins+1):
        content_consistency[i-history], _ = is_consistent_content(\
            online_contents[i-history:i], n_contents )
    
    return content_consistency


def compute_consitency_ratio( posteriors, grid_size, content_nodes, history=3 ):
    '''
    
    '''
    content_consistency = compute_content_consistency( posteriors, grid_size,\
                                                      content_nodes, history=3 )
    
    return content_consistency.sum()/len(posteriors)


class DecodingMachine():
    
    def __init__(self, tt_included, spike_ampl_mask_list, decoder, spike_features,\
        equalized_content_nodes, grid_size, win_half_low, win_half_high, bin_size ):
        self.tt_included = tt_included
        self.spike_ampl_mask_list = spike_ampl_mask_list
        self.decoder = decoder
        self.spike_features = spike_features
        self.equalized_content_nodes = equalized_content_nodes
        self.grid_size = grid_size
        self.win_half_low = win_half_low
        self.win_half_high = win_half_high
        self.bin_size = bin_size
    
    def compute_posteriors_combo( self, combo, bins, lazy_decode=False ):
        '''
        '''
        tt_include_use = self.tt_included.copy()
        for c in combo:
            tt_include_use[c] = False
            
        spike_ampl_mask_list_use = np.array(\
            [ sam for i,sam in enumerate(self.spike_ampl_mask_list) if tt_include_use[i]] )   
        
        #assert( np.isclose( self.bin_size, sp.stats.mode( bins.duration )[0][0] ) )
        
        if lazy_decode:
            posteriors, _, n_spikes, _ = self.decoder.lazy_decode( self.spike_features,
                bins, tt_include_use, spike_ampl_mask_list_use, self.bin_size,\
                sf_keys=["spike_times", "spike_amplitudes"] )
        else:
            posteriors, _, n_spikes, _ = self.decoder.decode( self.spike_features,
                bins, tt_include_use, spike_ampl_mask_list_use, self.bin_size,\
                sf_keys=["spike_times", "spike_amplitudes"] )
    
        return posteriors, n_spikes

    def compute_sharpness_combo( self, combo, bins, lazy_decode=False ):
        '''
        '''
        posteriors = self.compute_posteriors_combo( combo, bins, lazy_decode=lazy_decode )[0]
        
        posterior_event = adjust_posteriors( posteriors, self.equalized_content_nodes )
    
        return compute_sharpness( posterior_event,\
             self.grid_size, self.win_half_low, self.win_half_high )
             
    def compute_consistency_combo( self, combo, bins, lazy_decode=False):
         '''
         '''
         posteriors = self.compute_posteriors_combo( combo, bins, lazy_decode=lazy_decode )[0]
        
         posterior_event = adjust_posteriors( posteriors, self.equalized_content_nodes )
         
         return compute_consitency_ratio( posterior_event,\
             self.grid_size, self.equalized_content_nodes )
        

def transform_spike_format( tt_spikes ):
    '''
    from preprocessed format (used for offline spikes) to format used for
    online spikes analyzed during experiment
    '''
    falcon_spikes = []
    for tt_s in tt_spikes.keys():
        falcon_spikes.append( {
            'id' : tt_s,
            'time' : tt_spikes[tt_s]['spike_times'],
            'value' : tt_spikes[tt_s]['spike_amplitudes'] } )
        
    return falcon_spikes


def load_reference_replay( path_to_reference ):
    '''        
    '''
    
    pmj = pickle.load( open( path_to_reference, 'r' ) )
    # add fields in case they don't exist in the loaded version of the perfomnace meter object
    pmj.tp_semi = 0
    pmj.fp_semi = 0
    pmj.fn_semi = 0
    pmj.tn_semi = 0
    pmj.two_semi = 0
    
    return pmj

    
def argmax( array ):
    '''
    returns argmax index in array coordinates
    '''
    return np.unravel_index( np.nanargmax(array), array.shape)
    

def performance_mua_pnks( simulator, pmj, mua_range, pkns_range ):
    '''
    '''
    if not simulator._variables_computed:
        raise ValueError( "No variables computed in the simulator" )
        
    performance = []
    mua_thresholds = []
    for mua_std_thr in mua_range:            
        performance.append( [] )
        mua_thresholds.append( [] )
        for pkns_thr in pkns_range:
            
            perf, mua_thr = simulator.compute_performance(\
                       pmj, [mua_std_thr, pkns_thr, pkns_thr], lockout=.075 )
            
            performance[-1].append( perf )
            mua_thresholds[-1].append( mua_thr )
    
    return performance, mua_thresholds
    
