# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:39 2016

@author: davide
"""
from __future__ import division
import numpy as np
from os.path import join, dirname
import cPickle as pickle
import itertools
from scipy import special
from matplotlib import pyplot as plt
import support_replay_analysis.tools as tools

from fklab.io.preprocessed_data import read_dataset, check_ephys_dataset
from fklab.segments import Segment as Seg

import mua_undersampling_config as config

BIN_SIZE = 0.01
HISTORY = 3

nbins = 200
N_BOOTSTRAP = 50

def compute_mua_avg( nspikes, combo ):
    '''
    '''
    use = np.ones( n_tt_dataset, dtype='bool' )
    for c in combo: use[c] = False
    
    n_spikes_dataset = np.sum( n_spikes[use], axis=0)
    
    mua = n_spikes_dataset/BIN_SIZE
    mua = np.hstack( (np.zeros(HISTORY-1), mua[:-(HISTORY-1)]) )
    mua_avg = np.array( [np.mean( mua[i:i+HISTORY] ) for i in range(len(mua))] )
    
    return mua_avg
    

# read data
_, ephys, startstop = read_dataset( config.path_to_preprocessed_dataset ) 
check_ephys_dataset( ephys )
n_tt_dataset = len(ephys)
ephys_list = [ ephys[k] for k in ephys.keys()]

n_bins = np.int32(np.diff(startstop)/BIN_SIZE)[0]
n_spikes = np.array( [ np.histogram( tt['spike_times'], n_bins )[0] for tt in ephys_list ] )
    
# load candidate events
candidate = pickle.load( open( join( dirname( config.path_to_preprocessed_dataset ),\
    "candidate_events.p"), 'r') )
# add two bin to each candidate
candidate_ext = Seg.fromevents(candidate.start-(HISTORY-1)*BIN_SIZE, candidate.stop)

# load performance meter
pm = pickle.load( open( config.path_to_performance, 'r') )
assert( len(candidate_ext) == pm.n_events )

# separate non-candidate and with and without content events
non_candidate_ext = Seg.fromarray( [startstop[0], startstop[1]]).difference( candidate_ext )
candidate_ext_with = candidate_ext[np.logical_or(pm.is_joint, pm.is_single)]
candidate_ext_without = candidate_ext[~np.logical_or(pm.is_joint, pm.is_single)]

mua_avg_times = np.arange( startstop[0], startstop[1], BIN_SIZE) + BIN_SIZE

overlap_noncand_with = []
overlap_noncand_without = []
overlap_with_without = []
overlap_cand_noncand = []

mua_avg_noncand_all = []
mua_avg_with_all = []
mua_avg_without_all = []

for n_removed in range(n_tt_dataset):
    
    print "\n{0} tetrodes excluded" .format(n_removed)
    
    n_combos = int(special.binom( n_tt_dataset, n_removed ))
    n_bootstrap = min(n_combos, N_BOOTSTRAP)
    
    overlap_noncand_with.append( np.zeros(n_bootstrap) )
    overlap_noncand_without.append( np.zeros(n_bootstrap) )
    overlap_with_without.append( np.zeros(n_bootstrap) )
    overlap_cand_noncand.append( np.zeros(n_bootstrap) )
    
    mua_avg_noncand_all.append([])
    mua_avg_with_all.append([])
    mua_avg_without_all.append([])
    
    combo_indices = np.random.choice( range(n_combos), n_bootstrap, replace=False )
    combo_indices.sort()
    
    mua_avg = compute_mua_avg( n_spikes, () )
    rng = [0, mua_avg.max()]
    
    j = 0

    for i,combo in enumerate(itertools.combinations( range(n_tt_dataset), n_removed )):
        
        if i not in combo_indices:
            continue
        
        use = np.ones( n_tt_dataset, dtype='bool' )
        for c in combo: use[c] = False
        
        mua_avg = compute_mua_avg( n_spikes, combo )
        
        mua_avg_noncand = mua_avg[non_candidate_ext.contains( mua_avg_times )[0]]
        mua_avg_with = mua_avg[candidate_ext_with.contains( mua_avg_times )[0]]
        mua_avg_without = mua_avg[candidate_ext_without.contains( mua_avg_times )[0]]
        mua_avg_cand = mua_avg[candidate_ext.contains( mua_avg_times )[0]]
        
        mua_avg_noncand_all[-1].append( mua_avg_noncand )
        mua_avg_with_all[-1].append( mua_avg_with )
        mua_avg_without_all[-1].append( mua_avg_without )
        
        h_noncand = np.histogram( mua_avg_noncand, bins=nbins, range=rng, density = True)[0]
        h_with = np.histogram( mua_avg_with, bins=nbins, range=rng, density=True )[0]
        h_without = np.histogram( mua_avg_without, bins=nbins, range=rng, density=True )[0]
        h_cand = np.histogram( np.hstack( (mua_avg_with, mua_avg_without) ), bins=nbins,\
                                         range=rng, density=True )[0]
        
        h_noncand = h_noncand/sum(h_noncand)
        h_with = h_with/sum(h_with)
        h_without = h_without/sum(h_without)
        h_cand = h_cand/sum(h_cand)
        
        overlap_noncand_with[-1][j] = tools.bhattacharyya( h_noncand, h_with )
        overlap_noncand_without[-1][j] = tools.bhattacharyya( h_noncand, h_without )
        overlap_with_without[-1][j] = tools.bhattacharyya( h_with, h_without )
        overlap_cand_noncand[-1][j] = tools.bhattacharyya( h_cand, h_noncand )

        j += 1
        if j%10 == 0:
            print "\n\t{0} % completed" .format(int(j/n_bootstrap*100))


combined = [np.hstack( (tt[0],tt[1])) for tt in zip(mua_avg_with_all, mua_avg_without_all) ]

## plotting
plt.figure()
j=0
for c in range(7):
    for r in range(2):
        plt.subplot(2,7,j+1)
        h_noncand = plt.hist( np.array(mua_avg_noncand_all[j]).ravel(), bins=nbins, density=True,\
                 range=rng, color='cyan', alpha=.5, label='non-cand' )[0]
        h_with = plt.hist( np.array(mua_avg_with_all[j]).ravel(), bins=nbins, density=True,\
                 range=rng, color='yellow', alpha=.5, label='with content'  )[0]
        h_without = plt.hist( np.array(mua_avg_without_all[j]).ravel(), bins=nbins, density=True,\
                 range=rng, color='magenta', alpha=.5, label='without content'  )[0]
        plt.title( str(14-j) + " tetrodes" )
        plt.xlim( rng)
        if j ==0: plt.legend( loc='best' )
        j += 1
        
plt.figure()
j=0
for c in range(7):
    for r in range(2):
        plt.subplot(2,7,j+1)
        h_noncand = plt.hist( np.array(mua_avg_noncand_all[j]).ravel(), bins=nbins, density=True,\
                 range=rng, color='cyan', alpha=.5, label='non-cand' )[0]
        h_cand = plt.hist( np.array(combined[j]).ravel(), bins=nbins, density=True,\
              range=rng, color='brown', alpha=.5, label='cand' )[0]
        plt.title( str(14-j) + " tetrodes" )
        plt.xlim( rng)
        if j ==0: plt.legend( loc='best' )
        j += 1

plt.figure()
x = range(n_tt_dataset)
plt.plot( x, [ np.median(tt) for tt in mua_avg_noncand_all],\
         color='cyan', label='non-cand' )
#plt.fill_between( x, [ np.percentile(tt, 5) for tt in mua_avg_noncand_all],\
#  [ np.percentile(tt, 95) for tt in mua_avg_noncand_all], color='cyan', alpha=.2 )
plt.plot( x, [ np.median(tt) for tt in mua_avg_with_all],\
         color='yellow', label='with content' )
#plt.fill_between( x, [ np.percentile(tt, 5) for tt in mua_avg_with_all],\
#  [ np.percentile(tt, 95) for tt in mua_avg_with_all], color='green', alpha=.2 )
plt.plot( x, [ np.median(tt) for tt in mua_avg_without_all],\
         color='magenta', label='without content' )
#plt.fill_between( x, [ np.percentile(tt, 5) for tt in mua_avg_without_all],\
#  [ np.percentile(tt, 95) for tt in mua_avg_without_all], color='magenta', alpha=.2 )
plt.legend( loc='best' )
plt.xlim( xmin=0 )
plt.xlabel( "# tetrodes removed" )
plt.ylabel( "3-bin online MUA [spikes/s]" )

plt.figure()
plt.plot( range(n_tt_dataset), [ np.median(tt) for tt in mua_avg_noncand_all], color='cyan',  alpha=.75, label='non-cand' )
plt.plot( range(n_tt_dataset), [ np.median(tt) for tt in combined], color='brown',  alpha=.75, label='cand' )
plt.legend( loc='best' )
plt.xlabel( "# tetrodes removed" )
plt.ylabel( "3-bin online MUA [spikes/s]" )

plt.figure()
plt.plot( [ np.median( tt ) for tt in overlap_noncand_with ], label='noncand-with' )
plt.plot( [ np.median( tt ) for tt in overlap_noncand_without ], label='noncand-without')
plt.plot( [ np.median( tt ) for tt in overlap_with_without ], label='with-without' )
plt.plot( [ np.median( tt ) for tt in overlap_cand_noncand ], label='cand-noncand' )
plt.ylim( [0,1] )
plt.xlim( xmin=0 )
plt.ylabel("overlap as bhattacharyya distance")
plt.xlabel("# removed tetrodes")
plt.legend( loc='best')
        


plt.show()

