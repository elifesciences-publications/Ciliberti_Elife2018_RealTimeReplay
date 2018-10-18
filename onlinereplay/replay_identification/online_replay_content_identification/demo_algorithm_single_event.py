#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:50:49 2017

@author: davide
"""
from __future__ import division
import numpy as np
import cPickle as pickle
from os.path import join, dirname
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from fklab.plot import plot_raster
from fklab.io import neuralynx as nlx
from fklab.utilities import yaml

from falcon.io.deserialization import load_likelihood_data

import support_replay_analysis.tools as tools
import demo_algorithm_single_event_config as config

mpl.rcParams.update(mpl.rcParamsDefault)

BIN_SIZE = 0.01
H = 3
HALF_INTEGRATION_WINDOW = 4
n_contents = 2

# load data
falcon_spikes = np.load( join( config.path_to_falcon_dataset, "all_spike_amplitudes.npy") )
candidate = pickle.load( open( join( dirname( config.path_to_nlx_preprocessed_dataset ),\
    "candidate_events.p"), 'r') )
with open( join( dirname( config.path_to_nlx_preprocessed_dataset ),\
"preprocessing_info.yaml" ), 'r' ) as prepr_f:
    info_preprocess = yaml.load(prepr_f)
n_tt = len(falcon_spikes)

# load environment info
content_nodes = np.load( join ( config.path_to_environment, "content_nodes.npy" ) )
xgrid_vector_pixel = np.load( join( config.path_to_environment, "grid.npy" ) )
xgrid_vector_cm = xgrid_vector_pixel / info_preprocess["ConversionFactorPixelsToCm"]
grid = xgrid_vector_cm.reshape( (len(xgrid_vector_cm),1) )
grid_element_length = np.mean( np.diff( grid.ravel() ) )
new_content_nodes, equalized_content_nodes = tools.fix_content_nodes(\
    content_nodes, equalize=True,\
    arm_length_grid_units=np.int(info_preprocess["ArmLength_cm"]/grid_element_length+1 ) )
adjusted_grid = grid[:new_content_nodes[-1, -1]+1] 

# select candidate event and surrounding data
candidate_extended = candidate[config.CANDIDATE_ID]
candidate_extended.start = candidate[config.CANDIDATE_ID].start - config.extension_s

# read posteriors 
online_posteriors_filepath= join( config.path_to_falcon_dataset,\
                "sink_synch/sink_synch.0_synchronizer.estimates.0.bin" )
likelihood_data, grid_size, n_bins, data, header =\
    load_likelihood_data( online_posteriors_filepath )
log_posterior_allbins = likelihood_data['log_likelihood']
posterior_times = data["hardware_ts"]/1e6
posterior_allbins = np.exp(\
    log_posterior_allbins - np.max(log_posterior_allbins, axis=1)[:, np.newaxis] )
posterior_allbins = posterior_allbins / np.nansum( posterior_allbins, axis=1 )[:, np.newaxis]
posterior_candidate_mask = candidate_extended.contains( posterior_times )[0]
posteriors_plot = posterior_allbins[posterior_candidate_mask]
pos = tools.adjust_posteriors( posteriors_plot, equalized_content_nodes )  
pos_times = posterior_times[posterior_candidate_mask]

# compute params online replay content identification
attachment = np.zeros( H -1 )
nbins_plot = pos.shape[0]
nspikes_per_bin = np.hstack( ( attachment, likelihood_data["n_spikes"][posterior_candidate_mask]) )
mua_hist_avg =\
    np.array( [np.mean( nspikes_per_bin[i:i+H] ) for i in range(nbins_plot)] )
xpeak = np.uint( np.hstack( (attachment, np.argmax( pos, axis=1 )) ) ) # in grid indices
win_half_low = [ min( HALF_INTEGRATION_WINDOW, g ) for g in range(grid_size) ] # in grid indices
win_half_high = [ min( HALF_INTEGRATION_WINDOW, len(grid) - g - 1 )\
    for g in range(grid_size) ] # in grid indices
sharpness = np.zeros( nbins_plot )
for b in range(nbins_plot):
    x = xpeak[b+H-1]
    wh1, wh2 = (win_half_low[x], win_half_high[x])
    low = np.uint( x - wh1 )
    high = np.uint( x + wh2 + 1 )
    sharpness[b] = np.sum( pos[b, low:high] )
#sharpness_hist_avg = np.zeros_like( nspikes_per_bin )
sharpness_hist_avg = np.hstack( ( attachment, np.array( [np.mean(sharpness[i:i+H])\
                                 for i in range(nbins_plot) ] )[:-2] ) )
content_consistency = np.zeros( nbins_plot, dtype='bool' )
online_contents = -1 * np.ones_like( xpeak, dtype='int16' )
for content_idx, nodes in enumerate(new_content_nodes):
    m = np.logical_and( xpeak >= nodes[0], xpeak < nodes[1] )
    online_contents[m] = content_idx
for i in range(nbins_plot):
    content_consistency[i], _ = tools.consistent_content(\
        online_contents[i:i+H], n_contents )

if config.clamp_posteriors != False:
    n_clamped = np.sum(pos>config.clamp_posteriors)
    pos[pos>config.clamp_posteriors] = config.clamp_posteriors
    print "{0} values clamped at {1}" .format( n_clamped, config.clamp_posteriors )

# read spikes
falcon_spike_times = [ falcon_spikes[tt]["time"] for tt in range(n_tt) ]
spike_times = [ [] for n in range(n_tt) ]
                      
# read TTL events
f_nlx_events = nlx.NlxOpen( join( dirname( config.path_to_nlx_preprocessed_dataset ),\
                                 "Events.nev" ) )
all_events = f_nlx_events.readdata( start=candidate_extended.start, stop=candidate_extended.stop )
if config.replay_string not in np.unique( all_events.eventstring ):
    raise ValueError("Replay event string " + config.replay_string + " not found: try with: " +\
    "TTL Input on AcqSystem1_0 board 0 port 3 value (0x0040).")
replay_ttl_times = all_events.time[all_events.eventstring == config.replay_string]
if len(replay_ttl_times) > 1:
    print "Extended candidate event has more than one TTL detection"
t_ttl = replay_ttl_times[config.detection]


############################# PLOT ###################3################

# prepare figure
fig = plt.figure( figsize=(config.fig_width,config.fig_height) )
gs = gridspec.GridSpec( 5, 1 )
gs.update( top=0.88, bottom=0.11, left=0.12, right=0.90, hspace=.2, wspace=0.2 )
x_ttl = candidate_extended.duration/BIN_SIZE *\
    ( (t_ttl-candidate_extended.start)/candidate_extended.duration )
xticks = np.arange( 0, nbins_plot, step=20)
xtick_labels = [ str(xt) for xt in np.int32((xticks)*10 - config.extension_s/BIN_SIZE*10) ]
adjust = candidate_extended.start - pos_times[0]
xticks = xticks + adjust
x_ttl = x_ttl + adjust
x_detection = np.round( x_ttl )
t_detection = np.round( t_ttl*1e2 )/1e2 - BIN_SIZE/2


# RASTER PLOT
ax1 = plt.subplot( gs[0] )
plt.xlim( candidate_extended.start, candidate_extended.stop )
plt.ylim( 0, n_tt+1 )
#ax1.tick_params( axis=u'both', which=u'both',length=0 )
fastrasters = plot_raster( [candidate_extended.start+10e-3 for n in range(n_tt)],\
    color='k', spacing=1.2, linewidth=1, background_color='white', background_alpha=0.05 )
for n,fr in enumerate(fastrasters):
    fr.set_data( candidate_extended.start+10e-3 )
    spike_times = [ falcon_spike_times[tt][candidate_extended.contains(\
       falcon_spike_times[tt] )[0]] for tt in range(n_tt) ]
    fr.set_data( spike_times[n] )
plt.ylabel( "tetrode\nindex", fontsize=config.fs, rotation="horizontal" )
ax1.yaxis.set_label_coords( config.ylabel_coord1, config.ylabel_coord2 )
xticks_raster = np.arange( candidate_extended.start, candidate_extended.stop,\
                          step=10*BIN_SIZE ) + adjust*1e-3
plt.xticks( xticks_raster, color='none' )
ax1.set_xticklabels( xtick_labels, color='none' )
ax1.set_yticks( [ax1.get_yticks()[0], ax1.get_yticks()[-1]] )
ax1.set_yticklabels( ["1", str(n_tt-1)], fontsize=config.fs )
ax1.vlines( t_detection, 0, n_tt+1, color='red', lw=config.lw,\
           alpha=config.alpha_detections, zorder=config.z_order_detections )

#plt.grid()

# POSTERIOR
ax2 = plt.subplot( gs[1] )
ylimit = (adjusted_grid[-1]+adjusted_grid[0])[0]
time_bins = pos_times + BIN_SIZE/2
pos_bins = adjusted_grid
im = plt.imshow( pos.T, interpolation='none', origin='lower', aspect='auto',\
        cmap=plt.cm.gray_r, extent=[xticks[0], nbins_plot, 0, ylimit] )
plt.ylabel('linearized\nposition', fontsize=config.fs, rotation="horizontal"  )
ax2.yaxis.set_label_coords( config.ylabel_coord1, config.ylabel_coord2 )
ax2.set_xticks( xticks )
ax2.set_xticklabels( xtick_labels, color='none' )
plt.yticks( [] )
plt.tick_params( left='off', right='off', bottom='on' )
ax2.spines["top"].set_visible( False )
ax2.spines["right"].set_visible( False )
ax2.spines["left"].set_visible( False )
ax2.spines["bottom"].set_visible( False )
# add frames to posterior plot
ax2.hlines( np.vstack( (adjusted_grid[new_content_nodes[:, 0]],\
    adjusted_grid[-1]+adjusted_grid[0]) ), xticks[0], nbins_plot, color='black',\
    linestyle="solid", lw=config.lw/2 )
ax2.vlines( [0+adjust, nbins_plot], 0, adjusted_grid[-1]+adjusted_grid[0],\
   color='black', linestyle="solid", lw=config.lw/2 )
ax2.vlines( x_detection, adjusted_grid[new_content_nodes[config.content, 0]],\
    adjusted_grid[new_content_nodes[config.content, 1]], color='red',\
    lw=config.lw, alpha=config.alpha_detections, linestyle='solid' )

#plt.grid()

# MUA
mua_falcon = np.load( config.mua_path )
mua_threshold = np.mean( mua_falcon ) + config.mua_threshold_std * np.std( mua_falcon )
mua_threshold = mua_threshold*BIN_SIZE
ax3 = plt.subplot( gs[2], sharex=ax2 )
t = np.arange( 1, nbins_plot+1, step=1 )
plt.plot( t, mua_hist_avg, color='k', lw=config.lw )
max_mua = mua_hist_avg.max()
ax3.vlines( x_detection, 0, max_mua+1, color='red', lw=config.lw ,\
           linestyles="solid", alpha=config.alpha_detections, zorder=config.z_order_detections )
plt.ylabel( "MUA\n[spikes/bin]", fontsize=config.fs, rotation="horizontal" )
ax3.yaxis.set_label_coords( config.ylabel_coord1, config.ylabel_coord2 )
plt.ylim( [0, max_mua*1.2] )
plt.yticks( np.arange(np.trunc(max_mua*1.2/10)*10+10, step=20, dtype='int32'),\
           fontsize=config.fs, rotation="horizontal" )
ax3.set_xticks( xticks )
ax3.set_xticklabels( xtick_labels, color='none' )
ax3.hlines( mua_threshold, xticks[0], nbins_plot, linestyle="dashed", alpha=config.alpha_detections,\
           lw=config.lw, color='gray', zorder=config.z_order_thresholds )

#plt.grid()

# SHARPNESS
ax4 = plt.subplot( gs[3], sharex=ax3 )
plt.plot( t, sharpness_hist_avg, color='k', lw=config.lw )
ax4.set_yticks( [0, .5, 1] )
ax4.set_yticklabels( ["0", "0.5", "1"], fontsize=config.fs )
ax4.set_xticklabels( xtick_labels, color='none' )
ax4.vlines( x_detection, 0, max_mua+1, color='red', lw=config.lw ,\
           linestyles="solid", alpha=config.alpha_detections, zorder=config.z_order_detections )
plt.ylabel( "sharpness\n[a.u.]", fontsize=config.fs, rotation="horizontal" )
ax4.yaxis.set_label_coords( config.ylabel_coord1, config.ylabel_coord2 )
plt.ylim( [0, 1] )
ax4.hlines( config.sharpness_threshold, xticks[0], nbins_plot, linestyle="dashed",\
           lw=config.lw, color='gray', alpha=config.alpha_detections, zorder=config.z_order_thresholds )

#plt.grid()

## DECODED CONTENT and CONTENT CONSISTENCY
ax5 = plt.subplot( gs[4], sharex=ax3 )
plt.plot( t[content_consistency]+.5, online_contents[2:][content_consistency],\
         lw=0, color='black', marker="s", markersize=2*config.lw, label="consistent" )
plt.plot( t[~content_consistency]+.5, online_contents[2:][~content_consistency],\
         lw=0, color='gray', marker="s", markersize=2*config.lw, label="inconsistent" )
ax5.set_xticklabels( xtick_labels, color='none' )
plt.ylim( [-.5, 2.5] )
ax5.set_yticks( range(n_contents) )
ax5.set_yticklabels( [str(yt+1) for yt in range(n_contents)], fontsize=config.fs )
plt.ylabel( "\ndecoded\ncontent\n[arm ID]", fontsize=config.fs, rotation="horizontal" )
ax5.yaxis.set_label_coords( config.ylabel_coord1, config.ylabel_coord2 )
ax5.vlines( x_detection, config.content-.5, config.content+.5, color='red', lw=config.lw ,\
           linestyles="solid", alpha=config.alpha_detections, zorder=config.z_order_detections )
plt.xlim( xticks[0], nbins_plot )
#plt.legend( loc="upper left", frameon=False )

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_tick_params( direction='in' )
    ax.yaxis.set_tick_params( direction='in' )
    ax.tick_params(axis='x', which='major', pad=config.padsize )
    
plt.show()
