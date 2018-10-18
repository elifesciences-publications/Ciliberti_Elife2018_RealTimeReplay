# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:45:46 2015

@author: davide
"""

from __future__ import division
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from os.path import join
import os
import errno
from pylab import cm, imshow

from fklab.io import neuralynx as nlx
from compressed_decoder.kde import MergingCompressionDensity as mcd
from compressed_decoder.kde import Decoder, save_mixture
from falcon.io.deserialization import load_file

import support_tools as tools
import config_file as config

# read Videotrack data
print "Loading the video data"
if config.fullpath_to_VTdata[-3:] == "nvt":
    videotrack_file = nlx.NlxOpen( config.fullpath_to_VTdata )
    vt_position = videotrack_file.readdata( start=videotrack_file.starttime,\
        stop=videotrack_file.endtime )
else:
    vt_position_data, header = load_file( config.fullpath_to_VTdata )     
    vt_position = tools.VT_Position( vt_position_data )

# define environment
print "I'm setting the environment"
multi_arm_maze, pixel_to_cm =\
    tools.set_multi_track_maze( config.path_to_environmentfile, config.path_length_cm )
n_tracks = len( multi_arm_maze )


# extract linear position and run segments
print "Linearizing position and extracting run segments"
linear_d_track = []
sel_on_track = []
RUN_track = []
for track in multi_arm_maze:
    ld, sel = tools.extract_linear( vt_position, track, config.max_occlusion_time,\
            config.max_dist_to_path, config.max_dist_to_path_occlusions, config.plot_env )
    if config.speed_from_velocity:
        run, _ =  tools.extract_run( np.array( [ vt_position.x, vt_position.y ]),\
            vt_position.time, config.velocity_bw, config.min_speed, pixel_to_cm, sel )   
    else:
        run, _ =  tools.extract_run( ld, vt_position.time, config.velocity_bw,\
            config.min_speed, pixel_to_cm, sel ) 
    linear_d_track.append( ld )
    sel_on_track.append( sel )
    RUN_track.append( run )

# superimpose room image to VT points
maze_img = plt.imread( config.path_to_room_image )
plt.imshow( maze_img )

linear_d, sel_on_maze = tools.all_tracks_behavior( linear_d_track, sel_on_track,\
    distance=config.extra_distance )
    
RUN = tools.combine_epochs( RUN_track )
selected_run = RUN[ RUN.duration > config.min_run_duration ]


# select test and train partitions
test, train = tools.partition( selected_run, config.n_partitions,\
    partition_index=config.partition_index, fold=config.fold,\
    method=config.partition_method, coincident=False )
test_binned = test.split( size=config.bin_size_run )

# load spike files
print ("I'm loading the spike features from " + config.spike_features_filename)
spike_features = np.load( config.spike_features_filename )

# prepare encoding
sel_behav = train.contains( vt_position.time )[0]
train_behav = np.vstack([linear_d[sel_behav], vt_position.time[sel_behav]])
train_spike = []
tt_included = np.zeros( len(spike_features), dtype='bool' )
n_tt_all = len(spike_features)
spike_ampl_mask_list = []
id_selected_sensors = []
print "Number of training spikes per tetrode"
for i, tt in enumerate(spike_features):
    assert( len(tt['time']) > 0 )
    sel_train_spike = train.contains(tt["time"])[0]
    print(sum(sel_train_spike))
    if sum(sel_train_spike) > config.min_n_spikes:  
        tt_sel = tt['value'][sel_train_spike]
        if len( np.shape( tt_sel ) ) == 1:
            tt_sel = np.expand_dims(tt_sel, axis=1)
        spike_ampl_mask = np.logical_not( np.median(tt_sel, axis=0) == 0 )
        # remove broken channels with no data
        tt_sel = tt_sel[:, spike_ampl_mask].T
        train_spike.append(np.vstack((tt_sel, tt['time'][sel_train_spike])))
        tt_included[i] = True
        spike_ampl_mask_list.append( spike_ampl_mask )
        id_selected_sensors.append( tt["id"] )
    else:
        print tt["id"]  + " excluded"
      
    
n_selected_sensors = len( spike_ampl_mask_list )

assert( train_spike != [] )

#### ENCODING ####
print "I'm creating the encoding model ..."
# create bandwidth matrix
covars = []
spf_dim = 4
behav_ndim = 1
for i in range( behav_ndim ):
    covars.append( (config.behav_bw_cm * pixel_to_cm ) **2)
for i in range( spf_dim ):
    covars.append( (config.spf_bw * 1000)**2 )
    
assert (len(train_spike) == len(spike_ampl_mask_list))

# created encoding points
encoding_points = []
for tt in train_spike:
    encoding_points.append( tools.attach( train_behav, tt ) )
    
# compressed joint density 
mcd_spikebehav = []
for i in range(n_selected_sensors):
    cov_tt = np.array( covars )[np.concatenate((np.array([True]),\
        spike_ampl_mask_list[i]))].tolist()
    mcd_spikebehav.append( mcd( ndim=len(cov_tt), sample_covariance =\
        cov_tt, method='bandwidth', threshold=config.compression_threshold,\
        name=(id_selected_sensors[i] ) ) )
for dec, points in zip( mcd_spikebehav, encoding_points ):
    dec.addsamples( points )
    
# uncompressed density of behavior data
mcd_behav = mcd( train_behav[0], sample_covariance=covars[:1], threshold=0, name='behavior' )
training_time = np.sum(train.duration)


#### DECODING ####
# define grid
half_grid_size_pxl = config.grid_element_size / 2
xgrid_vector = np.arange( linear_d.min() + half_grid_size_pxl, linear_d.max() - half_grid_size_pxl,\
    config.grid_element_size )
grid = xgrid_vector.reshape( (len(xgrid_vector),1) )

# decode during test run epochs
print "I'm decoding {0} run epochs with {1} grid elements " .format(len(test), len(grid))
decoder = Decoder( mcd_behav, mcd_spikebehav, training_time, grid, config.offset )

posterior, logpos, n_spikes, n_tt =\
    decoder.decode( spike_features[tt_included], test_binned,\
        np.ones(n_selected_sensors, dtype='bool'), spike_ampl_mask_list,\
    config.bin_size_run )
    
decoded_behavior = grid[np.nanargmax(logpos, axis=1)].flatten()

# compute decoding error
true_behavior = sp.interpolate.interp1d( vt_position.time[sel_on_maze],\
    linear_d[sel_on_maze], kind='linear', axis=0 ) (test_binned.center)       
assert(len(true_behavior) == len(decoded_behavior))
n_test_bins = len(decoded_behavior)
errors = np.array([np.linalg.norm(pred_i - true_behav_i) \
    for pred_i, true_behav_i in zip(decoded_behavior, true_behavior)])
error = np.median(errors)
print "\nCompressed decoding completed with {0} tetrodes done with median error of:\t{1} cm"\
    .format(n_tt, error/pixel_to_cm)
print "\nGrid element size : {0} cm" .format( config.grid_element_size / pixel_to_cm )


# define content_nodes
content_nodes = np.zeros( (n_tracks, 2) )
max_ld = np.max(linear_d_track, axis=1) + np.arange( n_tracks ) * config.extra_distance
bw_extension = np.uint( config.behav_bw_cm * pixel_to_cm / config.grid_element_size *\
    config.content_node_extension )
print "Content nodes were extended of {0} grid elements" .format( bw_extension )
for tr in range(n_tracks):
    content_nodes[tr, 0] =\
    np.nonzero(grid > config.extra_distance * tr)[0][0] - bw_extension
    content_nodes[tr, 1] = np.nonzero(grid < max_ld[tr])[0][-1] + bw_extension
content_nodes[0, 0] = 0
content_nodes[ n_tracks - 1, 1] = len(grid) - 1
content_nodes = np.int32( content_nodes ) # for export to Falcon  

##### SAVING #####
if config.save_linear_d_single_track:
    np.save( open( config.path_to_processed_behavior, "w" ),\
        [linear_d_track, sel_on_track] )
  
if config.save_encoding_model:
    tools.clean_directory( config.path_to_daily_model )
    np.save( open( join( config.path_to_daily_model, "covars.npy" ), "w"),\
        covars )
    np.save( open( join( config.path_to_daily_model, "T.npy" ), "w" ),\
        training_time )
    np.save( open( join( config.path_to_daily_model, "tetrode_mask.npy" ), "w" ),\
        tt_included )
    np.save( open( join( config.path_to_daily_model, "channel_mask.npy" ), "w" ),\
        spike_ampl_mask_list )
    np.save( open( join( config.path_to_daily_model, "pix.npy" ), "w" ),\
        decoder.pix() )  
    save_mixture( mcd_behav, join( config.path_to_daily_model, "mixture_" +\
        mcd_behav.name + ".dat" ) )
    tt_s = "TT"
    for i, density in enumerate(mcd_spikebehav):
        tt_folder_path = join( config.path_to_daily_model, density.name )
        if not os.path.exists( tt_folder_path ):
            try:
                os.makedirs( tt_folder_path )
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        save_mixture( density, join( tt_folder_path, "mixture_pax.dat" ) )
        np.save( open( join( tt_folder_path, "lambda_x.npy" ), "w"), decoder.lx()[i] )
        np.save( open( join( tt_folder_path, "mu.npy" ), "w"), decoder.mu[i] )
        np.save( open( join( tt_folder_path, "pix.npy" ), "w" ), decoder.pix() )
        np.save( open( join( tt_folder_path, "offset.npy" ), 'w'), config.offset )
        print "\nModel, precomputations and support data of " + density.name +\
            " saved in " + tt_folder_path
    
        print "\nUse this string in Falcon to index to the tetrode indices"
    print( tools.create_compact_list([ int(_id[2:]) for _id in id_selected_sensors]) )
            
if config.save_environment:
    conv_factor = 1
    if config.unit_grid == 'cm':
        conv_factor = pixel_to_cm
    np.save( open( join( config.path_to_daily_model, "grid.npy" ) , "w" ),\
        grid.ravel() / conv_factor )
    np.save( open( join( config.path_to_environment, "grid.npy"),'w'),\
        grid.ravel() / conv_factor )
    print "\nGrid was saved in the two specified locations"
    np.save( open( join( config.path_to_environment, "content_nodes.npy" ), 'w'), content_nodes )
    print "\nContent nodes of {0} arms were saved" .format( n_tracks )

### PLOTTING ### 
cmap = tools.get_cmap(n_tracks+1, .3)
# plot train and test behavior
if config.plot_train_test:
    fig2, ax1 = plt.subplots(nrows=2, ncols=1,figsize=(20, 12) )
    ax1[0].plot(train_behav[0], linewidth=3 )
    ax1[0].set_title("train behavior")
    ax1[0].set_xlabel("behavior samples")
    ax1[0].set_ylabel("linear position [pixel]")
    for i, nodes in enumerate( content_nodes):
        ax1[0].fill_between( np.arange(len(train_behav[0])), grid[nodes[0]][0],\
            grid[nodes[1]][0], color=cmap(i))
    ax1[0].grid()
    ax1[1].plot( linear_d[test.contains(vt_position.time)[0]], linewidth=3  )
    ax1[1].set_xlabel("behavior samples")
    ax1[1].set_title("test behavior")
    ax1[1].set_ylabel("linear position [pixel]")
    for i, nodes in enumerate( content_nodes):
        ax1[1].fill_between( np.arange( len(linear_d[test.contains(vt_position.time)[0]]) ),\
            grid[nodes[0]][0], grid[nodes[1]][0], color=cmap(i))
    ax1[1].grid()

# plot decoded and true position during test run epochs
time_points = np.arange(0, len(true_behavior))
if config.plot_dec_behav:
    
    fig3 = plt.figure(3, figsize=(30,5))
    plt.clf()
    plt.plot(time_points, true_behavior, color='black', linewidth=2)
    plt.plot(time_points, decoded_behavior, color='grey', linewidth=2)
    plt.figtext(0.25, 0.96, "true position", fontsize='xx-large',\
        color='black', ha ='right', weight='bold')
    plt.figtext(0.50, 0.96, "decoded position", fontsize='xx-large',\
        color='grey', ha ='right', weight='bold')
    plt.yticks(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    for i, nodes in enumerate( content_nodes):
        plt.fill_between( time_points, grid[nodes[0]][0],\
            grid[nodes[1]][0], color=cmap(i))
    plt.grid()
    
if config.plot_dec_error:
    fig4 = plt.figure(4)
    plt.boxplot(errors / pixel_to_cm)
    plt.ylim(ymax = 30)
    plt.ylabel("decoding error [cm]", fontsize='xx-large',  weight='bold')
    plt.xticks([""])
    plt.yticks(fontsize="xx-large", weight = 'bold')
    
    fig5 = plt.figure(5) # plot the cumulative function
    values, base = np.histogram(errors, bins=400)
    cumulative = np.double(np.cumsum(values))
    plt.plot(base[:-1], cumulative/cumulative.max(), c='blue')
if config.plot_posterior_run:
    fig6 = plt.figure(6, figsize=(60, 20))
    plt.clf()
    imshow(posterior.T, cmap=cm.gist_heat, interpolation='none')
    plt.figtext(0.2, 0.60, "Posteriors of RUN epochs", fontsize='xx-large',\
        color='gray', ha ='right', weight='bold')
    
if config.plot_occupancy:
    fig7 = plt.figure(7, figsize=(10,4))
    plt.title("occupancy")
    fig5 = plt.plot(decoder.pix())


plt.show()