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

from fklab.io import neuralynx as nlx
from fklab.segments import Segment as Seg
from falcon.io.deserialization import load_file
import support_tools as tools
    
import check_decoding_config_file as config



# READ VIDEOTRACK DATA
if config.fullpath_to_VTdata[-3:] == "nvt":
    videotrack_file = nlx.NlxOpen( config.fullpath_to_VTdata )
    if config.epoch == []:
        vt_position = videotrack_file.readdata( start=videotrack_file.starttime,\
                                               stop=videotrack_file.endtime )
    else:
        vt_position = videotrack_file.readdata( start=config.epoch[0],\
                                               stop=config.epoch[1] )
else:
    vt_position_data, VT_header = load_file( config.fullpath_to_VTdata )  
    vt_position_all = tools.VT_Position( vt_position_data )
    if config.epoch == []:
        vt_position = vt_position_all
    else:
        assert( len(config.epoch) == 2 )
        vt_position = tools.VT_Position()
        sel_mask = Seg( config.epoch ).contains( vt_position_all.time )[0]
        vt_position.x = vt_position_all.x[ sel_mask ]
        vt_position.y = vt_position_all.y[ sel_mask ]
        vt_position.time = vt_position_all.time[ sel_mask ]
print("\nVideotrack data was read")

# SET ENVIRONMENT
maze, pixel_to_cm =\
    tools.set_multi_track_maze( config.path_to_environmentfile, config.path_length_cm )
n_tracks = len( maze )

print "\nEnvironment was set"

# EXTRACT LINEAR POSITION AND RUN SEGMENTS
linear_d_track = []
sel_on_track = []
RUN_track = []
for track in maze:
    ld, sel = tools.extract_linear( vt_position, track, config.max_occlusion_time,\
            config.max_dist_to_path, config.max_dist_to_path_occlusions, config.plot_env )
    run, _ =  tools.extract_run( ld, vt_position.time, config.velocity_bw,\
        config.min_speed, pixel_to_cm, sel )        
    linear_d_track.append( ld )
    sel_on_track.append( sel )
    RUN_track.append( run )

linear_d, sel_on_maze = tools.all_tracks_behavior( linear_d_track, sel_on_track,\
    distance=config.extra_distance )

# compute ld boundaries
boundaries = np.zeros( n_tracks-1 )
boundaries[0] = (config.extra_distance-config.path_length_cm*pixel_to_cm)/2 +\
    config.path_length_cm*pixel_to_cm
for b in range(1, n_tracks-1):
    boundaries[b] = boundaries[b-1] + config.extra_distance
all_boundaries = np.zeros( n_tracks+1 )
all_boundaries[1:n_tracks] = boundaries
all_boundaries[n_tracks] = linear_d.max() + 1

RUN = tools.combine_epochs( RUN_track )
print "{0} run epochs" .format(len(RUN))
selected_RUN = RUN[ RUN.duration > config.min_run_duration ]
print "{0} run epochs selected (last longer than {1} s)" .format(\
       len(selected_RUN), config.min_run_duration)
print "\nPosition was linearized and run segments extracted"

# READ FALCON OUTPUT
data, behav_header = load_file( config.path_to_decoded_position )
    
# each position is marked by the initial timestamp of its decoding bin;
# add half decoding bin time to have the bin time in the middle of the bin
# as per the true position
decoded_behavior_t_all = data['hardware_ts']/1e6
if config.epoch == []:
    decoded_behavior = data["linear_position"]
    decoded_behavior_t = decoded_behavior_t_all
else:
    decoded_behavior = data["linear_position"][ Seg( config.epoch).contains(\
                           decoded_behavior_t_all )[0] ]
    decoded_behavior_t = decoded_behavior_t_all[ Seg( config.epoch).contains(\
                           decoded_behavior_t_all )[0] ]
decoding_bin = np.median( np.diff( decoded_behavior_t ) )
decoded_behavior_t = decoded_behavior_t + decoding_bin*config.bin_factor
mask_run, _, _ = selected_RUN.contains( decoded_behavior_t )
decoded_behavior_run = decoded_behavior[ mask_run ]
decoded_behavior_run_t = decoded_behavior_t[ mask_run ]

# COMPUTE DECODING ERROR
true_behavior_run = sp.interpolate.interp1d( vt_position.time[sel_on_maze],\
    linear_d[sel_on_maze], kind='linear' ) ( decoded_behavior_run_t )       
assert( len(true_behavior_run) == len(decoded_behavior_run) )
n_test_bins = len(decoded_behavior_run)

errors = np.array([np.linalg.norm(pred_i - true_behav_i) \
    for pred_i, true_behav_i in zip(decoded_behavior_run, true_behavior_run)])
error = np.nanmedian(errors)

arm_mask = np.zeros( (len(true_behavior_run), n_tracks), dtype='bool' )
for t in range(n_tracks):
    arm_mask[:,t] = np.logical_and( true_behavior_run>=all_boundaries[t],\
            true_behavior_run<all_boundaries[t+1]) 

print "\nMedian error:\t{0:.3f} cm" .format(error/pixel_to_cm)
print "\nN test bins: {0}" .format( n_test_bins )
print "\n(Grid element size : {0:.2f} cm)" .format( config.grid_element_size / pixel_to_cm )

for a in range(n_tracks):
    print "\nMedian error (arm {0}):\t{1:.3f} cm" .format(a+1, np.median(errors[arm_mask[:,a]])/pixel_to_cm)

if config.save_errors:
    np.save( open( config.path_to_errors, 'w'), errors/pixel_to_cm )
    print "\nDecoding errors (all arms) saved in " + config.path_to_errors
    for a in range(n_tracks):
        path = config.path_to_errors[:-4] + '_arm' + str(a+1) + config.path_to_errors[-4:]
        np.save( open( path, 'w'), errors[arm_mask[:,a]]/pixel_to_cm )
        print "\nDecoding errors arm {0} saved in {1}". format( a+1, path )
    

### PLOTTING ### 
# plot decoded and true position during test run epochs
time_points = np.arange(0, len(true_behavior_run)) * decoding_bin

if config.plot_dec_behav:
    
    fig1 = plt.figure(1, figsize=(40,8))
    cmap = tools.get_cmap( n_tracks+1, .3 )
    grid = np.load( join( config.path_to_environment, "grid.npy") )
    content_nodes = np.load( join( config.path_to_environment, "content_nodes.npy" ) )
    fig1.clf()
    plt.plot(time_points, true_behavior_run, color='black', linewidth=2)
    plt.plot(time_points, decoded_behavior_run, color='red', linewidth=2)
    plt.figtext(0.25, 0.96, "true position", fontsize='xx-large',\
        color='black', ha ='right', weight='bold')
    plt.figtext(0.50, 0.96, "decoded position", fontsize='xx-large',\
        color='red', ha ='right', weight='bold')
    plt.yticks(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.ylabel("linear position [pixel]", fontsize='xx-large')
    plt.xlabel("time [s]", fontsize='xx-large')
    for i, nodes in enumerate( content_nodes ):
        plt.fill_between( time_points, grid[nodes[0]], grid[nodes[1]],\
            color=cmap(i) )
    
if config.plot_dec_error:
    
    fig2 = plt.figure(2, figsize=(20,4))
    import seaborn
    seaborn.set_context('poster')
    seaborn.kdeplot( (errors / pixel_to_cm), shade=True, bw="2", color="gray")
    xt = np.array([0, 10, 20, 30, 40])
    plt.xticks( (xt), [str(k) for k in xt], fontsize='x-large',  weight='bold' )
    plt.xlabel("decoding error [cm]", fontsize='x-large',  weight='bold')
    plt.ylabel("probability distribution")
    plt.vlines(error / pixel_to_cm, 0, 0.1, colors='r', linewidth=5 )
    plt.xlim([0, 75]) 
    
    fig3 = plt.figure(4)
    seaborn.reset_orig()
    bp = plt.boxplot(errors / pixel_to_cm)
    for box in bp['boxes']:
        # change outline color
        box.set( linewidth=5)
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set( linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set( linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set( linewidth=5 )
    plt.ylim(ymax = 50)
    plt.ylabel("decoding error [cm]", fontsize='xx-large',  weight='bold')
    plt.xticks([""])
    plt.yticks(fontsize="xx-large", weight = 'bold')
    
    fig4 = plt.figure(5) # plot the cumulative function
    values, base = np.histogram(errors / pixel_to_cm, bins=400)
    cumulative = np.double(np.cumsum(values))
    plt.plot(base[:-1], cumulative/cumulative.max(), c='blue')
    plt.figtext(0.45, 0.96, "cumulative error", fontsize='xx-large',\
        color='black', ha ='center', weight='bold')
    plt.xlabel("error [cm]", weight='bold')



plt.show()