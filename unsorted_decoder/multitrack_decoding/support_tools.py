# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:40:15 2015

@author: davide
"""
import numpy as np
import scipy as sp
from os import path, unlink, listdir
from matplotlib import pyplot as plt

from fklab.io import neuralynx as nlx
import fklab.segments as seg
import fklab.signals.kernelsmoothing as ks
from fklab.geometry import shapes
import fklab.utilities.yaml as yaml

def dummy(): return


def extract_spike_amplitudes(path_to_spikes, n_tt, min_ampl=0, perc=50):
    print ("I'm reading the spike files and extracting the spike feaatures ... ")
    spike_features = []
    spf_dim_l = []
    start_time = np.inf
    stop_time = 0
    for tt in range(n_tt):
        fullpath_spike_file = path_to_spikes + "TT" + str(tt+1) + ".ntt"
        if path.isfile(fullpath_spike_file):
            spike_file = nlx.NlxOpen(fullpath_spike_file)
            spike_file.correct_inversion = False
            spf_dim_l.append(spike_file.nchannels)
            spikes = spike_file.readdata(start=spike_file.starttime[0], stop=spike_file.endtime[0])
            n_spikes = np.shape(spikes["waveform"])[0]
            amplitudes_tt = dict()
            amplitudes_tt["value"] = np.zeros((n_spikes, spike_file.nchannels))
            amplitudes_tt["time"] = np.zeros(n_spikes)
            n_sf = 0
            for n in range(n_spikes):
                sf_temp = [np.max(spikes["waveform"][n, :, j]) for j in range(spf_dim_l[-1])]
                # amplitudes could be better computed with quadratic interpolation
                if np.percentile(sf_temp, perc) > min_ampl:
                    amplitudes_tt["value"][n_sf] = sf_temp   
                    amplitudes_tt["time"][n_sf] = spikes["time"][n]
                    n_sf += 1
            amplitudes_tt["value"] = amplitudes_tt["value"][:n_sf]
            amplitudes_tt["time"] = amplitudes_tt["time"][:n_sf]
            spike_features.append(amplitudes_tt)
            print "\nspike features from TT{0} are loaded" .format(tt+1)
            start_time = min(spike_file.starttime[0], start_time)
            stop_time = max(stop_time, spike_file.endtime[0])
    spf_dim = max(spf_dim_l)
    return spike_features, spf_dim, start_time, stop_time


def partition(segmented_data, n_partitions, partition_index=0, fold=0, method='block', coincident=False, keepremainder=False):
    partitions = [i for i in segmented_data.partition(nparts=n_partitions, method=method, keepremainder=False)]
    if n_partitions == 1:
        test = train = partitions[0]
    elif n_partitions == 2:
            if fold > 1:
                raise ValueError("Incorrect fold index")
            test = partitions[fold]
            if coincident:
                train = test
            else:
                train = partitions[1-fold]
    elif n_partitions > 2:
        if partition_index > n_partitions/2:
            raise ValueError("Incorrect partition index")
        if fold > 1:
            raise ValueError("Incorrect fold index")
        train = partitions[partition_index + fold]  
        if coincident:
            test = train            
        else:
            test = partitions[partition_index + 1-fold]
    return train, test
    
    
def attach(X, tt, interpolation_method='linear' ):
    '''
    exposes the encoding points created internally in the decoder
    '''
    a_t = tt[-1, :]
    amp = tt[:-1, :]
    interpolated_x = sp.interpolate.interp1d( X[-1, :], X[0:-1],\
        kind=interpolation_method, axis=1, bounds_error=False )(a_t)
    nan_idx = np.isnan( np.sum(interpolated_x, axis=0) )
    xa = np.vstack( (interpolated_x[:, ~nan_idx], amp[:, ~nan_idx]) )
    xa = np.ascontiguousarray( xa.T )
    return xa
    

def extract_linear( vt_position, maze, max_occlusion_time, max_dist_to_path, max_dist_to_path_occlusions, plot_env=False ):  
    
    # compute distance to linearized path to identify invalid points that are most likely to be due to occlusions
    # (out-of-range points are considered invalid here)
    vt_points = np.array( [vt_position.x, vt_position.y] ).T
    
    ld_raw, dist_to_path1, _, (_, _) = maze.point2path( vt_points )
    mask1 = np.abs(dist_to_path1) > max_dist_to_path_occlusions
    invalid_time_segs = seg.Segment.fromlogical(mask1, x=vt_position.time)
    
    to_be_fixed = invalid_time_segs[invalid_time_segs.duration <= max_occlusion_time]
    to_be_fixed_m, _, _ = to_be_fixed.contains( vt_position.time )
    valid_m = ~to_be_fixed_m
    
    vt_points_fixed = sp.interpolate.interp1d( vt_position.time[valid_m],\
        vt_points[valid_m], kind='linear', axis=0, fill_value="extrapolate" ) ( vt_position.time ) 
    
    # compute distance to linearized path to identify epochs of run
    ld_fixed, dist_to_path2, _, (_, _) = maze.point2path( vt_points_fixed )
    mask2 = np.abs( dist_to_path2 ) > max_dist_to_path
    segs = seg.Segment.fromlogical( ~mask2, x=vt_position.time ) 
    linear_d, _, point_on_path, (_, _) = maze.point2path( vt_points_fixed )
    sel_on_maze = segs.contains(vt_position.time)[0]
    
    # plot fixed and linearized points
    if plot_env:
        plt.clf()
        plt.gcf().set_size_inches(9, 9)
        plt.plot(vt_points[:, 0], vt_points[:, 1], '.', color='gray')
        plt.plot(vt_points_fixed[sel_on_maze][:, 0],\
            vt_points_fixed[sel_on_maze][:, 1], '.', color='blue')
        plt.plot(point_on_path[:, 0], point_on_path[:, 1], '.', color='red')
        
    return linear_d, sel_on_maze


def extract_run( behav, behav_t, velocity_bw, min_speed, pixel_to_cm, sel_on_maze ):  
    '''
        returns run segments using the linearized position
    ''' 
    
    dt = np.median( np.diff( behav_t ) )
    if len( np.shape( behav )) == 2:
        velocity = np.gradient( behav[0, :] + 1j*behav[1, :], dt )
    else:
        velocity = np.gradient( behav, dt) / pixel_to_cm
    speed = np.abs( ks.Smoother( kernel = \
        ks.GaussianKernel( bandwidth=velocity_bw ) )( velocity, delta=dt ) )
    sel_run = np.logical_and( speed > min_speed, sel_on_maze )    
        
    return seg.Segment.fromlogical( sel_run, x=behav_t ), speed
    

def extract_stop( behav, behav_t, velocity_bw, max_speed, pixel_to_cm, sel_on_maze ):  
    '''
        returns stop segments using the linearized position
    ''' 
    
    dt = np.median( np.diff( behav_t ) )
    velocity = np.gradient( behav, dt) / pixel_to_cm
    speed = np.abs( ks.Smoother( kernel = \
        ks.GaussianKernel( bandwidth=velocity_bw ) )( velocity, delta=dt ) )
    sel_stop = np.logical_and( speed < max_speed, sel_on_maze )    
        
    return seg.Segment.fromlogical( sel_stop, x=behav_t )  
    
    
def all_tracks_behavior( linear_d_track, sel_on_track, distance=1000, out_of_track_value=0 ):   
    '''
        combines linear tracks behavior
    '''
    
    n_tracks = len(linear_d_track)
    assert( n_tracks == len(sel_on_track) )
    
    linear_d = np.ones_like( linear_d_track[0] ) * out_of_track_value
    sel_on_maze = np.zeros_like( linear_d, dtype='bool' )

    for i, (ld, sel) in enumerate(zip(linear_d_track, sel_on_track)):
        linear_d[sel] = ld[sel] + i*distance
        sel_on_maze = np.logical_or( sel_on_maze, sel )
    
    return linear_d, sel_on_maze
    
    
def combine_epochs( segs_tracks ):
    '''
        combine segmented epochs of each linear track into a single set of 
        sorted segments
    '''
    result = seg.Segment()
    for s in segs_tracks:
        result = result + s
    result.isort()
    return result


def set_multi_track_maze( path_to_environmentfile, arm_length_cm ):
    '''
        combines linear tracks behavior to obtain the behavior over a unique 
        linearized distance,
        and combines the individual run epochs
    '''
    with open( path_to_environmentfile ) as f:
        environment = yaml.load( f )
    env_name = environment.keys()[0]
    track_names = environment[env_name]['shapes'].keys()
    arms = [ environment[env_name]['shapes'][track]['shape'].vertices for track in track_names ]
    arm_lengths_pixel = [ sp.spatial.distance.euclidean( arm[0], arm[1] ) for arm in arms ]
    pixel_to_cm = np.mean( arm_lengths_pixel ) / arm_length_cm
    maze = [ shapes.polyline( np.array( [arm[0], arm[1]] ) ) for arm in arms ]
    return maze, pixel_to_cm


def clean_directory( folder_path, clean_subdirs=False ):
    '''
        remove all files from a given folder
    '''
    import shutil
    for the_file in listdir( folder_path ):
        file_path = path.join( folder_path, the_file)
        try:
            if path.isfile(file_path):
                unlink(file_path)
            elif path.isdir(file_path) and clean_subdirs: shutil.rmtree(file_path)
        except Exception, e:
            print e


def int_to_roman(input):
   """
   Convert an integer to Roman numerals.

   Examples:
   >>> int_to_roman(0)
   Traceback (most recent call last):
   ValueError: Argument must be between 1 and 3999

   >>> int_to_roman(-1)
   Traceback (most recent call last):
   ValueError: Argument must be between 1 and 3999

   >>> int_to_roman(1.5)
   Traceback (most recent call last):
   TypeError: expected integer, got <type 'float'>

   >>> for i in range(1, 21): print int_to_roman(i)
   ...
   I
   II
   III
   IV
   V
   VI
   VII
   VIII
   IX
   X
   XI
   XII
   XIII
   XIV
   XV
   XVI
   XVII
   XVIII
   XIX
   XX
   >>> print int_to_roman(2000)
   MM
   >>> print int_to_roman(1999)
   MCMXCIX
   """
   if type(input) != type(1):
      raise TypeError, "expected integer, got %s" % type(input)
   if not 0 < input < 4000:
      raise ValueError, "Argument must be between 1 and 3999"   
   ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
   nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   result = ""
   for i in range(len(ints)):
      count = int(input / ints[i])
      result += nums[i] * count
      input -= ints[i] * count
   return result


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
    
    
def get_cmap(N, alpha):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    '''
    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index, alpha=alpha)
        
    return map_index_to_rgb_color


class VT_Position( ):
    '''
    Provide the same interface to VT data of a Neuralynx file
    '''    
    def __init__( self, falcon_vt_data=None ):
        if falcon_vt_data is not None:
            self.x = falcon_vt_data['xy'][:,0]
            self.y = falcon_vt_data['xy'][:,1]
            self.time = falcon_vt_data['timestamp']/1e6
        else:
            self.x = None
            self.y = None
            self.time = None
        
        
def create_compact_list( indices ):
    '''
        from (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14) to 
            (1-4, 6-11, 13-14)
    '''
    output = '('
    previous_ind = indices[0]
    acc = 0
    initial = 99
    prnt = False
    
    print "{0} elements in the list" .format(len(indices))

    for ind in indices[1:]+[100]:  
        if (ind-previous_ind) == 1:
            acc += 1
            if acc == 1:
                initial = previous_ind
        else:
            prnt = True
        
        if prnt:
            if acc == 0:
                output += str(previous_ind) + ', '
            else:
                output += str(initial) + '-' + str(previous_ind) + ', '
                acc = 0
            initial == previous_ind
            prnt = False
        previous_ind = ind
    
    return output[:-2] + ')'
  
    
def compute_mua( spike_features, bin_size=0.001, smooth_bw=0.015, discard=() ):
    '''
    Compute MUA from detected spikes on multi-tetrode array as smoothed histogram
    of a selected combination of tetrodes
    bin_size_raw_mua in sec
    smooth_mua_bw in sec
    '''

    start_time_spikes = np.inf
    end_time_spikes = 0
    all_spike_times = np.empty(0)
    n_tot_spikes = np.uint64(0)
    
    for i,tt in enumerate(spike_features):
        
        if i in discard:
            continue
        if tt["time"][0] < start_time_spikes:
            start_time_spikes = tt["time"][0]
        if tt["time"][-1] > end_time_spikes:
            end_time_spikes = tt["time"][-1]
        all_spike_times = np.concatenate( (all_spike_times, tt['time']) )
        n_tot_spikes += np.uint( len(tt['time']) )
        
    all_spike_times.sort()
    assert (len(all_spike_times) == n_tot_spikes)
    
    tot_duration = end_time_spikes - start_time_spikes
    nsteps_mua = tot_duration / bin_size
    
    histogram, mua_t = np.histogram( all_spike_times,\
        bins=np.linspace( start_time_spikes, end_time_spikes, num=nsteps_mua ), density=False )
    mua_t = mua_t[:-1]
    smoothed_histogram = ks.Smoother( kernel=ks.GaussianKernel( bandwidth=smooth_bw )) \
        ( histogram, delta=smooth_bw )
    mua = smoothed_histogram / bin_size
    
    return mua, mua_t

