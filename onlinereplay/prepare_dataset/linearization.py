# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:50:00 2016

@author: davide
"""

import numpy as np
from multitrack_decoding.support_tools import set_multi_track_maze,\
    extract_linear, all_tracks_behavior, cartesian

import create_linerization_matrix_config as config
    
RESOLUTION = np.array([720, 576])

class VTPoints():
    
    def __init__( self, x, y, time ):
        self.x = x
        self.y = y
        self.time = time


def create_linearization_matrix( path_to_environmentfile, path_length_cm,\
extra_distance, out_of_track_value ):
    """
    Create linearization matrix for analysis multi-track environments.
    All tracks are assumed to be of the same length
    
    Params:
    -------
    path_to_environmentfile : string
    path_length_cm : float, length of a single track
    extra_distance : int, number of extra pixels to be added for decoding purposes
    between two consectuive arms
    
    Returns:
    --------
    linearization_matrix: 2d array of 720 x 576 elements
    pixel_to_cm : scalar, conversion factor of the videocamera
    """

    vt_table = cartesian( [np.arange(RESOLUTION[0]), np.arange(RESOLUTION[1]) ] )
    vt_points = VTPoints( vt_table[:, 0], vt_table[:, 1], np.arange( RESOLUTION.prod() ) )
    
    maze, pixel_to_cm = set_multi_track_maze( path_to_environmentfile,\
        path_length_cm )
    
    linear_d_track = []
    sel_on_track = []
    for track in maze:
        ld, sel = extract_linear( vt_points, track, 0, config.max_dist_to_path, np.inf, True )      
        linear_d_track.append( ld )
        sel_on_track.append( sel )
    
    linear_d, sel_on_maze = all_tracks_behavior( linear_d_track, sel_on_track,\
        distance=extra_distance, out_of_track_value=out_of_track_value )
    
    return linear_d.reshape( RESOLUTION ), pixel_to_cm
