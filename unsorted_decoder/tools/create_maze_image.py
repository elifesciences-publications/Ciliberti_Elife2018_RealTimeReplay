# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:05:55 2017

@author: fklab
"""

from fklab.ui.localize.localize_tools import get_nlx_video_time,\
    extract_video_image
import os
import numpy as np
import create_maze_image_config as config

dirname = os.path.dirname( config.path_to_room_video )
output_file = os.path.join( dirname, "room.jpg" )

with open( os.path.join( dirname, "VT1.smi"), 'r' ) as f:
    s = f.readlines(20)[60]

start = s.find("ENUSCC") + len('ENUSCC') + 1
stop = s.find("SYNC", 8) - 2 
timestamp = np.uint64(s[start:stop])

video_t, video = get_nlx_video_time( dirname, timestamp*1e-6 )

if video_t is None: 
    raise RuntimeError("Room image could not be generated. (timestamp = {0})" .format(timestamp))

if os.path.isfile( output_file ):
    os.remove( output_file )

ret = extract_video_image( config.path_to_room_video, video_t,\
    output_file )

print "\nRoom image saved in " + output_file
    
