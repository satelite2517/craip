#!/usr/bin/env python

import os
import argparse

vlmaps_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_args():
    
    parser = argparse.ArgumentParser(description="configs for converters")
    
    parser.add_argument('--data_path', type=str, default='{0}/data'.format(vlmaps_directory), metavar='PATH', help='the path for data directory')
    parser.add_argument('--bag_path', type=str, default='{0}/bags'.format(vlmaps_directory), metavar='PATH', help='the path for bagfile directory')
    parser.add_argument('--pose_topic_name', type=str, default='/odom', metavar='NAME', help='the topic name corresponding to pose')
    parser.add_argument('--depth_topic_name', type=str, default='/d435/depth/image_raw', metavar='NAME', help='the topic name corresponding to depth')
    parser.add_argument('--rgb_topic_name', type=str, default='/d435/color/image_raw', metavar='NAME', help='the topic name corresponding to rgb')
    parser.add_argument('--rosbag_name', type=str, default='gazebo.bag', help='the name for bagfile')
    parser.add_argument('--sync_time_th', type=float, default=0.005, help='threshold to cut out values which have sync delay greater than threshold (unit: seconds)')
    parser.add_argument('--depth_scale_factor', type=float, default=0.001, help='depth image scale factor')
    parser.add_argument('--save_interval', type=float, default=5000, help='synchronization and saving interval while extracting bagfile')
    parser.add_argument('--min_timestamp_th', type=float, default=0.1, help='minimum interval between timestamps for ignoring redundant data')
    
    args = parser.parse_args()
    return args