#!/usr/bin/env python3

import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from vlmaps_parser import get_args
import rosbag
from tqdm import tqdm
from PIL import Image as Im
import math

class DataConverter():
    def __init__(self, args):
        self.data_path = args.data_path
        self.pose_path = os.path.join(args.data_path, 'pose')
        self.depth_path = os.path.join(args.data_path, 'depth')
        self.rgb_path = os.path.join(args.data_path, 'rgb')
        self.pose_topic_name = args.pose_topic_name
        self.depth_topic_name = args.depth_topic_name
        self.rgb_topic_name = args.rgb_topic_name
        self.depth_scale_factor = args.depth_scale_factor
        self.sync_time_th = args.sync_time_th
        self.save_interval = args.save_interval
        self.min_timestamp_th = args.min_timestamp_th
        self.bridge = CvBridge()
        self.cnt = 0
        self.bag_cnt = 0
        
        self.poses = []
        self.pose_timestamps = []
        self.depth_images = []
        self.depth_timestamps = []
        self.rgb_images = []
        self.rgb_timestamps = []
        
        assert os.path.isdir(args.bag_path), \
            "There is no directory: {}. Please make the directory and put {} in it.".format(args.bag_path, args.rosbag_name)
        assert os.path.isfile(os.path.join(args.bag_path, args.rosbag_name)), \
            "Please put {} in the following directory: {}.".format(args.rosbag_name, args.bag_path)

        self.bagfile = rosbag.Bag(os.path.join(args.bag_path, args.rosbag_name))
        
        self.check_and_make_directories()
        self.check_and_clean_directories()
        self.extract_and_save_all_data_from_bag_in_once()
    
    def check_and_make_directories(self):
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.depth_path):
            os.mkdir(self.depth_path)
        if not os.path.isdir(self.pose_path):
            os.mkdir(self.pose_path)
        if not os.path.isdir(self.rgb_path):
            os.mkdir(self.rgb_path)
            
    def check_and_clean_directories(self):
        if os.listdir(self.depth_path):
            files = glob.glob(self.depth_path + "/*")
            for f in tqdm(files, desc='Remove previous depth data...') :
                os.remove(f)
        if os.listdir(self.rgb_path):
            files = glob.glob(self.rgb_path + "/*")
            for f in tqdm(files, desc='Remove previous RGB data...') :
                os.remove(f)
        if os.listdir(self.pose_path):
            files = glob.glob(self.pose_path + "/*")
            for f in tqdm(files, desc='Remove previous pose data...') :
                os.remove(f)
    
    def clear_all_data(self):
        self.poses.clear()
        self.pose_timestamps.clear()
        self.depth_images.clear()
        self.depth_timestamps.clear()
        self.rgb_images.clear()
        self.rgb_timestamps.clear()
        
    def extract_depth_from_bag(self, msg, t):
        if hasattr(msg, 'format'):
            if "compressed" in msg.format:
                raise TypeError("This code does not support compressed images.")
        else:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_timestamp = msg.header.stamp if msg._has_header else t
        depth_npy = np.array(depth_img) * self.depth_scale_factor
        self.depth_images.append(depth_npy)
        if len(self.depth_timestamps)!=0 and depth_timestamp.nsecs*1e-9 < self.depth_timestamps[-1]:
            self.depth_timestamps.append(depth_timestamp.nsecs * 1e-9 + math.ceil(self.depth_timestamps[-1] - depth_timestamp.nsecs*1e-9))
        else:
            self.depth_timestamps.append(depth_timestamp.nsecs * 1e-9)
                
    def extract_rgb_from_bag(self, msg, t):
        if hasattr(msg, 'format'):
            if "compressed" in msg.format:
                raise TypeError("This code does not support compressed images.")
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rgb_timestamp = msg.header.stamp if msg._has_header else t
        rgb_npy = np.array(rgb_img)
        self.rgb_images.append(rgb_npy)
        if len(self.rgb_timestamps)!=0 and rgb_timestamp.nsecs*1e-9 < self.rgb_timestamps[-1]:
            self.rgb_timestamps.append(rgb_timestamp.nsecs * 1e-9 + math.ceil(self.rgb_timestamps[-1] - rgb_timestamp.nsecs*1e-9))
        else:
            self.rgb_timestamps.append(rgb_timestamp.nsecs * 1e-9)

    def extract_pose_from_bag(self, msg, t):
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        pose = np.array([pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]) 
        pose_timestamp = msg.header.stamp if msg._has_header else t
        self.poses.append(pose)
        if len(self.pose_timestamps)!=0 and pose_timestamp.nsecs*1e-9 < self.pose_timestamps[-1]:
            self.pose_timestamps.append(pose_timestamp.nsecs * 1e-9 + math.ceil(self.pose_timestamps[-1] - pose_timestamp.nsecs*1e-9))
        else:
            self.pose_timestamps.append(pose_timestamp.nsecs * 1e-9)
                
    def extract_and_save_all_data_from_bag_in_once(self):
        for topic, msg, t in tqdm(self.bagfile.read_messages(), desc="Extracting data from bagfile..."):
            self.bag_cnt += 1
            if topic == self.depth_topic_name:
                self.extract_depth_from_bag(msg, t)
            elif topic == self.rgb_topic_name:
                self.extract_rgb_from_bag(msg, t)
            elif topic == self.pose_topic_name:
                self.extract_pose_from_bag(msg, t)
            
            if self.bag_cnt%self.save_interval == 0:
                self.synchronize_rgb_depth()
                self.synchronize_image_pose()
                self.clear_all_data()
        
        self.synchronize_rgb_depth()
        self.synchronize_image_pose()
        self.clear_all_data()
    
    def save_pose(self, pose):
        pose_data = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6])
        f = open("{0}/{1:06d}.txt".format(self.pose_path, self.cnt), 'w')
        f.write(pose_data)
        f.close()
        
    def save_depth(self, depth_npy):
        np.save('{0}/{1:06d}'.format(self.depth_path, self.cnt), depth_npy)
        
    def save_rgb(self, rgb_npy):
        rgb_png = Im.fromarray(rgb_npy)
        rgb_png.save('{0}/{1:06d}.png'.format(self.rgb_path, self.cnt))
        
    def save_all(self, pose_idx, depth_idx, rgb_idx):
        pose = self.poses[pose_idx]
        depth_npy = self.depth_images[depth_idx]
        rgb_npy = self.rgb_images[rgb_idx]
        self.save_pose(pose)
        self.save_depth(depth_npy)
        self.save_rgb(rgb_npy)
        self.cnt = self.cnt + 1        
     
    def synchronize_rgb_depth(self):
        sync_rgb_images = []
        sync_rgb_timestamps = []
        sync_depth_images = []
        sync_depth_timestamps = []
        prev_timestamp = 0
        
        if len(self.depth_timestamps) > len(self.rgb_timestamps):
            for idx in tqdm(range(len(self.rgb_timestamps)), desc='Synchronize rgb and depth data...'):
                if idx !=0 and np.abs(self.rgb_timestamps[idx] - prev_timestamp) < self.min_timestamp_th:
                    continue
                
                time_diff = np.abs(np.array(self.depth_timestamps) - self.rgb_timestamps[idx])
                min_time_diff = np.min(time_diff)
                min_idx_time_diff = np.argmin(time_diff)
                if min_time_diff > self.sync_time_th:
                    continue
                
                sync_rgb_images.append(self.rgb_images[idx])
                sync_rgb_timestamps.append(self.rgb_timestamps[idx])
                sync_depth_images.append(self.depth_images[min_idx_time_diff])
                sync_depth_timestamps.append(self.depth_timestamps[min_idx_time_diff])
                prev_timestamp = self.rgb_timestamps[idx]
        else:
            for idx in tqdm(range(len(self.depth_timestamps)), desc='Synchronize rgb and depth data...'):
                time_diff = np.abs(np.array(self.rgb_timestamps) - self.depth_timestamps[idx])
                if idx !=0 and np.abs(self.depth_timestamps[idx] - prev_timestamp) < self.min_timestamp_th:
                    continue
                
                min_time_diff = np.min(time_diff)
                min_idx_time_diff = np.argmin(time_diff)
                if min_time_diff > self.sync_time_th:
                    continue
                
                sync_rgb_images.append(self.rgb_images[min_idx_time_diff])
                sync_rgb_timestamps.append(self.rgb_timestamps[min_idx_time_diff])
                sync_depth_images.append(self.depth_images[idx])
                sync_depth_timestamps.append(self.depth_timestamps[idx])
                prev_timestamp = self.depth_timestamps[idx]
                
        self.rgb_images = sync_rgb_images
        self.rgb_timestamps = sync_rgb_timestamps
        self.depth_images = sync_depth_images
        self.depth_timestamps = sync_depth_timestamps
                    
    def synchronize_image_pose(self):
        prev_timestamp = 0
        if len(self.pose_timestamps) > len(self.rgb_timestamps):
            for idx in tqdm(range(len(self.rgb_timestamps)), desc='Synchronize and save all data...'):
                if idx !=0 and np.abs(self.rgb_timestamps[idx] - prev_timestamp) < self.min_timestamp_th:
                    continue
                
                time_diff = np.abs(np.array(self.pose_timestamps) - self.rgb_timestamps[idx])
                min_time_diff = np.min(time_diff)
                min_idx_time_diff = np.argmin(time_diff)
                
                if min_time_diff > self.sync_time_th:
                    continue
                
                self.save_all(pose_idx=min_idx_time_diff, depth_idx=idx, rgb_idx=idx)
                prev_timestamp = self.rgb_timestamps[idx]
                
        else:
            for idx in tqdm(range(len(self.pose_timestamps)), desc='Synchronize and save all data...'):
                if idx !=0 and np.abs(self.pose_timestamps[idx] - prev_timestamp) < self.min_timestamp_th:
                    continue
                
                time_diff = np.abs(np.array(self.rgb_timestamps) - self.pose_timestamps[idx])
                min_time_diff = np.min(time_diff)
                min_idx_time_diff = np.argmin(time_diff)
                
                if min_time_diff > self.sync_time_th:
                    continue
                
                self.save_all(pose_idx=idx, depth_idx=min_idx_time_diff, rgb_idx=min_idx_time_diff)
                prev_timestamp = self.pose_timestamps[idx]
                
if __name__ == '__main__':
    args = get_args()
    pose_converter = DataConverter(args)