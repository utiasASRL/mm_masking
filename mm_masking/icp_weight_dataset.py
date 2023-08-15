import cv2
import torch
import numpy as np
import os.path as osp
import os
from pyboreas.utils.odometry import read_traj_file2, read_traj_file_gt2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pylgmath import se3op, Transformation
from radar_utils import load_radar, cfar_mask, extract_pc, load_pc_from_file, radar_cartesian_to_polar, radar_polar_to_cartesian_diff
from dICP.ICP import ICP
from pyboreas.utils.utils import (
    SE3Tose3,
    get_closest_index,
    get_inverse_tf,
    rotToRollPitchYaw,
)
import vtr_pose_graph
from vtr_pose_graph.graph_factory import Rosbag2GraphFactory
import vtr_pose_graph.graph_utils as g_utils
from vtr_pose_graph.graph_iterators import TemporalIterator
from utils.extract_graph import extract_points_and_map
import time
import pandas as pd

class ICPWeightDataset():

    def __init__(self, gt_data_dir, pc_dir, radar_dir, loc_pairs,
                 map_sensor='lidar', loc_sensor='radar',
                 random=False, num_samples=-1, verbose=False,
                 float_type=torch.float64, use_gt=False, gt_eye=True, pos_std=1.0, rot_std=0.1,
                 a_thresh=1.0, b_thresh=0.09):
        self.loc_pairs = loc_pairs
        self.float_type = float_type
        self.map_sensor = map_sensor
        self.loc_sensor = loc_sensor
        self.gt_eye = gt_eye

        if not random:
            np.random.seed(99)
            torch.manual_seed(99)

        # Assemble paths
        if map_sensor == 'lidar' and loc_sensor == 'radar':
            sensor_dir_name = 'radar_lidar'
            self.msg_prefix = 'radar_'
        elif map_sensor == 'radar' and loc_sensor == 'radar':
            sensor_dir_name = 'radar'
            self.msg_prefix = ''
        else:
            raise ValueError("Invalid sensor combination")

        data_dir = '../data'
        dataset_dir = osp.join(data_dir, 'vtr_data')
        vtr_result_dir = osp.join(data_dir, 'vtr_results')
        polar_res = 0.0596
        network_input_type = 'cartesian'

        self.v_id_vector = None
        self.graph_id_vector = None
        self.T_loc_gt = None
        self.T_loc_init = None
        self.T_map_sensor_robot = []
        self.graph_list = []
        self.loc_radar_path_list = []
        self.loc_cfar_path_list = []
        # Need to save max loc and map pointcloud size for padding for batch assembly
        self.max_loc_pts = 0
        self.max_map_pts = 0

        for pair_idx, pair in enumerate(loc_pairs):
            map_seq = pair[0]
            loc_seq = pair[1]

            gt_map_poses, gt_map_times = read_traj_file_gt2(osp.join(dataset_dir, map_seq, "applanix", map_sensor + "_poses.csv"), dim=2)
            gt_loc_poses, gt_loc_times = read_traj_file_gt2(osp.join(dataset_dir, loc_seq, "applanix", loc_sensor + "_poses.csv"), dim=2)
            graph_dir = osp.join(vtr_result_dir, sensor_dir_name, map_seq, loc_seq, 'graph')
            factory = Rosbag2GraphFactory(graph_dir)

            pair_graph = factory.buildGraph()
            self.graph_list.append(pair_graph)
            print(f"Graph {pair_graph} has {pair_graph.number_of_vertices} vertices and {pair_graph.number_of_edges} edges")
            
            v_start = pair_graph.get_vertex((1,0))

            # Save transform from map sensor to robot
            # This is needed because the map pointcloud is saved in robot frame,
            # but ground truth is between map sensor and loc sensor.
            # This transform is constant for a given map sequence
            yfwd2xfwd = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            T_applanix_lidar = np.loadtxt(osp.join(dataset_dir, map_seq, 'calib', 'T_applanix_lidar.txt'))
            if map_sensor == 'radar':
                T_radar_lidar = np.loadtxt(osp.join(dataset_dir, map_seq, 'calib', 'T_radar_lidar.txt'))
                T_robot_map_sensor = yfwd2xfwd @ T_applanix_lidar @ get_inverse_tf(T_radar_lidar)
            elif map_sensor == 'lidar':
                T_robot_map_sensor = yfwd2xfwd @ T_applanix_lidar
            T_map_sensor_robot = torch.from_numpy(get_inverse_tf(T_robot_map_sensor)).type(float_type)
            self.T_map_sensor_robot.append(T_map_sensor_robot)

            # Check if result directory contains a metadata file
            # If not, create one
            metadata_path = osp.join(vtr_result_dir, sensor_dir_name, map_seq, loc_seq, 'metadata.csv')
            if not osp.exists(metadata_path):
                df_data = {'complete' : 0, 'up_to_idx': -1, 'max_loc': -1, 'max_map': -1}
                df = pd.DataFrame(df_data, index=[0])
                df.to_csv(metadata_path, index=False)

            # Load in the metadata file to see if we need to extract max points during
            # data loading. 
            pair_df = pd.read_csv(metadata_path)
            # If we have sufficient metadata about what we wish to extract,
            # don't bother extracting more
            extract_pcs_metadata = True
            if (pair_df['complete'][0] == 1 or (pair_df['up_to_idx'][0] >= num_samples and num_samples>0)):
                extract_pcs_metadata = False
                # Check if new max is reached
                if pair_df['max_loc'][0] > self.max_loc_pts:
                    self.max_loc_pts = pair_df['max_loc'][0]
                if pair_df['max_map'][0] > self.max_map_pts:
                    self.max_map_pts = pair_df['max_map'][0]
            print(extract_pcs_metadata)
            ii = -1
            local_max_loc_pts = 0
            local_max_map_pts = 0
            for loc_v, e in TemporalIterator(v_start):
                ii += 1
                # Check if vertex is valid
                if e.from_id == vtr_pose_graph.INVALID_ID:
                    continue
                
                # Extract vertex info
                map_v = g_utils.get_closest_teach_vertex(loc_v)

                # Extract timestamps
                loc_stamp = int(loc_v.stamp * 1e-3)
                map_stamp = int(map_v.stamp * 1e-3)

                # Ensure radar image exists
                loc_radar_path = osp.join(dataset_dir, loc_seq, 'radar', str(loc_stamp) + '.png')
                if not osp.exists(loc_radar_path):
                    continue
                self.loc_radar_path_list.append(loc_radar_path)

                # Ensure CFAR of image exists, if it does not, create one
                # This is done to speed up training so that CFAR image does not need to be created every time
                cfar_dir = osp.join(data_dir, 'cfar', loc_seq, network_input_type, str(a_thresh) + '_' + str(b_thresh))
                if not osp.exists(cfar_dir):
                    os.makedirs(cfar_dir)
                loc_cfar_path = osp.join(cfar_dir, str(loc_stamp) + '.png')
                if not osp.exists(loc_cfar_path):
                    loc_radar_img = cv2.imread(loc_radar_path, cv2.IMREAD_GRAYSCALE)
                    fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
                    fft_data = torch.tensor(fft_data, dtype=self.float_type).unsqueeze(0)
                    azimuths = torch.tensor(azimuths, dtype=self.float_type).unsqueeze(0)
                    az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type).unsqueeze(0)
                    fft_cfar = cfar_mask(fft_data, polar_res, a_thresh=a_thresh, b_thresh=b_thresh, diff=False)

                    # Save CFAR image
                    if network_input_type == 'cartesian':
                        fft_cfar = radar_polar_to_cartesian_diff(fft_cfar, azimuths, polar_res)
                    cv2.imwrite(loc_cfar_path, 255*fft_cfar.squeeze(0).numpy())

                self.loc_cfar_path_list.append(loc_cfar_path)

                # Check that timestamps are matching to gt poses
                assert loc_stamp == gt_loc_times[ii], "query: {}".format(loc_stamp)
                closest_map_t = get_closest_index(map_stamp, gt_map_times)
                assert map_stamp == gt_map_times[closest_map_t], "query: {}".format(map_stamp)
                # Extract gt map pose
                gt_map_pose_idx = gt_map_poses[closest_map_t]
                gt_T_s2_s1 = get_inverse_tf(gt_loc_poses[ii]) @ gt_map_pose_idx

                # Save ground truth localization to map pose
                T_gt_idx = torch.tensor(gt_T_s2_s1, dtype=float_type)

                # Now that we have ground truth, we can filter the map points to know max point size
                # We only do filtering for lidar and only if we dont already have
                # pointcloud metadata. This is done to speed up data loading
                if extract_pcs_metadata:
                    curr_raw_pts, curr_filt_pts, map_pts, map_norms, loc_stamp, map_stamp = extract_points_and_map(pair_graph, loc_v, msg_prefix=self.msg_prefix)
                    assert curr_raw_pts.shape == curr_filt_pts.shape, 'Raw and filtered pointclouds dont match!'

                    map_pts_sensor_frame = (T_map_sensor_robot[:3,:3] @ map_pts.T + T_map_sensor_robot[:3, 3:4]).T
                    map_norms_sensor_frame = (T_map_sensor_robot[:3,:3] @ map_norms.T).T
                    map_pts, map_norms = self.filter_map(map_pts_sensor_frame, map_norms_sensor_frame, T_gt_idx)
                    
                    if curr_raw_pts.shape[0] > local_max_loc_pts:
                        local_max_loc_pts = curr_raw_pts.shape[0]
                    if map_pts.shape[0] > local_max_map_pts:
                        local_max_map_pts = map_pts.shape[0]

                # Also, generate random perturbation to ground truth pose
                # The map pointcloud is transformed into the scan frame using T_gt
                # T_init is the initial guess that is offset from T_gt that the ICP
                # needs to "unlearn" to get to identity
                if use_gt:
                    if gt_eye:
                        T_init_idx = np.eye(4)
                    else:
                        T_init_idx = gt_T_s2_s1
                else:
                    xi_rand = torch.rand((6,1), dtype=float_type)
                    # Zero out z, pitch, and roll
                    xi_rand[2:5] = 0.0
                    # Scale x and y
                    xi_rand[0:2] = pos_std*xi_rand[0:2]
                    # Scale yaw
                    xi_rand[5] = rot_std*xi_rand[5]
                    T_rand = Transformation(xi_ab=xi_rand)
                    if gt_eye:
                        T_init_idx = T_rand.matrix() # @ identity
                    else:
                        T_init_idx = T_rand.matrix() @ gt_T_s2_s1
                T_init_idx = torch.tensor(T_init_idx, dtype=float_type)

                # Stack data for more efficient storage and retrieval
                if self.v_id_vector is None:
                    self.v_id_vector = np.array([loc_v.id])
                    self.graph_id_vector = np.array([pair_idx])
                    self.T_loc_gt = T_gt_idx.unsqueeze(0)
                    self.T_loc_init = T_init_idx.unsqueeze(0)
                else:
                    self.v_id_vector = np.append(self.v_id_vector, loc_v.id)
                    self.graph_id_vector = np.append(self.graph_id_vector, pair_idx)
                    self.T_loc_gt = torch.cat((self.T_loc_gt, T_gt_idx.unsqueeze(0)), dim=0)
                    self.T_loc_init = torch.cat((self.T_loc_init, T_init_idx.unsqueeze(0)), dim=0)

                if (ii % 100) == 0:
                    print(str(ii) + " samples generated")

                if num_samples > 0 and self.v_id_vector.shape[0] >= num_samples:
                    break
            
            # Update metadata if it was collected
            if extract_pcs_metadata:
                # Save metadata
                meta_complete = (num_samples == -1)
                df_data = {'complete' : meta_complete, 'up_to_idx': ii, 'max_loc': local_max_loc_pts, 'max_map': local_max_map_pts}
                df = pd.DataFrame(df_data, index=[0])
                df.to_csv(metadata_path, index=False)
                
                # Overwrite max point sizes if they are larger
                if local_max_loc_pts > self.max_loc_pts:
                    self.max_loc_pts = local_max_loc_pts
                if map_pts.shape[0] > self.max_map_pts:
                    self.max_map_pts = map_pts.shape[0]

        # Assert that the number of all elements are the same
        assert self.v_id_vector.shape[0] == self.graph_id_vector.shape[0] == self.T_loc_gt.shape[0] \
            == self.T_loc_init.shape[0] == len(self.loc_radar_path_list) == len(self.loc_cfar_path_list)

    def __len__(self):
        return self.v_id_vector.shape[0]

    def __getitem__(self, index):
        # Load in initial guess
        T_init = self.T_loc_init[index]

        # Load in ground truth localization to map pose
        T_ml_gt = self.T_loc_gt[index]

        # Load in pointclouds and timestamps
        scan_pc_raw, scan_pc_filt, map_pc, loc_stamp, map_stamp = self.load_graph_data(index, T_ml_gt)
        assert scan_pc_raw.shape == scan_pc_filt.shape, 'Raw and filtered pointclouds dont match!'
        """
        map_pts_curr_frame = (T_ml_gt[:3,:3] @ map_pc[:,:3].T + T_ml_gt[:3,3].unsqueeze(1)).T
        plt.figure(figsize=(15,15))
        plt.scatter(map_pts_curr_frame[:,0], map_pts_curr_frame[:,1], s=1.0, c='red')
        #plt.scatter(map_pts[:,0], map_pts[:,1], s=1.0, c='red')
        plt.scatter(scan_pc[:,0], scan_pc[:,1], s=0.5, c='blue')
        # plt.scatter(scan_pc[:,0], scan_pc[:,1], s=0.5, c='green')
        #plt.scatter(curr_pts_map_frame[:,0], curr_pts_map_frame[:,1], s=0.5, c='green')
        plt.ylim([-80, 80])
        plt.xlim([-80, 80])
        plt.savefig('align.png')
        plt.close()
        adfsdsfd
        """

        # Load in fft data
        loc_radar_img = cv2.imread(self.loc_radar_path_list[index], cv2.IMREAD_GRAYSCALE)
        fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
        fft_data = torch.tensor(fft_data, dtype=self.float_type)
        azimuths = torch.tensor(azimuths, dtype=self.float_type)
        az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type)
        fft_cfar = cv2.imread(self.loc_cfar_path_list[index], cv2.IMREAD_GRAYSCALE)
        fft_cfar = torch.tensor(fft_cfar, dtype=self.float_type)/255.0

        loc_data = {'raw_pc': scan_pc_raw, 'filtered_pc': scan_pc_filt, 'timestamp' : loc_stamp,
                    'fft_data' : fft_data, 'azimuths' : azimuths, 'az_timestamps' : az_timestamps,
                    'fft_cfar' : fft_cfar}
        map_data = {'pc': map_pc, 'timestamp' : map_stamp}
        T_data = {'T_ml_init' : T_init, 'T_ml_gt' : T_ml_gt}

        return {'loc_data': loc_data, 'map_data': map_data, 'transforms': T_data}
    
    def load_graph_data(self, idx, T_ml_gt):
        v_id = self.v_id_vector[idx].item() # Need .item() as v_id must be int, not np.int32/64
        graph_id = self.graph_id_vector[idx]
        pair_graph = self.graph_list[graph_id]
        vertex = pair_graph.get_vertex(v_id)
        curr_raw_pts, curr_filt_pts, map_pts, map_norms, loc_stamp, map_stamp = extract_points_and_map(pair_graph, vertex, msg_prefix=self.msg_prefix)
        
        # Make scan_pc batchable
        curr_raw_pts = torch.from_numpy(curr_raw_pts)
        curr_filt_pts = torch.from_numpy(curr_filt_pts)
        scan_pc_pad = torch.zeros((self.max_loc_pts - curr_raw_pts.shape[0], 3), dtype=self.float_type)
        scan_pc_raw = torch.cat((curr_raw_pts, scan_pc_pad), dim=0)
        scan_pc_filt = torch.cat((curr_filt_pts, scan_pc_pad), dim=0)
        
        # Transform map pointcloud to scan frame
        map_pts = torch.from_numpy(map_pts)
        map_norms = torch.from_numpy(map_norms)
        T_map_sensor_robot_idx = self.T_map_sensor_robot[graph_id]
        map_pts_sensor_frame = (T_map_sensor_robot_idx[:3,:3] @ map_pts.T + T_map_sensor_robot_idx[:3, 3:4]).T
        map_norms_sensor_frame = (T_map_sensor_robot_idx[:3,:3] @ map_norms.T).T

        # Next, filter the map points based on field of view and z-normal value
        # We only do filtering for lidar
        map_pts_sensor_frame, map_norms_sensor_frame = self.filter_map(map_pts_sensor_frame, map_norms_sensor_frame, T_ml_gt, return_aligned=self.gt_eye)

        # Make map_pc batchable
        map_pc_pad = torch.zeros((self.max_map_pts - map_pts_sensor_frame.shape[0], 3), dtype=self.float_type)
        map_pts_pc = torch.cat((map_pts_sensor_frame, map_pc_pad), dim=0)
        map_norms_pc = torch.cat((map_norms_sensor_frame, map_pc_pad), dim=0)
        map_pc = torch.cat((map_pts_pc, map_norms_pc), dim=1)

        return scan_pc_raw, scan_pc_filt, map_pc, loc_stamp, map_stamp

    def filter_map(self, map_pts, map_norms, T_ml_gt, return_aligned=False):
        # Transform map points to loc frame using gt to filter
        map_pts_loc_frame = (T_ml_gt[:3,:3] @ map_pts.T + T_ml_gt[:3, 3:4]).T
        map_norms_loc_frame = (T_ml_gt[:3,:3] @ map_norms.T).T

        # Filter by elevation and z-normal score
        # TODO: Make these parameters
        elevation_threshold = 0.05
        z_normal_threshold = 0.9
        p_in_s = map_pts_loc_frame
        elev = torch.abs(torch.atan2(p_in_s[:,2], torch.sqrt(p_in_s[:,0] * p_in_s[:,0] + p_in_s[:,1] * p_in_s[:,1])))
        z_norm = torch.abs(map_norms_loc_frame[:,2])
        if self.map_sensor == 'lidar':
            valid_pts = (elev <= elevation_threshold) & (z_norm <= z_normal_threshold)
        else:
            valid_pts = torch.ones((map_pts_loc_frame.shape[0],), dtype=torch.bool)
        
        # Extract only valid points
        if return_aligned:
            return map_pts_loc_frame[valid_pts], map_norms_loc_frame[valid_pts]
        else:
            return map_pts[valid_pts], map_norms[valid_pts]