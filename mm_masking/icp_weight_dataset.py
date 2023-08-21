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
from radar_utils import load_radar, cfar_mask, extract_pc, load_pc_from_file, radar_cartesian_to_polar, radar_polar_to_cartesian_diff, extract_bev_from_pts
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

    def __init__(self, loc_pairs, params=None, dataset_type='train'):

        # Load in params
        map_sensor=params["map_sensor"]
        loc_sensor=params["loc_sensor"]
        random=params["random"]
        if dataset_type == 'train':
            num_samples=params["num_train"]
            augment_factor = params["augment_factor"]
        else:
            num_samples=params["num_val"]
            augment_factor = 1
        
        float_type=params["float_type"]
        use_gt=params["use_gt"]
        gt_eye=params["gt_eye"]
        pos_std=params["pos_std"]
        rot_std=params["rot_std"]
        a_thresh=params["a_thresh"]
        b_thresh=params["b_thresh"]
        network_input_type = params["network_input_type"]

        # Set up available augmentations
        # flip is a symmetric flip about the x-axis of the fft data
        # rot is the number of 90 degree rotations to apply to the fft data
        self.augmentations = [
            {'flip': False, 'rot': 0},  # Baseline (no augmentation)
            {'flip': False, 'rot': 1},
            {'flip': False, 'rot': 3},
            {'flip': True, 'rot': 0},
            {'flip': True, 'rot': 1},
            {'flip': True, 'rot': 3},
        ]

        self.loc_pairs = loc_pairs
        self.float_type = float_type
        self.map_sensor = map_sensor
        self.loc_sensor = loc_sensor
        self.gt_eye = gt_eye
        self.network_input_type = network_input_type

        # Load in ICP to get target padding value
        config_path = '../external/dICP/config/dICP_config.yaml'
        temp_ICP_alg = ICP(icp_type='pt2pt', config_path=config_path,)
        self.target_pad_val = temp_ICP_alg.target_pad_val

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
        
        self.v_id_vector = None
        self.graph_id_vector = None
        self.augment_id_vector = None
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
            print("Loading from metadata: " + str(extract_pcs_metadata))
            local_max_loc_pts = 0
            local_max_map_pts = 0
            for ii, (loc_v, e) in enumerate(TemporalIterator(v_start)):
                # Check if vertex is valid
                if e.from_id == vtr_pose_graph.INVALID_ID:
                    continue
                
                # Extract vertex info
                map_v = g_utils.get_closest_teach_vertex(loc_v)
                map_ptr = map_v.get_data("pointmap_ptr")
                map_v = pair_graph.get_vertex(map_ptr.map_vid)

                # Extract timestamps
                loc_stamp = int(loc_v.stamp * 1e-3)
                map_stamp = int(map_v.stamp * 1e-3)

                # Ensure radar image exists
                loc_radar_fft_path = osp.join(dataset_dir, loc_seq, 'radar', str(loc_stamp) + '.png')
                if network_input_type == 'cartesian':
                    loc_radar_path = osp.join(dataset_dir, loc_seq, 'radar', 'cart', str(loc_stamp) + '.png')
                else:
                    loc_radar_path = loc_radar_fft_path
                if not osp.exists(loc_radar_path):
                    continue

                # Ensure CFAR of image exists, if it does not, create one
                # This is done to speed up training so that CFAR image does not need to be created every time
                cfar_dir = osp.join(data_dir, 'cfar', loc_seq, network_input_type, str(a_thresh) + '_' + str(b_thresh))
                if not osp.exists(cfar_dir):
                    os.makedirs(cfar_dir)
                loc_cfar_path = osp.join(cfar_dir, str(loc_stamp) + '.png')
                if not osp.exists(loc_cfar_path):
                    loc_radar_img = cv2.imread(loc_radar_fft_path, cv2.IMREAD_GRAYSCALE)
                    fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
                    fft_data = torch.tensor(fft_data, dtype=self.float_type).unsqueeze(0)
                    azimuths = torch.tensor(azimuths, dtype=self.float_type).unsqueeze(0)
                    az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type).unsqueeze(0)
                    fft_cfar = cfar_mask(fft_data, polar_res, a_thresh=a_thresh, b_thresh=b_thresh, diff=False)

                    # Save CFAR image
                    if network_input_type == 'cartesian':
                        fft_cfar = radar_polar_to_cartesian_diff(fft_cfar, azimuths, polar_res)
                    cv2.imwrite(loc_cfar_path, 255*fft_cfar.squeeze(0).numpy())

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
                    

                    # Plot for visualization
                    """
                    map_pts_loc_frame = (T_gt_idx[:3,:3] @ map_pts_sensor_frame[:,:3].T + T_gt_idx[:3, 3:4]).T
                    plt.figure(figsize=(15,15))
                    plt.scatter(map_pts_loc_frame[:,0], map_pts_loc_frame[:,1], s=1.0, c='red')
                    plt.scatter(curr_filt_pts[:,0], curr_filt_pts[:,1], s=0.5, c='blue')
                    plt.ylim([-80, 80])
                    plt.xlim([-80, 80])
                    plt.savefig('align.png')
                    plt.close()
                    time.sleep(0.1)
                    """


                    if curr_raw_pts.shape[0] > local_max_loc_pts:
                        local_max_loc_pts = curr_raw_pts.shape[0]
                    if map_pts.shape[0] > local_max_map_pts:
                        local_max_map_pts = map_pts.shape[0]

                # Now, loop through the number of augmented samples we want to generate
                # For each sample, generate a new perturbation, and save a duplicate
                # of the rest of the information. The __getitem__ function will deal
                # with the actual augmentation based on the augment_id of the sample
                for aug_id in range(augment_factor):
                    # Generate random perturbation to ground truth pose
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
                        self.augment_id_vector = np.array([aug_id])
                        self.T_loc_gt = T_gt_idx.unsqueeze(0)
                        self.T_loc_init = T_init_idx.unsqueeze(0)
                    else:
                        self.v_id_vector = np.append(self.v_id_vector, loc_v.id)
                        self.graph_id_vector = np.append(self.graph_id_vector, pair_idx)
                        self.augment_id_vector = np.append(self.augment_id_vector, aug_id)
                        self.T_loc_gt = torch.cat((self.T_loc_gt, T_gt_idx.unsqueeze(0)), dim=0)
                        self.T_loc_init = torch.cat((self.T_loc_init, T_init_idx.unsqueeze(0)), dim=0)

                    self.loc_radar_path_list.append(loc_radar_path)
                    self.loc_cfar_path_list.append(loc_cfar_path)
                if (ii % 100) == 0:
                    print(str(ii) + " data samples processed")

                if num_samples > 0 and self.v_id_vector.shape[0] >= num_samples * augment_factor:
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
                if local_max_map_pts > self.max_map_pts:
                    self.max_map_pts = local_max_map_pts

        # Assert that the number of all elements are the same
        assert self.v_id_vector.shape[0] == self.graph_id_vector.shape[0] == self.T_loc_gt.shape[0] \
            == self.augment_id_vector.shape[0] \
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
        plt.figure(figsize=(15,15))
        plt.scatter(map_pc[:,0], map_pc[:,1], s=1.0, c='red')
        #plt.scatter(map_pts[:,0], map_pts[:,1], s=1.0, c='red')
        plt.scatter(scan_pc_filt[:,0], scan_pc_filt[:,1], s=0.5, c='blue')
        # plt.scatter(scan_pc[:,0], scan_pc[:,1], s=0.5, c='green')
        #plt.scatter(curr_pts_map_frame[:,0], curr_pts_map_frame[:,1], s=0.5, c='green')
        plt.ylim([-80, 80])
        plt.xlim([-80, 80])
        plt.savefig('align.png')
        plt.close()
        """

        # Load in fft data
        loc_radar_img = cv2.imread(self.loc_radar_path_list[index], cv2.IMREAD_GRAYSCALE)
        if self.network_input_type == 'cartesian':
            fft_data = torch.tensor(loc_radar_img, dtype=self.float_type)/255.0
        else:
            raise NotImplementedError('Only cartesian input is supported for now')
            fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
            fft_data = torch.tensor(fft_data, dtype=self.float_type)
            azimuths = torch.tensor(azimuths, dtype=self.float_type)
            az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type)

        fft_cfar = cv2.imread(self.loc_cfar_path_list[index], cv2.IMREAD_GRAYSCALE)
        fft_cfar = torch.tensor(fft_cfar, dtype=self.float_type)/255.0

        """
        cart_indeces = point_to_cart_idx(scan_pc_raw.unsqueeze(0), cart_resolution=0.2384, cart_pixel_width=640, min_to_plus_1=False)
        print(cart_indeces.shape)
        figure = plt.figure(figsize=(15,15))
        plt.imshow(fft_data, cmap='gray')
        plt.scatter(cart_indeces[0,:,1], cart_indeces[0,:,0], s=2.5, c='red')
        plt.savefig('fft_og.png')
        plt.close()
        """

        # Deal with data augmentation
        aug_id = self.augment_id_vector[index]
        if aug_id > 0:
            scan_pc_raw, scan_pc_filt, map_pc, fft_data, fft_cfar = \
                self.augment_data(scan_pc_raw, scan_pc_filt, map_pc, fft_data,
                                  fft_cfar, aug_id)

        # DELETE
        """
        raw_indeces = point_to_cart_idx(scan_pc_raw.unsqueeze(0), cart_resolution=0.2384, cart_pixel_width=640, min_to_plus_1=False)
        filt_indeces = point_to_cart_idx(scan_pc_filt.unsqueeze(0), cart_resolution=0.2384, cart_pixel_width=640, min_to_plus_1=False)
        
        map_pc_idx = map_pc[torch.abs(map_pc[:,0]) < 100].unsqueeze(0)
        map_indeces = point_to_cart_idx(map_pc_idx, cart_resolution=0.2384, cart_pixel_width=640, min_to_plus_1=False)
        figure = plt.figure(figsize=(15,15))
        plt.imshow(fft_data, cmap='gray')
        plt.scatter(map_indeces[0,:,1], map_indeces[0,:,0], s=2.5, c='red')
        plt.scatter(raw_indeces[0,:,1], raw_indeces[0,:,0], s=2.5, c='blue')
        plt.scatter(filt_indeces[0,:,1], filt_indeces[0,:,0], s=2.5, c='green')
        plt.savefig('fft_aug.png')
        plt.close()
        time.sleep(2.0)
        

        map_pc_batch = torch.stack([map_pc, map_pc+10], dim=0)
        print(map_pc_batch.shape)
        map_pt_mask = extract_bev_from_pts(map_pc_batch)

        map_pt_mask = map_pt_mask[0]

        figure = plt.figure(figsize=(15,15))
        #plt.imshow(fft_data, cmap='gray')
        plt.imshow(map_pt_mask.squeeze(0), cmap='gray')
        plt.savefig('map_pt_mask.png')
        plt.close()
        time.sleep(2.0)
        fadsdfsa
        """


        loc_data = {'raw_pc': scan_pc_raw, 'filtered_pc': scan_pc_filt,
                    'fft_data' : fft_data, 'fft_cfar' : fft_cfar, 'timestamp' : loc_stamp}
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
        map_pc_pad = self.target_pad_val*torch.ones((self.max_map_pts - map_pts_sensor_frame.shape[0], 3), dtype=self.float_type)
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
    
    def augment_data(self, scan_pc_raw, scan_pc_filt, map_pc, fft_data, fft_cfar, aug_id):
        if not self.gt_eye:
            raise NotImplementedError('Only gt_eye=True is supported at this time')

        aug = self.augmentations[aug_id]
        flip = aug['flip']
        rot = aug['rot']
        if flip:
            fft_data = torch.flipud(fft_data)
            fft_cfar = torch.flipud(fft_cfar)
            scan_pc_raw[:,0] *= -1
            scan_pc_filt[:,0] *= -1
            map_pc[:,0] *= -1
        if rot != 0:
            fft_data = torch.rot90(fft_data, k=rot)
            fft_cfar = torch.rot90(fft_cfar, k=rot)
            if rot == 1:
                rot90_mat = torch.tensor([[0, -1],
                                    [1,  0]], dtype=self.float_type)
            elif rot == 2:
                rot90_mat = torch.tensor([[-1,  0],
                                    [ 0, -1]], dtype=self.float_type)
            elif rot == 3:
                rot90_mat = torch.tensor([[ 0, 1],
                                    [-1, 0]], dtype=self.float_type)
            else:
                rot90_mat = torch.eye(2, dtype=self.float_type)

            scan_pc_raw[:,:2] = torch.matmul(scan_pc_raw[:,:2], rot90_mat)
            scan_pc_filt[:,:2] = torch.matmul(scan_pc_filt[:,:2], rot90_mat)
            map_pc[:,:2] = torch.matmul(map_pc[:,:2], rot90_mat)
            if map_pc.shape[1] == 6:
                map_pc[:,3:5] = torch.matmul(map_pc[:,3:5], rot90_mat)

        return scan_pc_raw, scan_pc_filt, map_pc, fft_data, fft_cfar