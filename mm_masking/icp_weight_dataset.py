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
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
import vtr_pose_graph.graph_utils as g_utils
from vtr_pose_graph.graph_iterators import TemporalIterator
from utils.extract_graph import extract_points_and_map, extract_T_v_this
import time
import pandas as pd

class ICPWeightDataset():

    def __init__(self, loc_pairs, params=None, dataset_type='train'):
        # Frame explainer:
        # F_w: world frame (to which all grountruth poses are in reference to)
        # F_map: global map frame (global frame in which the reference submaps are resolved in, NOT THE MAP SENSOR FRAME)
        # F_ms: map sensor frame (e.g. lidar in radar-lidar loc)
        # F_ls: loc sensor frame (e.g. radar in radar-lidar loc)
        # F_r: (teach) robot frame, don't need repeat robot frame so just using r
        # Pointclouds:
        # submap: submap pointcloud being localized against (collected using map sensor, originally in F_map)
        # raw: raw pointcloud being localized (collected using loc sensor, originally in F_ls)
        # filt: filtered pointcloud being localized (collected using loc sensor, originally in F_ls)
        # 
        # We want to estimate the transform from F_map (in which submap points are resolved in originally) to F_ls [T_map_ls]

        # Load in params
        map_sensor=params["map_sensor"]
        loc_sensor=params["loc_sensor"]
        random=params["random"]
        if dataset_type == 'train':
            num_samples=params["num_train"]
            self.augment = params["augment"]
        else:
            num_samples=params["num_val"]
            self.augment = False
        
        float_type=params["float_type"]
        use_gt=params["use_gt"]
        gt_eye=params["gt_eye"]
        pos_std=params["pos_std"]
        rot_std=params["rot_std"]
        a_thresh=params["a_thresh"]
        b_thresh=params["b_thresh"]
        network_input_type = params["network_input_type"]

        self.load_from_pc = True

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
        elif map_sensor == 'lidar' and loc_sensor == 'lidar':
            sensor_dir_name = 'lidar'
            self.msg_prefix = ''
        else:
            raise ValueError("Invalid sensor combination")

        data_dir = '../data'
        dataset_dir = osp.join(data_dir, 'vtr_data')
        vtr_result_dir = osp.join(data_dir, 'vtr_results')
        vtr_raw_result_dir = osp.join(data_dir, 'vtr_results_raw')
        self.polar_res = 0.0596
        
        self.v_id_vector = None
        self.graph_id_vector = None
        self.T_map_ls_gt = None
        self.T_map_ls_init = None
        self.T_ms_r = []
        self.T_r_map = []
        self.graph_list = []
        self.loc_radar_path_list = []
        self.loc_cfar_path_list = []
        # Need to save max loc and map pointcloud size for padding for batch assembly
        self.max_loc_pts = 0
        self.max_map_pts = 0

        for pair_idx, pair in enumerate(loc_pairs):
            map_seq = pair[0]
            loc_seq = pair[1]

            gt_w_ms_poses, gt_ms_times = read_traj_file_gt2(osp.join(dataset_dir, map_seq, "applanix", map_sensor + "_poses.csv"), dim=2)
            gt_w_ls_poses, gt_ls_times = read_traj_file_gt2(osp.join(dataset_dir, loc_seq, "applanix", loc_sensor + "_poses.csv"), dim=2)
            graph_dir = osp.join(vtr_result_dir, sensor_dir_name, map_seq, loc_seq, 'graph')
            factory = Rosbag2GraphFactory(graph_dir)
            pair_graph = factory.buildGraph()
            self.graph_list.append(pair_graph)
            v_start = pair_graph.get_vertex((1,0))
            print("Loading loc pair: " + str(pair) + " with " + str(pair_graph.number_of_vertices) + " vertices and " + str(pair_graph.number_of_edges) + " edges")
            
            # Save transform from robot to map sensor
            # This is needed because the map pointcloud is saved in robot frame (through submap frame),
            # but ground truth is between map sensor and loc sensor.
            # This transform is constant for a given map sequence
            T_axel_applanix = np.array([[0.0299955, 0.99955003, 0, 0.51],
                                    [-0.99955003, 0.0299955, 0., 0.0],
                                    [ 0, 0, 1, 1.45],
                                    [ 0, 0, 0, 1]])

            T_applanix_lidar = np.loadtxt(osp.join(dataset_dir, map_seq, 'calib', 'T_applanix_lidar.txt'))
            if map_sensor == 'radar':
                T_radar_lidar = np.loadtxt(osp.join(dataset_dir, map_seq, 'calib', 'T_radar_lidar.txt'))
                T_robot_map_sensor = T_axel_applanix @ T_applanix_lidar @ get_inverse_tf(T_radar_lidar)
            elif map_sensor == 'lidar':
                T_robot_map_sensor = T_axel_applanix @ T_applanix_lidar
            T_ms_r = torch.from_numpy(get_inverse_tf(T_robot_map_sensor)).type(float_type)
            self.T_ms_r.append(T_ms_r)

            # Check if result directory contains a metadata file
            # If not, create one
            metadata_path = osp.join(vtr_result_dir, sensor_dir_name, map_seq, loc_seq, 'metadata.csv')
            if not osp.exists(metadata_path):
                df_data = {'complete' : 0, 'up_to_idx': -1, 'max_loc': -1, 'max_map': -1}
                df = pd.DataFrame(df_data, index=[0])
                df.to_csv(metadata_path, index=False)

            # Form directories for direct pointclouds
            if self.load_from_pc:
                map_pc_path = osp.join(vtr_raw_result_dir, sensor_dir_name, map_seq, 'map_pc')
                if not osp.exists(map_pc_path):
                    os.makedirs(map_pc_path)
                loc_path = osp.join(vtr_raw_result_dir, sensor_dir_name, map_seq, loc_seq)
                if not osp.exists(loc_path):
                    os.makedirs(loc_path)
                    os.makedirs(osp.join(loc_path, 'raw_pts'))
                    os.makedirs(osp.join(loc_path, 'filt_pts'))
                gt_transform_path = osp.join(loc_path, 'groundtruth.csv')

            # Load in the metadata file to see if we need to extract max points during
            # data loading. 
            pair_df = pd.read_csv(metadata_path)
            # If we have sufficient metadata about what we wish to extract,
            # don't bother extracting more
            extract_pcs_metadata = True
            save_gt_path = False
            if (pair_df['complete'][0] == 1 or (pair_df['up_to_idx'][0] >= num_samples and num_samples>0)):
                extract_pcs_metadata = False
                # Check if new max is reached
                if pair_df['max_loc'][0] > self.max_loc_pts:
                    self.max_loc_pts = pair_df['max_loc'][0]
                if pair_df['max_map'][0] > self.max_map_pts:
                    self.max_map_pts = pair_df['max_map'][0]
                
                if self.load_from_pc:
                    if not osp.exists(gt_transform_path):
                        save_gt_path = True
                    else:
                        save_gt_path = False
            else:
                if self.load_from_pc:
                    # Delete gt_transform_path file
                    if osp.exists(gt_transform_path):
                        os.remove(gt_transform_path)
                    save_gt_path = True

            print("Loading from metadata: " + str(not extract_pcs_metadata))
            local_max_loc_pts = 0
            local_max_map_pts = 0
            for ii, (loc_v, e) in enumerate(TemporalIterator(v_start)):
                # Check if vertex is valid
                if e.from_id == vtr_pose_graph.INVALID_ID:
                    print("Skipping vertex ", loc_v, "due to invalid edge")
                    continue
                
                # Extract vertex info
                try:
                    map_v = g_utils.get_closest_teach_vertex(loc_v)
                except g_utils.GraphError:
                    # Sometimes will trigger "Graph is malformed, repeat pass does not connect to teach vertex."
                    # Not 100% sure why this happens on runs that have an otherwise
                    # good error, but just skip this for training
                    print("Skipping vertex ", loc_v, "due to malformed graph error")
                    continue
                map_ptr = map_v.get_data("pointmap_ptr")
                map_v = pair_graph.get_vertex(map_ptr.map_vid)

                # Extract timestamps
                loc_stamp = int(loc_v.stamp * 1e-3)
                map_stamp = int(map_v.stamp * 1e-3)

                if not (map_sensor == 'lidar' and loc_sensor == 'lidar'):
                    # Ensure radar image exists
                    loc_radar_path = osp.join(dataset_dir, loc_seq, 'radar', str(loc_stamp) + ".png")
                    
                    if not osp.exists(loc_radar_path):
                        print("Radar image does not exist at: ", loc_radar_path)
                        continue

                    # Ensure CFAR of image exists, if it does not, create one
                    # This is done to speed up training so that CFAR image does not need to be created every time
                    #cfar_dir = osp.join(data_dir, 'cfar', loc_seq, network_input_type, str(a_thresh) + '_' + str(b_thresh))
                    cfar_dir = osp.join(data_dir, 'cfar', loc_seq, 'polar', str(a_thresh) + '_' + str(b_thresh))
                    if not osp.exists(cfar_dir):
                        os.makedirs(cfar_dir)
                    loc_cfar_path = osp.join(cfar_dir, str(loc_stamp) + ".png")
                    if not osp.exists(loc_cfar_path):
                        loc_radar_img = cv2.imread(loc_radar_path, cv2.IMREAD_GRAYSCALE)
                        fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
                        fft_data = torch.tensor(fft_data, dtype=self.float_type).unsqueeze(0)
                        azimuths = torch.tensor(azimuths, dtype=self.float_type).unsqueeze(0)
                        az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type).unsqueeze(0)
                        fft_cfar = cfar_mask(fft_data, self.polar_res, a_thresh=a_thresh, b_thresh=b_thresh, diff=False)

                        # Save CFAR image
                        #if network_input_type == 'cartesian':
                        #    fft_cfar = radar_polar_to_cartesian_diff(fft_cfar, azimuths, self.polar_res)
                        cv2.imwrite(loc_cfar_path, (255*fft_cfar.squeeze(0).numpy()).astype(np.uint8))
                else:
                    loc_radar_path = 0
                    loc_cfar_path = 0

                # Check that timestamps are matching to gt poses
                assert loc_stamp == gt_ls_times[ii], "query: {}, gt stamp: {}".format(loc_stamp, gt_ls_times[ii])
                closest_map_t = get_closest_index(map_stamp, gt_ms_times)
                assert map_stamp == gt_ms_times[closest_map_t], "query: {}".format(map_stamp)

                # Extract gt map pose
                gt_w_ms_pose_idx = gt_w_ms_poses[closest_map_t]
                T_ls_ms_gt = get_inverse_tf(gt_w_ls_poses[ii]) @ gt_w_ms_pose_idx

                # Save ground truth map sensor to localization sensor pose
                T_ls_ms_gt = torch.tensor(T_ls_ms_gt, dtype=float_type)
                
                # Now that we have ground truth, we can filter the map points to know max point size
                # We only do filtering for lidar and only if we dont already have
                # pointcloud metadata. This is done to speed up data loading
                if extract_pcs_metadata:
                    if map_sensor == 'lidar' and loc_sensor == 'lidar':
                        extract_raw_pts = False
                    else:
                        extract_raw_pts = True
                    # Curr points are in the sensor frame they were collected in for repeat, map points are in robot frame
                    raw_ls, filt_ls, submap_map, T_r_map, _, _ = extract_points_and_map(pair_graph, loc_v, msg_prefix=self.msg_prefix, extract_raw_pts=extract_raw_pts)
                    assert filt_ls.shape == raw_ls.shape, 'Raw and filtered pointclouds dont match!'
                    
                    # Save loc pointcloud to loc_path
                    if self.load_from_pc:
                        loc_pc_raw_path = osp.join(loc_path, 'raw_pts', str(loc_stamp) + '.bin')
                        raw_ls.tofile(loc_pc_raw_path)
                        loc_pc_filt_path = osp.join(loc_path, 'filt_pts', str(loc_stamp) + '.bin')
                        filt_ls.tofile(loc_pc_filt_path)
                        map_pc_path_idx = osp.join(map_pc_path, str(loc_stamp) + '.bin')
                        if not osp.exists(map_pc_path_idx):
                            submap_map.tofile(map_pc_path_idx)

                    # Update max point sizes for metadata
                    if raw_ls.shape[0] > local_max_loc_pts:
                        local_max_loc_pts = raw_ls.shape[0]
                    if submap_map.shape[0] > local_max_map_pts:
                        local_max_map_pts = submap_map.shape[0]

                    # # Plot for visualization
                    # T_ls_map = T_ls_ms_gt.numpy() @ T_ms_r.numpy() @ T_r_map
                    # submap_ls = (T_ls_map[:3,:3] @ submap_map[:,:3].T + T_ls_map[:3, 3:4]).T
                    # print(ii)
                    # plt.figure(figsize=(15,15))
                    # plt.scatter(submap_ls[:,0], submap_ls[:,1], s=1.0, c='red')
                    # plt.scatter(filt_ls[:,0], filt_ls[:,1], s=0.5, c='blue')
                    # plt.ylim([-80, 80])
                    # plt.xlim([-80, 80])
                    # plt.savefig('align.png')
                    # plt.close()
                    # time.sleep(0.5)
                else:
                    T_r_map = extract_T_v_this(loc_v, msg='submap_loc')

                T_r_map = torch.from_numpy(T_r_map).type(float_type)
                self.T_r_map.append(T_r_map)
                
                # Compute groundtruth transform from map to loc sensor frame
                T_map_ls_gt = torch.tensor(get_inverse_tf(T_ls_ms_gt.numpy() @ T_ms_r.numpy() @ T_r_map.numpy()), dtype=float_type)

                # Generate random perturbation to ground truth pose
                # The map pointcloud is transformed into the scan frame using T_gt
                # T_init is the initial guess that is offset from T_gt that the ICP
                # needs to "unlearn" to get to identity
                if use_gt:
                    if gt_eye:
                        T_map_ls_init = np.eye(4)
                    else:
                        T_map_ls_init = T_map_ls_gt.numpy()
                else:
                    if dataset_type == 'train':
                        xi_rand = 2 * torch.rand((6,1), dtype=float_type) - 1
                        # Scale x and y
                        xi_rand[0:2] = pos_std*xi_rand[0:2]
                        # Scale yaw
                        xi_rand[5] = rot_std*xi_rand[5]
                        # Zero out z, pitch, and roll
                        xi_rand[2:5] = 0.0
                    else:
                        xi_phi = np.random.normal(0.0, rot_std)
                        xi_x = np.random.normal(0.0, pos_std)
                        xi_y = np.random.normal(0.0, pos_std)
                        xi_rand = torch.tensor([[xi_x], [xi_y], [0.0], [0.0], [0.0], [xi_phi]], dtype=float_type)

                    T_rand = Transformation(xi_ab=xi_rand).matrix()
                    if gt_eye:
                        T_map_ls_init = T_rand # @ identity
                    else:
                        T_map_ls_init = T_rand @ T_map_ls_gt.numpy()

                T_map_ls_init = torch.tensor(T_map_ls_init, dtype=float_type)

                # Stack data for more efficient storage and retrieval
                if self.v_id_vector is None:
                    self.v_id_vector = np.array([loc_v.id])
                    self.graph_id_vector = np.array([pair_idx])
                    self.T_map_ls_gt = T_map_ls_gt.unsqueeze(0)
                    self.T_map_ls_init = T_map_ls_init.unsqueeze(0)
                else:
                    self.v_id_vector = np.append(self.v_id_vector, loc_v.id)
                    self.graph_id_vector = np.append(self.graph_id_vector, pair_idx)
                    self.T_map_ls_gt = torch.cat((self.T_map_ls_gt, T_map_ls_gt.unsqueeze(0)), dim=0)
                    self.T_map_ls_init = torch.cat((self.T_map_ls_init, T_map_ls_init.unsqueeze(0)), dim=0)

                self.loc_radar_path_list.append(loc_radar_path)
                self.loc_cfar_path_list.append(loc_cfar_path)
                if (ii % 100) == 0:
                    print(str(ii) + " data samples processed")

                # Save map timestamp, loc timestamp, and gt transform to file
                if self.load_from_pc and save_gt_path:
                    T_map_ls_gt = T_map_ls_gt.numpy()
                    df_data = {'map_stamp' : map_stamp, 'loc_stamp': loc_stamp, 
                               'T_map_ls_0_0': T_map_ls_gt[0,0], 'T_map_ls_0_1': T_map_ls_gt[0,1], 'T_map_ls_0_2': T_map_ls_gt[0,2], 'T_map_ls_0_3': T_map_ls_gt[0,3],
                               'T_map_ls_1_0': T_map_ls_gt[1,0], 'T_map_ls_1_1': T_map_ls_gt[1,1], 'T_map_ls_1_2': T_map_ls_gt[1,2], 'T_map_ls_1_3': T_map_ls_gt[1,3],
                               'T_map_ls_2_0': T_map_ls_gt[2,0], 'T_map_ls_2_1': T_map_ls_gt[2,1], 'T_map_ls_2_2': T_map_ls_gt[2,2], 'T_map_ls_2_3': T_map_ls_gt[2,3],
                               'T_map_ls_3_0': T_map_ls_gt[3,0], 'T_map_ls_3_1': T_map_ls_gt[3,1], 'T_map_ls_3_2': T_map_ls_gt[3,2], 'T_map_ls_3_3': T_map_ls_gt[3,3]}
                    df = pd.DataFrame(df_data, index=[0])
                    if not osp.exists(gt_transform_path):
                        df.to_csv(gt_transform_path, index=False)
                    else:
                        df.to_csv(gt_transform_path, mode='a', header=False, index=False)


                if num_samples > 0 and self.v_id_vector.shape[0] >= num_samples:
                    break
            
            # Update metadata if it was collected
            if extract_pcs_metadata:
                # Save metadata
                meta_complete = (num_samples == -1)
                df_data = {'complete' : meta_complete, 'up_to_idx': ii, 'max_loc': local_max_loc_pts, 'max_map': local_max_map_pts}
                df = pd.DataFrame(df_data, index=[0])
                df.to_csv(metadata_path, index=False)

                if self.load_from_pc:
                    metadata_raw_path = osp.join(vtr_raw_result_dir, sensor_dir_name, map_seq, loc_seq, 'metadata.csv')
                    df.to_csv(metadata_raw_path, index=False)
                
                # Overwrite max point sizes if they are larger
                if local_max_loc_pts > self.max_loc_pts:
                    self.max_loc_pts = local_max_loc_pts
                if local_max_map_pts > self.max_map_pts:
                    self.max_map_pts = local_max_map_pts        

        # Assert that the number of all elements are the same
        assert self.v_id_vector.shape[0] == self.graph_id_vector.shape[0] == self.T_map_ls_gt.shape[0] \
            == self.T_map_ls_init.shape[0] == len(self.loc_radar_path_list) == len(self.loc_cfar_path_list)

    def __len__(self):
        return self.v_id_vector.shape[0]

    def __getitem__(self, index):
        # Load in initial guess
        T_map_ls_init = self.T_map_ls_init[index]

        # Load in ground truth localization to map pose
        T_map_ls_gt = self.T_map_ls_gt[index]

        # Load in pointclouds and timestamps
        scan_pc_raw, scan_pc_filt, map_pc, loc_stamp, map_stamp = self.load_graph_data(index)
        assert scan_pc_raw.shape == scan_pc_filt.shape, 'Raw and filtered pointclouds dont match!'

        if not (self.map_sensor == 'lidar' and self.loc_sensor == 'lidar'):
            # Load in fft data
            loc_radar_img = cv2.imread(self.loc_radar_path_list[index], cv2.IMREAD_GRAYSCALE)
            fft_data, azimuths, az_timestamps = load_radar(loc_radar_img)
            fft_data = torch.tensor(fft_data, dtype=self.float_type)
            azimuths = torch.tensor(azimuths, dtype=self.float_type)
            az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type)

            fft_cfar = cv2.imread(self.loc_cfar_path_list[index], cv2.IMREAD_GRAYSCALE)
            fft_cfar = torch.tensor(fft_cfar, dtype=self.float_type)/255.0

            # Deal with data augmentation
            if self.augment:
                scan_pc_raw, scan_pc_filt, map_pc, azimuths, fft_data, fft_cfar = \
                    self.augment_data(scan_pc_raw, scan_pc_filt, map_pc, azimuths, fft_data, fft_cfar)

            if self.network_input_type == 'cartesian':
                fft_data = radar_polar_to_cartesian_diff(fft_data.unsqueeze(0), azimuths.unsqueeze(0), self.polar_res).squeeze(0)
                fft_cfar = radar_polar_to_cartesian_diff(fft_cfar.unsqueeze(0), azimuths.unsqueeze(0), self.polar_res).squeeze(0)
        else:
            fft_data = 0.0
            fft_cfar = 0.0

        loc_data = {'raw_pc': scan_pc_raw, 'filtered_pc': scan_pc_filt,
                    'fft_data' : fft_data, 'fft_cfar' : fft_cfar, 'timestamp' : loc_stamp}
        map_data = {'pc': map_pc, 'timestamp' : map_stamp}
        T_data = {'T_map_ls_init' : T_map_ls_init, 'T_map_ls_gt' : T_map_ls_gt}

        return {'loc_data': loc_data, 'map_data': map_data, 'transforms': T_data}
    
    def load_graph_data(self, idx):
        v_id = self.v_id_vector[idx].item() # Need .item() as v_id must be int, not np.int32/64
        graph_id = self.graph_id_vector[idx]
        pair_graph = self.graph_list[graph_id]
        vertex = pair_graph.get_vertex(v_id)
        if self.map_sensor == 'lidar' and self.loc_sensor == 'lidar':
            extract_raw_pts = False
        else:
            extract_raw_pts = True
        
        raw_ls, filt_ls, submap_map, _, loc_stamp, map_stamp = extract_points_and_map(pair_graph, vertex, msg_prefix=self.msg_prefix, extract_raw_pts=extract_raw_pts)
        
        # Make scan_pc batchable
        raw_ls = torch.from_numpy(raw_ls[:,:3])
        filt_ls = torch.from_numpy(filt_ls[:,:3])
        scan_pc_pad = torch.zeros((self.max_loc_pts - raw_ls.shape[0], 3), dtype=self.float_type)
        scan_pc_raw = torch.cat((raw_ls, scan_pc_pad), dim=0)
        scan_pc_filt = torch.cat((filt_ls, scan_pc_pad), dim=0)
        
        # Transform map pointcloud to scan frame
        submap_map = torch.from_numpy(submap_map)

        if (self.gt_eye):
            T_ls_map_gt = torch.tensor(get_inverse_tf(self.T_map_ls_gt[idx].numpy()), dtype=self.float_type)
            submap_map[:,:3] = (T_ls_map_gt[:3,:3] @ submap_map[:,:3].T + T_ls_map_gt[:3, 3:4]).T
            submap_map[:,3:6] = (T_ls_map_gt[:3,:3] @ submap_map[:,3:6].T).T

        # Make map_pc batchable
        map_pc_pad = self.target_pad_val*torch.ones((self.max_map_pts - submap_map.shape[0], submap_map.shape[1]), dtype=self.float_type)
        map_pc = torch.cat((submap_map, map_pc_pad), dim=0)

        return scan_pc_raw, scan_pc_filt, map_pc, loc_stamp, map_stamp
    
    def augment_data(self, scan_pc_raw, scan_pc_filt, map_pc, azimuths, fft_data, fft_cfar):
        if not self.gt_eye:
            raise NotImplementedError('Only gt_eye=True is supported at this time')

        # Generate random angle between 0 and 2pi
        angle = 2*np.pi*torch.rand(1, dtype=self.float_type)
        rot_mat = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                [torch.sin(angle), torch.cos(angle)]], dtype=self.float_type)
        
        # Rotate map pointcloud
        scan_pc_raw[:,:2] = torch.matmul(scan_pc_raw[:,:2], rot_mat)
        scan_pc_filt[:,:2] = torch.matmul(scan_pc_filt[:,:2], rot_mat)
        map_pc[:,:2] = torch.matmul(map_pc[:,:2], rot_mat)
        if map_pc.shape[1] == 6:
            map_pc[:,3:5] = torch.matmul(map_pc[:,3:5], rot_mat)

        # Rotate fft data by shifting the azimuths
        azimuths = azimuths - angle
        # Cap azimuths to 0-2pi
        azimuths[azimuths < 0.0] = azimuths[azimuths < 0.0] + 2*np.pi
        # Find new min azimuth index
        min_az_idx = torch.argmin(azimuths)
        # Roll azimuths and fft data so that min azimuth is at index 0
        azimuths = torch.roll(azimuths, -min_az_idx.item(), dims=0)
        fft_data = torch.roll(fft_data, -min_az_idx.item(), dims=0)
        fft_cfar = torch.roll(fft_cfar, -min_az_idx.item(), dims=0)

        return scan_pc_raw, scan_pc_filt, map_pc, azimuths, fft_data, fft_cfar

    def get_item_from_loc_timestamp(self, loc_stamp_req):
        # Find the index of the loc_stamp
        # We know the path will contain the loc_stamp
        loc_radar_path_to_find = str(loc_stamp_req) + ".png"
        # Find loc_radar_path_to_find in self.loc_radar_path_list
        index = [i for i, s in enumerate(self.loc_radar_path_list) if loc_radar_path_to_find in s]
        assert index != [], 'loc_stamp_req not found in dataset'
        index = index[0]

        get_item_res = self.__getitem__(index)
        get_item_res['index'] = index

        return get_item_res