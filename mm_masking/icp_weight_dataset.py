import cv2
import torch
import numpy as np
import os.path as osp
from pyboreas.utils.odometry import read_traj_file2
import matplotlib.pyplot as plt
from pylgmath import se3op, Transformation
from radar_utils import load_radar, cfar_mask, extract_pc, load_pc_from_file, radar_cartesian_to_polar, radar_polar_to_cartesian_diff
from dICP.ICP import ICP

class ICPWeightDataset():

    def __init__(self, gt_data_dir, pc_dir, radar_dir, loc_pairs, sensor,
                 random=False, num_samples=-1, size_pc=-1, verbose=False,
                 float_type=torch.float64, use_gt=False, gt_eye=True, pos_std=1.0, rot_std=0.1,
                 a_thresh=1.0, b_thresh=0.09):
        self.loc_pairs = loc_pairs
        self.size_pc = size_pc
        self.float_type = float_type

        if not random:
            np.random.seed(99)
            torch.manual_seed(99)

        # Loop through loc_pairs and load in ground truth loc poses
        T_loc_gt = None
        T_loc_init = None
        map_pc_list = []
        self.loc_timestamps = []
        self.map_timestamps = []
        
        for pair in loc_pairs:
            map_seq = pair[0]
            loc_seq = pair[1]
            
            # Load in ground truth localization to map poses
            # T_gt is the ground truth localization transform between the current scan
            # and the reference submap scan
            gt_file = osp.join(gt_data_dir, map_seq, loc_seq + '.txt')
            T_gt, loc_times, map_times, _, _ = read_traj_file2(gt_file)

            # Find localization cartesian images with names corresponding to pred_times
            loc_timestamps = []
            map_timestamps = []
            all_fft_data = None
            all_azimuths = None
            all_az_timestamps = None
            incomplete_loc_times = []
            for idx, loc_time in enumerate(loc_times):
                # Load in localization paths
                loc_radar_path = osp.join(radar_dir, loc_seq, str(loc_time) + '.png')
                #loc_pc_path = osp.join(pc_dir, sensor, loc_seq, str(loc_time) + '.bin')
                loc_pc_path = osp.join(pc_dir, loc_seq, str(loc_time) + '.bin')

                # Load in map paths
                #map_cart_path = osp.join(cart_dir, map_seq, str(map_times[idx]) + '.png')
                map_pc_path = osp.join(pc_dir, sensor, map_seq, str(map_times[idx]) + '.bin')
                #map_pc_path = osp.join(pc_dir, map_seq, str(map_times[idx]) + '.bin')

                # Check if the paths exist, if not then save timestamps that don't exist
                if not osp.exists(loc_radar_path) or not osp.exists(loc_pc_path) or \
                    not osp.exists(map_pc_path):
                    if verbose:
                        print('WARNING: Images or point clouds don\'t exist')
                        print('Localization time: ' + str(loc_time))
                        print('Map time: ' + str(map_times[idx]))

                    # Save timestamp that does not exist
                    incomplete_loc_times.append(loc_time)
                    continue
                else:
                    # Save timestamp
                    loc_timestamps.append(loc_time)
                    map_timestamps.append(map_times[idx])

                    # Save fft data
                    # Load in localization polar image
                    loc_radar_img = cv2.imread(loc_radar_path, cv2.IMREAD_GRAYSCALE)
                    loc_radar_mat = np.asarray(loc_radar_img)
                    fft_data, azimuths, az_timestamps = load_radar(loc_radar_mat)
                    fft_data = torch.tensor(fft_data, dtype=self.float_type).unsqueeze(0)
                    azimuths = torch.tensor(azimuths, dtype=self.float_type).unsqueeze(0)
                    az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type).unsqueeze(0)

                    if all_fft_data is None:
                        all_fft_data = fft_data
                        all_azimuths = azimuths
                        all_az_timestamps = az_timestamps
                    else:
                        all_fft_data = torch.cat((all_fft_data, fft_data), dim=0)
                        all_azimuths = torch.cat((all_azimuths, azimuths), dim=0)
                        all_az_timestamps = torch.cat((all_az_timestamps, az_timestamps), dim=0)

                    # Save ground truth localization to map pose
                    T_gt_idx = torch.tensor(T_gt[idx], dtype=float_type)
                    # Also, generate random perturbation to ground truth pose
                    # The map pointcloud is transformed into the scan frame using T_gt
                    # T_init is the initial guess that is offset from T_gt that the ICP
                    # needs to "unlearn" to get to identity
                    if use_gt:
                        if gt_eye:
                            T_init_idx = np.eye(4)
                        else:
                            T_init_idx = T_gt[idx]
                    else:
                        xi_rand = torch.randn((6,1), dtype=float_type)
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
                            T_init_idx = T_rand.matrix() @ T_gt[idx]
                    T_init_idx = torch.tensor(T_init_idx, dtype=float_type)

                    # Stack transformations
                    if T_loc_gt is None:
                        T_loc_gt = T_gt_idx.unsqueeze(0)
                        T_loc_init = T_init_idx.unsqueeze(0)
                    else:
                        T_loc_gt = torch.cat((T_loc_gt, T_gt_idx.unsqueeze(0)), dim=0)
                        T_loc_init = torch.cat((T_loc_init, T_init_idx.unsqueeze(0)), dim=0)
                
                    # Load in map pointcloud
                    map_pc_ii = load_pc_from_file(map_pc_path, to_type=self.float_type, flip_y=True)
                    # If want groundtruth to be identity, transform map point cloud to scan frame
                    if gt_eye:
                        T_sm = torch.linalg.inv(T_gt_idx)
                        # Transform points
                        map_pc_ii[:, :3] = (T_sm[:3, :3] @ map_pc_ii[:, :3].T).T + T_sm[:3, 3]
                        # Transform normals
                        n_hg = torch.cat((map_pc_ii[:, 3:], torch.ones((map_pc_ii.shape[0], 1), dtype=self.float_type)), dim=1)
                        n_hg = (torch.linalg.inv(T_sm).T @ n_hg.T).T
                        map_pc_ii[:, 3:] = n_hg[:, :3]

                    map_pc_list.append(map_pc_ii)


                if num_samples > 0 and T_loc_gt.shape[0] >= num_samples:
                    break
            # Remove ground truth localization to map poses that do not have corresponding images
            if len(incomplete_loc_times) != 0:
                print('WARNING: Number of localization cartesian images does not match number of localization poses')
                # Remove ground truth localization to map poses that do not have corresponding images
                for time in incomplete_loc_times:
                    index = loc_times.index(time)
                    loc_times.pop(index)
                    map_times.pop(index)
                    T_gt.pop(index)

            #self.T_loc_init += T_gt_used
            self.loc_timestamps += loc_timestamps
            self.map_timestamps += map_timestamps

        self.fft_data = all_fft_data
        self.azimuths = all_azimuths
        self.az_timestamps = all_az_timestamps


        # Want to bind the range of polar data to fit within cartesian image
        polar_res = 0.0596

        # Precompute CFAR of fft data
        self.fft_cfar = cfar_mask(all_fft_data, polar_res, a_thresh=a_thresh, b_thresh=b_thresh, diff=False)

        # Extract pointcloud from fft data
        if gt_eye:
            T_scan_pc = T_loc_gt
        else:
            T_scan_pc = None
        # Note, we've already transformed the map pointcloud to the scan frame
        scan_pc_list = extract_pc(self.fft_cfar, polar_res, all_azimuths, all_az_timestamps,
                                  T_ab=None, diff=False)

        # Form batch of pointclouds and initial guesses for batching
        config_path = '../external/dICP/config/dICP_config.yaml'
        temp_ICP = ICP(config_path=config_path)
        scan_pc_batch, map_pc_batch, _, _ = temp_ICP.batch_size_handling(scan_pc_list, map_pc_list)
        # Want to bind the range of polar data to fit within cartesian image
        cart_res = 0.2384
        cart_pixel_width = 640
        # Compute the range (m) captured by pixels in cartesian scan
        if (cart_pixel_width % 2) == 0:
            cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_res
        else:
            cart_min_range = cart_pixel_width / 2 * cart_res
        scan_pc_x_outrange = torch.abs(scan_pc_batch[:, :, 0]) > cart_min_range
        scan_pc_y_outrange = torch.abs(scan_pc_batch[:, :, 1]) > cart_min_range
        scan_pc_outrange = scan_pc_x_outrange | scan_pc_y_outrange
        scan_pc_batch[scan_pc_outrange] = 0.0

        self.scan_pc = scan_pc_batch
        self.map_pc = map_pc_batch
        self.T_loc_init = T_loc_init
        self.T_loc_gt = T_loc_gt

        # Save fft data statistics for potential normalization
        self.fft_mean = torch.mean(self.fft_data)
        self.fft_std = torch.std(self.fft_data)
        self.fft_max = torch.max(self.fft_data)
        self.fft_min = torch.min(self.fft_data)

        # Assert that the number of all lists are the same
        assert len(self.T_loc_gt) \
             == self.fft_data.shape[0] == self.azimuths.shape[0] \
             == self.az_timestamps.shape[0] == len(self.loc_timestamps)

    def __len__(self):
        return self.T_loc_gt.shape[0]

    def __getitem__(self, index):
        # Load in fft data
        fft_data = self.fft_data[index]
        azimuths = self.azimuths[index]
        az_timestamps = self.az_timestamps[index]
        fft_cfar = self.fft_cfar[index]
        scan_pc = self.scan_pc[index]
        map_pc = self.map_pc[index]
        T_init = self.T_loc_init[index]

        # Load in timestamps
        loc_timestamp = self.loc_timestamps[index]
        map_timestamp = self.map_timestamps[index]

        # Load in ground truth localization to map pose
        T_ml_gt = self.T_loc_gt[index]

        #loc_data = {'pc' : loc_pc, 'timestamp' : loc_timestamp}
        loc_data = {'pc': scan_pc, 'timestamp' : loc_timestamp,
                    'fft_data' : fft_data, 'azimuths' : azimuths, 'az_timestamps' : az_timestamps,
                    'fft_cfar' : fft_cfar}
        map_data = {'pc': map_pc, 'timestamp' : map_timestamp}
        T_data = {'T_ml_init' : T_init, 'T_ml_gt' : T_ml_gt}

        return {'loc_data': loc_data, 'map_data': map_data, 'transforms': T_data}