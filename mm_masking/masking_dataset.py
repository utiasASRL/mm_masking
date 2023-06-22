import cv2
import torch
import numpy as np
import os.path as osp
from pyboreas.utils.odometry import read_traj_file2
import matplotlib.pyplot as plt
from pylgmath import se3op, Transformation
from RadarUtils import load_radar, extract_pc, extract_pc2, load_pc_from_file, pol_2_cart

class MaskingDataset():

    def __init__(self, gt_data_dir, pc_dir, radar_dir, loc_pairs, sensor, random=False, num_samples=-1, size_pc=-1, verbose=False, float_type=torch.float64, use_gt=False):
        self.loc_pairs = loc_pairs
        self.size_pc = size_pc
        self.float_type = float_type

        if not random:
            np.random.seed(99)
            torch.manual_seed(99)

        # Loop through loc_pairs and load in ground truth loc poses
        self.T_loc_gt = []
        self.T_loc_init = []
        self.loc_radar_paths = []
        self.loc_pc_paths = []
        self.loc_timestamps = []
        self.map_cart_paths = []
        self.map_pc_paths = []
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
            loc_radar_paths = []
            loc_pc_paths = []
            map_cart_paths = []
            map_pc_paths = []
            loc_timestamps = []
            map_timestamps = []
            incomplete_loc_times = []
            T_gt_used = []
            T_init_used = []
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
                    # Load in cartesian images
                    loc_radar_paths.append(loc_radar_path)
                    # Map is submap so no cartesian image
                    #map_cart_paths.append(map_cart_path)
                    # Load in point cloud binaries
                    loc_pc_paths.append(loc_pc_path)
                    map_pc_paths.append(map_pc_path)
                    # Save timestamp
                    loc_timestamps.append(loc_time)
                    map_timestamps.append(map_times[idx])

                    # Save ground truth localization to map pose
                    T_gt_idx = torch.tensor(T_gt[idx], dtype=float_type)
                    T_gt_used.append(T_gt_idx)
                    # Also, generate random perturbation to ground truth pose
                    if use_gt:
                        xi_rand = torch.zeros((6,1), dtype=float_type)
                    else:
                        xi_rand = 0.1*torch.randn((6,1), dtype=float_type)
                    xi_rand[2:5] = 0.0
                    T_rand = Transformation(xi_ab=xi_rand)
                    T_init_idx = T_rand.matrix() @ T_gt[idx]
                    T_init_idx = torch.tensor(T_init_idx, dtype=float_type)
                    T_init_used.append(T_init_idx)
                
                if num_samples > 0 and len(loc_radar_paths) >= num_samples:
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

            self.loc_radar_paths += loc_radar_paths
            self.loc_pc_paths += loc_pc_paths
            self.map_cart_paths += map_cart_paths
            self.map_pc_paths += map_pc_paths
            self.T_loc_gt += T_gt_used
            self.T_loc_init += T_init_used
            #self.T_loc_init += T_gt_used
            self.loc_timestamps += loc_timestamps
            self.map_timestamps += map_timestamps

        # Assert that the number of all lists are the same
        assert len(self.T_loc_gt) == len(self.loc_radar_paths) \
             == len(self.loc_pc_paths) == len(self.map_pc_paths)

    def __len__(self):
        return len(self.T_loc_gt)

    def __getitem__(self, index):
        # Load in localization and map cartesian image
        loc_radar_img = cv2.imread(self.loc_radar_paths[index], cv2.IMREAD_GRAYSCALE)
        loc_radar_mat = np.asarray(loc_radar_img)

        fft_data, azimuths, az_timestamps = load_radar(loc_radar_mat)
        fft_data = torch.tensor(fft_data, dtype=self.float_type)
        azimuths = torch.tensor(azimuths, dtype=self.float_type)
        az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type)

        #map_cart = cv2.imread(self.map_cart_paths[index])
        # Load in localization and map point cloud
        # The point cloud contains float32 values in the following format:
        # [x, y, z, x_norm, y_norm, z_norm]

        # Load in timestamps
        loc_timestamp = self.loc_timestamps[index]
        map_timestamp = self.map_timestamps[index]

        """
        loc_pc = np.fromfile(self.loc_pc_paths[index], dtype=np.float32)
        loc_pc = loc_pc.reshape((len(loc_pc) // 6, 6))
        map_pc = np.fromfile(self.map_pc_paths[index], dtype=np.float32)
        map_pc = map_pc.reshape((len(map_pc) // 6, 6))
        

        # Downsample all points to num_samples
        if self.size_pc > 0:
            # If less than size_pc points, skip
            if len(loc_pc) < self.size_pc or len(map_pc) < self.size_pc:
                msg = 'Not enough points in point cloud to be downsamples' + '\n'
                msg += 'Timestamps: ' + str(loc_timestamp) + ', ' + str(map_timestamp) + '\n'
                msg += 'Number of points in localization point cloud: ' + str(len(loc_pc)) + '\n'
                msg += 'Number of points in map point cloud: ' + str(len(map_pc)) + '\n'
                msg += 'Number of points requested: ' + str(self.size_pc)
                raise Exception(msg)
            down_idx_loc = np.linspace(0, len(loc_pc)-1, self.size_pc, dtype=int)
            down_idx_map = np.linspace(0, len(map_pc)-1, self.size_pc, dtype=int)
            loc_pc = loc_pc[down_idx_loc, :]
            map_pc = map_pc[down_idx_map, :]
        """

        # Load in ground truth localization to map pose
        T_ml_gt = self.T_loc_gt[index]
        T_ml_init = self.T_loc_init[index]

        #T_ml = np.linalg.inv(T_lm_gt)
        T_ml = T_ml_gt

        #print("Curr loc timestamp: " + str(loc_timestamp))
        #print("Curr map timestamp: " + str(map_timestamp))
        #print(T_ml)

        #self.visualize_pointclouds(loc_pc, map_pc, T_ml)

        # Save loc_cart
        #loc_cart = np.rot90(loc_cart, 3)
        #cv2.imwrite('loc_cart.png', loc_cart)
        
        #loc_data = {'cart' : loc_cart, 'pc' : loc_pc, 'timestamp' : loc_timestamp}
        #map_data = {'cart' : map_cart, 'pc' : map_pc, 'timestamp' : map_timestamp}

        #loc_data = {'pc' : loc_pc, 'timestamp' : loc_timestamp}
        loc_data = {'pc_path' : self.loc_pc_paths[index], 'timestamp' : loc_timestamp,
                    'fft_data' : fft_data, 'azimuths' : azimuths, 'az_timestamps' : az_timestamps}
        #loc_data = {'pc_path' : self.loc_pc_paths[index], 'radar_path': self.loc_radar_paths[index], 'timestamp' : loc_timestamp}
        map_data = {'pc_path' : self.map_pc_paths[index], 'timestamp' : map_timestamp}
        T_data = {'T_ml_init' : T_ml_init, 'T_ml_gt' : T_ml_gt}

        return {'loc_data': loc_data, 'map_data': map_data, 'transforms': T_data}
    
    def visualize_pointclouds(self, loc_pc, map_pc, T_ml):
        print(loc_pc.shape)
        print(map_pc.shape)
        plt.figure(figsize=(15,15))
        plt.scatter(loc_pc[:,0], loc_pc[:,1], s=0.5, c='green')
        plt.savefig('loc_vis.png')

        plt.figure(figsize=(15,15))
        plt.scatter(map_pc[:,0], map_pc[:,1], s=0.5, c='red')
        plt.savefig('map_vis.png')

        loc_pc[:, 0:3] = np.matmul(T_ml[:3, :3], loc_pc[:, 0:3].T).T + T_ml[:3, 3]
        
        plt.figure(figsize=(15,15))
        plt.scatter(loc_pc[:,0], loc_pc[:,1], s=0.5, c='blue')
        plt.scatter(map_pc[:,0], map_pc[:,1], s=0.5, c='red')
        plt.savefig('loc_align_vis.png')

    def visualize_pointcloud_old(self, pc, pc_color, ref_pc=None):
        # Save x and y coordinates of pointcloud to image
        # pc: (N, 3) array of pointcloud 
        # Returns: (H, W, 3) array of image with color values at x and y coordinates of pointcloud
        H = 1000
        W = 1000
        img = np.zeros((H, W, 3))
        # Map pointcloud to image bounds
        buffer = 125
        #max_x = np.max(pc[:, 0]) - buffer
        #min_x = np.min(pc[:, 0]) + buffer
        #max_y = np.max(pc[:, 1]) - buffer
        #min_y = np.min(pc[:, 1]) + buffer

        #pc[:, 0] = (pc[:, 0] - min_x) / (max_x - min_x) * W
        #pc[:, 1] = (pc[:, 1] - min_y) / (max_y - min_y) * H

        for i in range(pc.shape[0]):
            x = int(pc[i, 0])
            y = int(pc[i, 1])
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            # To improve visibility, plot x,y coordinate and the 4 surrounding pixels
            if pc_color == "green":
                img[y-1:y+1, x-1:x+1] = [0, 255, 0]
            elif pc_color == "blue":
                # Assign an azure blue colour
                img[y-1:y+1, x-1:x+1] = [255, 255, 0]
            elif pc_color == "red":
                img[y-1:y+1, x-1:x+1] = [0, 0, 255]
        
        # Rotate image 90 degrees counter-clockwise
        #img = np.rot90(img, 3)

        return img