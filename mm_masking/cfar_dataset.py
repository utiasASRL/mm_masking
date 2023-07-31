import cv2
import torch
import numpy as np
import os.path as osp
from pyboreas.utils.odometry import read_traj_file2
import matplotlib.pyplot as plt
from pylgmath import se3op, Transformation
from radar_utils import load_radar, cfar_mask

class CFARDataset():

    def __init__(self, gt_data_dir, radar_dir, loc_pairs, 
                 random=False, num_samples=-1, float_type=torch.float64,
                 log_transform=True, normalization_type="minmax",
                 use_gt=False):
        self.loc_pairs = loc_pairs
        self.float_type = float_type
        self.log_transform = log_transform
        self.normalization_type = normalization_type
        self.use_gt = use_gt

        if not random:
            np.random.seed(99)
            torch.manual_seed(99)

        # Loop through loc_pairs and load in ground truth loc poses
        self.loc_radar_paths = []
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
            for idx, loc_time in enumerate(loc_times):
                # Load in localization paths
                loc_radar_path = osp.join(radar_dir, loc_seq, str(loc_time) + '.png')
                if osp.exists(loc_radar_path):
                    # Load in cartesian images
                    loc_radar_paths.append(loc_radar_path)

                if num_samples > 0 and len(loc_radar_paths) >= num_samples:
                    break

            self.loc_radar_paths += loc_radar_paths

    def __len__(self):
        return len(self.loc_radar_paths)

    def __getitem__(self, index):
        # Load in localization and map cartesian image
        loc_radar_img = cv2.imread(self.loc_radar_paths[index], cv2.IMREAD_GRAYSCALE)
        loc_radar_mat = np.asarray(loc_radar_img)
        fft_data, _, _ = load_radar(loc_radar_mat)
        fft_data = torch.tensor(fft_data, dtype=self.float_type)
        gt_cfar_mask = cfar_mask(fft_data.unsqueeze(0), res, a_thresh=1.0, b_thresh=0.09, diff=False)
        #print(torch.max(fft_data), torch.min(fft_data), torch.mean(fft_data), torch.median(fft_data))
        if self.log_transform:
            print("using log")
            fft_data = torch.log(fft_data + 1e-6)
        
        if self.normalization_type == "minmax":
            print("using minmax")
            fft_min = torch.min(fft_data)
            fft_max = torch.max(fft_data)
            fft_data = (fft_data - fft_min) / (fft_max - fft_min)
        elif self.normalization_type == "standardize":
            print("using standardize")
            fft_data = fft_data - torch.mean(fft_data)
            fft_data = fft_data / torch.std(fft_data)

        #print(torch.max(fft_data), torch.min(fft_data), torch.mean(fft_data), torch.median(fft_data))

        #print(torch.max(fft_data), torch.min(fft_data), torch.mean(fft_data), torch.median(fft_data))

        if self.use_gt:
            print("using gt")
            return {'fft_data': gt_cfar_mask.squeeze(0), 'cfar_mask': gt_cfar_mask.squeeze(0)}
        return {'fft_data': fft_data, 'cfar_mask': gt_cfar_mask.squeeze(0)}
        #return {'fft_data': gt_cfar_mask.squeeze(0), 'cfar_mask': gt_cfar_mask.squeeze(0)}