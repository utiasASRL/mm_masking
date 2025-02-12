import argparse
import torch
from icp_weight_dataset import ICPWeightDataset
from torch.utils.data import Dataset, DataLoader
from icp_weight_policy import LearnICPWeightPolicy
import time
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pylgmath import se3op
import os
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
import torch.nn as nn
from radar_utils import radar_polar_to_cartesian_diff, extract_bev_from_pts
import os.path as osp
from datetime import datetime
from pytz import timezone
import zipfile
import shutil
import pandas as pd
import resource
import cv2


def extract_checkpoints(run, checkpoint_dir):
    val_seq = run["sys/id"].fetch()
    print("Loading checkpoints from run: ", val_seq)

    # Extract number of epochs saved in run
    stored_losses = run["training/epoch/loss"].fetch_values()
    num_remote_epochs = len(stored_losses)
    print("Number of epochs saved remotely: ", num_remote_epochs)

    # Check if checkpoints have already been downloaded
    if osp.exists(checkpoint_dir):
        num_downloaded = len(os.listdir(checkpoint_dir))
        # Check if "best_policy.pt" is among the downloaded checkpoints
        if osp.exists(osp.join(checkpoint_dir, "best_policy.pt")):
            num_downloaded -= 1
    else:
        num_downloaded = 0

    if num_downloaded < num_remote_epochs:
        # Download checkpoints
        temp_dir = osp.join(checkpoint_dir, "temp_checkpoints")
        if not osp.exists(temp_dir):
            os.makedirs(temp_dir)
        run["training/model_checkpoints"].download(temp_dir)
        print("Downloaded checkpoints")
        # Unzip checkpoints
        zip_path = osp.join(temp_dir, "model_checkpoints.zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # This extracts to temp_checkpoints/SOME_DIRECTORY_STRUCTURE
            zip_ref.extractall(temp_dir)
            # We now want to move all .pt files from SOME_DIRECTORY_STRUCTURE to checkpoint_dir
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".pt"):
                        shutil.copy(osp.join(root, file), checkpoint_dir)
            # Finally, delete temp directory and all contents
            shutil.rmtree(temp_dir)
        print("Extracted checkpoints")
        num_checkpoints = len(os.listdir(checkpoint_dir))
        if osp.exists(osp.join(checkpoint_dir, "best_policy.pt")):
                    num_checkpoints -= 1

        if num_checkpoints == num_remote_epochs:
            print("Number of checkpoints downloaded does not match number of epochs saved remotely")
            print("Number of checkpoints downloaded: ", num_checkpoints, ", number of epochs saved remotely: ", num_remote_epochs)
    else:
        print("Checkpoints already downloaded")
        num_checkpoints = num_downloaded

    print("Number of checkpoints available: ", num_checkpoints)
    return num_checkpoints

def main():
    print("Starting mask gen script!", flush=True)

    run_id = "MMICP-631"
    best_epoch = 40
    run = neptune.init_run(with_id=run_id,
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODZkYzJmNS1iMWY3LTRlMWYtYWNjYy0zNTFhOWJjYjNiMTQifQ==",
                            project='asrl/mm-icp',
                            mode="read-only")

    print("Run ID: ", run["sys/id"].fetch())
    checkpoints_storage_dir = osp.join('results', 'checkpoints')
    checkpoint_dir = osp.join(checkpoints_storage_dir, run["sys/id"].fetch())

    if not osp.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    extract_checkpoints(run, checkpoint_dir)

    best_policy_path = osp.join(checkpoint_dir, f'epoch_{best_epoch}.pt')

    # Define test datasets
    test_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2020-12-04-14-00'],
                    #   ["boreas-2020-11-26-13-58", 'boreas-2021-01-26-10-59'],
                    #     ["boreas-2020-11-26-13-58", 'boreas-2021-02-09-12-55'],
                    #     ["boreas-2020-11-26-13-58", 'boreas-2021-03-09-14-23'],
                    #     ["boreas-2020-11-26-13-58", 'boreas-2021-06-29-18-53'],
                    #     ["boreas-2020-11-26-13-58", 'boreas-2021-09-08-21-00']
                        ]

    #test_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2020-12-04-14-00'],
    #                    ["boreas-2020-11-26-13-58", 'boreas-2021-01-26-10-59'],
    #                  ["boreas-2020-11-26-13-58", 'boreas-2021-02-09-12-55']]


    data_dir = '../data'
    dataset_dir = osp.join(data_dir, 'vtr_data')
    mask_dir = osp.join(data_dir, 'masks', run["sys/id"].fetch())
    if not osp.exists(mask_dir): os.makedirs(mask_dir)

    device = "cuda"
    num_workers = 8
    iterator_batch_size = 32

    params = {
        "device": device,

        # Dataset params
        "num_train": -1,
        "num_val": -1,
        "augment": False,
        "random": False,
        "float_type": torch.float32,
        "use_gt": False,
        "pos_std": 0.5,             # Standard deviation of position initial guess
        "rot_std": 0.1,             # Standard deviation of rotation initial guess
        "gt_eye": True,             # Should ground truth transform be identity?
        "map_sensor": "lidar",
        "loc_sensor": "radar",
        "log_transform": False,      # True or false for log transform of fft data
        "normalize": ["minmax"],  # Options are "minmax", "standardize", and none
                                    # happens after log transform if log transform is true

        # Iterator params
        "batch_size": iterator_batch_size,
        "shuffle": False,

        # Training params
        "icp_type": "pt2pt", # Options are "pt2pt" and "pt2pl"
        "icp_dim": 3, # Dimension of points used for icp
        "num_epochs": 100,
        "learning_rate": 1e-5,#5*1e-5,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.05,   # Dropout rate, set 0 for no dropout
        "batch_norm": False, # True or false for batch norm
        "init_weights": True, # True or false for manually initializing weights
        "clip_value": 0.0, # Value to clip gradients at, set 0 for no clipping
        "a_thresh": 1.0, # Threshold for CFAR
        "b_thresh": 0.09, # Threshold for CFAR
        "use_gumbel": False, # True or false for using gumbel softmax

        # Choose weights for loss function
        "loss_icp_rot_weight": 0.0, # Weight for icp rotation error loss
        "loss_icp_trans_weight": 0.0, # Weight for icp translation error loss
        "loss_fft_mask_weight": 0.0, # Weight for fft mask loss
        "loss_map_pts_mask_weight": 1.0, # Weight for map pts mask loss
        "loss_cfar_mask_weight": 0.0, # Weight for cfar mask loss
        "num_pts_weight": 0.0, # Weight for number of points loss
        "optimizer": "adam", # Options are "adam" and "sgd"
        "icp_loss_only_iter": -1, # Number of iterations after which to only use icp loss
        "max_iter": 10, # Maximum number of iterations for icp
        "max_iter_inference": 40, # Maximum number of iterations for icp during inference

        # Model setup
        "network_input_type": "cartesian", # Options are "cartesian" and "polar", what the network takes in
        "network_output_type": "cartesian", # Options are "cartesian" and "polar"
        "binary_inference": False, # Options are True and False, whether the mask is binary or not during inference
        "norm_weights": True, # Options are True and False, whether to normalize weights to always have max weight of 1
        # Choose inputs to network
        "fft_input": True,
        "cfar_input": False,
        "range_input": False,
    }

    # Initialize policy
    policy = LearnICPWeightPolicy(params = params)
    policy = policy.to(device=params["device"])
    print("Policy created")

    policy.load_state_dict(torch.load(best_policy_path, weights_only=True))
    policy.eval()

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    for test_pair in test_loc_pairs:
        radar_img_dir = osp.join(dataset_dir, test_pair[1], 'radar', 'cart')

        # For mask dir, we want to replace the dashes with underscores in the seq name
        seq_name_under = test_pair[1].replace('-', '_')
        radar_mask_dir = osp.join(mask_dir, seq_name_under)
        if not osp.exists(radar_mask_dir): os.makedirs(radar_mask_dir)

        # Load in all files that end with png
        for file in os.listdir(radar_img_dir):
            if file.endswith(".png"):
                stamp = file.split('.')[0]
                print(stamp)

                file_dir = osp.join(radar_img_dir, file)
                loc_radar_img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
                fft_data = torch.tensor(loc_radar_img, dtype=torch.float32)/255.0

                # Create dummy data for all other inputs
                fft_cfar = torch.zeros_like(fft_data)
                scan_pc_raw = torch.ones((1,5,3))
                filtered_pc = torch.ones((1,5,3))
                map_pc = torch.ones((1,10,6))

                batch_scan = {'fft_data': fft_data.unsqueeze(0), 'fft_cfar': fft_cfar.unsqueeze(0), 'raw_pc': scan_pc_raw, 'filtered_pc':filtered_pc, 'timestamp': 1}
                batch_map = {'pc': map_pc, 'timestamp': 1}
                batch_T_init = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(device)

                mask = policy(batch_scan, batch_map, batch_T_init, mask_only=True)

                mask = mask[0].detach().cpu().numpy()

                # Zero out top corner
                mask[0, 0] = 0.0
                mask[0, 1] = 1.0
                mask[1, 0] = 1.0

                # Write mask to file
                mask_path = osp.join(radar_mask_dir, stamp + '.png')
                plt.imsave(mask_path, mask, cmap='gray')

if __name__ == "__main__":
    main()