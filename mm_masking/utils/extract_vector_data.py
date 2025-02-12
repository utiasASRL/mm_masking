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
from dICP.ICP import ICP
np.set_printoptions(suppress=True)


def main():
    print("Starting testing script!", flush=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # Define datasets

    test_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2020-12-04-14-00'],
                      ["boreas-2020-11-26-13-58", 'boreas-2021-01-26-10-59'],
                        ["boreas-2020-11-26-13-58", 'boreas-2021-02-09-12-55'],
                        ["boreas-2020-11-26-13-58", 'boreas-2021-03-09-14-23'],
                        ["boreas-2020-11-26-13-58", 'boreas-2021-06-29-18-53'],
                        ["boreas-2020-11-26-13-58", 'boreas-2021-09-08-21-00'],
                        ["boreas-2021-04-08-12-44", "boreas-2020-12-01-13-26"]]


    # Define resource parameters
    device = "cuda"
    num_workers = 4
    iterator_batch_size = 8

    # Set random seeds to reproduce results
    # Note, setting random = True in the dataloader generates identical ICs for each
    # test traj. We just want idenitcal ICs across different runs of this script
    np.random.seed(99)
    torch.manual_seed(99)

    params = {
        "device": device,

        # Dataset params
        "num_train": -1,
        "num_val": -1,
        "augment": False,
        "random": False,
        "float_type": torch.float32,
        "use_gt": True,
        "pos_std": 0.0,             # Standard deviation of position initial guess
        "rot_std": 0.0,             # Standard deviation of rotation initial guess
        "gt_eye": False,             # Should ground truth transform be identity?
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
        "num_epochs": 100,
        "learning_rate": 1e-5,#5*1e-5,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.05,   # Dropout rate, set 0 for no dropout
        "batch_norm": False, # True or false for batch norm
        "init_weights": True, # True or false for manually initializing weights
        "clip_value": 0.0, # Value to clip gradients at, set 0 for no clipping
        "a_thresh": 1.0, # Threshold for CFAR
        "b_thresh": 0.09, # Threshold for CFAR

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
        "max_iter_inference": 200, # Maximum number of iterations for icp during inference

        # Model setup
        "network_input_type": "cartesian", # Options are "cartesian" and "polar", what the network takes in
        "network_output_type": "cartesian", # Options are "cartesian" and "polar"
        "binary_inference": False, # Options are True and False, whether the mask is binary or not during inference
        "low_cap_weight": 0.0, # Minimum weight for a point, only during inference
        "norm_weights": True, # Options are True and False, whether to normalize weights to always have max weight of 1
        # Choose inputs to network
        "fft_input": True,
        "cfar_input": False,
        "range_input": False,
    }

    # Load in test dataset, extracting PC in process
    for test_pair in test_loc_pairs:
        print("Loading test dataset from pair: ", test_pair)
        tic = time.time()
        test_dataset = ICPWeightDataset(loc_pairs=[test_pair], params=params, dataset_type='test')
        toc = time.time()
        print("Time to load test dataset: ", toc-tic)
        print("Number of test examples: ", len(test_dataset))
            
if __name__ == "__main__":
    main()