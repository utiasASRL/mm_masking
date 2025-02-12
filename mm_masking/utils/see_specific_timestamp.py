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
from radar_utils import radar_polar_to_cartesian_diff, extract_bev_from_pts, visualize_pointcloud_over_img
import os.path as osp
from datetime import datetime
from pytz import timezone
import zipfile
import shutil
import pandas as pd
import resource
from dICP.ICP import ICP
np.set_printoptions(suppress=True)
import cv2


def main():
    print("Starting testing script!", flush=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    run_id = "MMICP-635"
    best_epoch = 20
    print("Run ID: ", run_id)
    checkpoints_storage_dir = osp.join('results', 'checkpoints')
    checkpoint_dir = osp.join(checkpoints_storage_dir, run_id)

    if not osp.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print("hello")
    best_policy_path = osp.join(checkpoint_dir, f'epoch_{best_epoch}.pt')


    # Define test datasets
    #test_loc_pairs = [["boreas-2021-04-08-12-44", "boreas-2020-12-01-13-26"]]
    test_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    #test_loc_pairs = [["boreas-2021-04-08-12-44", 'boreas-2021-01-15-12-17']]
    #loc_timestamp = 1610731096453439
    #loc_timestamp = 1610731064701159
    loc_timestamp = 1607108460251588
    #loc_timestamp = 1607108624253928
    loc_timestamp = 1607108460001584

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
        "num_train": 1,
        "num_val": 5,
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
        "icp_dim": 3,
        "use_gumbel": False,
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
        "max_iter_inference": 25, # Maximum number of iterations for icp during inference

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

    # Load in test dataset
    for test_pair in test_loc_pairs:
        print("Loading test dataset from pair: ", test_pair)

        tic = time.time()
        test_dataset = ICPWeightDataset(loc_pairs=[test_pair], params=params, dataset_type='test')
        toc = time.time()
        print("Time to load test dataset: ", toc-tic)
        print("Number of test examples: ", len(test_dataset))

        data_stamp = test_dataset.get_item_from_loc_timestamp(loc_timestamp)

        loc_data = data_stamp['loc_data']
        map_data = data_stamp['map_data']
        T_data = data_stamp['transforms']

        print(loc_data['timestamp'])
        print(map_data['timestamp'])
        print(data_stamp['index'])

        print("Subsampling validation dataset to ", 1, " example")
        test_dataset = torch.utils.data.Subset(test_dataset, [data_stamp['index']])
        print("Number of subsampled validation examples: ", len(test_dataset))
        test_iterator = DataLoader(test_dataset, batch_size=iterator_batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        print("Test dataloader created")

        T_init = T_data['T_map_ls_init']

        #     0.999778 -0.00388375   0.0207187     10.4881
        # -0.00313314   -0.999342  -0.0361385    -2.61613
        # 0.0208454   0.0360656   -0.999132     2.01721
        #         0           0           0           1

        T_init = torch.tensor([[0.999778, -0.00388375, 0.0207187, 10.4881],
                                [-0.00313314, -0.999342, -0.0361385, -2.61613],
                                [0.0208454, 0.0360656, -0.999132, 2.01721],
                                [0, 0, 0, 1]]).to(device=params["device"])

        loc_pc = loc_data['filtered_pc'].to(device=params["device"])
        raw_pc = loc_data['raw_pc'].to(device=params["device"])
        map_pc = map_data['pc'].to(device=params["device"])

        # Print average value in loc_pc
        print("Average value in loc_pc: ", torch.mean(loc_pc))
        print("Average value in map_pc: ", torch.mean(map_pc))
        print("Average value in loc_pc_raw: ", torch.mean(raw_pc))

        # loc_pc = loc_pc[:1303]
        # map_pc = map_pc[:2697]

        # Run ICP
        config_path = '../external/dICP/config/dICP_config.yaml'
        ICP_alg = ICP(icp_type='pt2pt', config_path=config_path, differentiable=False, max_iterations=25, tolerance=1e-4)
        loss_fn = {"name": "cauchy", "metric": 1.0}
        print("Manual ICP")
        print("Raw pc: ", raw_pc[:3,:3])
        icp_result = ICP_alg.icp(loc_pc, map_pc,
                                    T_init=T_init,
                                    trim_dist=5.0, loss_fn=loss_fn, dim=3)

        map_pts = map_data['pc'].cpu().detach().numpy()
        map_pts = map_pts[map_pts[:,0] != 1000.0]
        curr_filt_pts = loc_data['filtered_pc'].cpu().detach().numpy()
        curr_filt_pts = curr_filt_pts[curr_filt_pts[:,0] != 0]
        curr_pts_gt = T_data['T_map_ls_gt'].cpu().detach().numpy() @ np.concatenate([curr_filt_pts.T, np.ones((1, curr_filt_pts.shape[0]))], axis=0)
        curr_pts_est = (icp_result['T'].cpu().detach().numpy() @ np.concatenate([curr_filt_pts.T, np.ones((1, curr_filt_pts.shape[0]))], axis=0)).squeeze()
        curr_raw_pts = raw_pc.cpu().detach().numpy()
        #curr_raw_pts = curr_raw_pts[curr_raw_pts[:,0] != 0]

        # Read in mask from file
        mask_file = osp.join('/home/dli/proj_repos/mm_masking/data/masks/MMICP-635/boreas_2020_12_04_14_00', str(loc_timestamp) + '.png')
        file_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # Initialize policy
        policy = LearnICPWeightPolicy(params = params)
        policy = policy.to(device=params["device"])
        print("Policy created")

        batch = next(iter(test_iterator))
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        policy.load_state_dict(torch.load(best_policy_path, weights_only=True))
        policy.eval()
        batch_T_init = T_init.unsqueeze(0).to(device=params["device"])
        T_est_mask, weight_mask, _, _ = policy(batch_scan, batch_map, batch_T_init, override_mask=torch.tensor(file_mask).to(device=params["device"]).unsqueeze(0))



        # override_mask = torch.ones_like(weight_mask)
        # T_est_ones, _ , _, _ = policy(batch_scan, batch_map, batch_T_init, override_mask=override_mask)

        print("Mask shape: ", file_mask.shape)
        print("Mask first 5 values: ", file_mask[0,:5])
        print("Estimated mask shape: ", weight_mask.shape)
        print("Estimated mask first 5 values: ", weight_mask[0, 0,:5])


        print("Size of full query: ", loc_pc.shape)
        print("Size of query: ", np.sum(loc_pc.cpu().detach().numpy()[:,0] != 0))
        print("Size of map: ", np.sum(map_data['pc'].cpu().detach().numpy()[:,0] != 1000.0))
        print("T initial: \n", T_init.cpu().detach().numpy())
        print("T gt: \n", T_data['T_map_ls_gt'].cpu().detach().numpy())
        print("ICP result: \n", icp_result['T'].cpu().detach().numpy())
        print(icp_result['stats'])
        print("T est mask: \n", T_est_mask.cpu().cpu().detach().numpy())
        # print("T_est_ones: \n", T_est_ones.cpu().cpu().detach().numpy())

        plt.figure(figsize=(15,15))
        plt.scatter(map_pts[:,0], map_pts[:,1], s=1.0, c='red')
        plt.scatter(curr_filt_pts[:,0], curr_filt_pts[:,1], s=0.5, c='blue')
        plt.scatter(curr_raw_pts[:,0], curr_raw_pts[:,1], s=0.5, c='green')
        # plt.scatter(curr_pts_gt[0,:], curr_pts_gt[1,:], s=0.5, c='green')
        # plt.scatter(curr_pts_est[0,:], curr_pts_est[1,:], s=0.5, c='purple')
        # plt.ylim([-80, 80])
        # plt.xlim([-80, 80])
        plt.savefig('align_2.png')
        plt.close()

        # # Also save the raw and filtered pointclouds overlaid on top of fft image
        # fft_img = loc_data['fft_data']

        # fig = plt.figure(figsize=(15,15))
        # visualize_pointcloud_over_img(fft_img, raw_pc, start_fig=False)
        # plt.savefig('raw_overlay.png', bbox_inches='tight')

        # fig = plt.figure(figsize=(15,15))
        # visualize_pointcloud_over_img(fft_img, loc_pc, start_fig=False)
        # plt.savefig('filt_overlay.png', bbox_inches='tight')


            
if __name__ == "__main__":
    main()