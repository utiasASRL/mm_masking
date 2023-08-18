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
from radar_utils import radar_polar_to_cartesian_diff


scaler = torch.cuda.amp.GradScaler()

def train_policy(model, iterator, opt, loss_weights=[],
                 device='cpu', epoch=None,
                 icp_loss_only_iter=0, gt_eye=True):
    model.train()
    loss_hist = 0.0
    loss_comp_hist = []

    for i_batch, batch in enumerate(iterator):
        #print("Batch: ", i_batch)
        # Load in data
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        batch_T = batch['transforms']
        batch_T_init = batch_T['T_ml_init'].to(device)

        # Zero grad
        opt.zero_grad()

        #with torch.cuda.amp.autocast():
        T_pred, mask, num_non0 = model(batch_scan, batch_map, batch_T_init)
        del batch_map, batch_T_init

        # Compute loss
        batch_T_gt = batch_T['T_ml_gt'].to(device)
        loss, loss_comp = eval_training_loss(T_pred, mask, num_non0, batch_T_gt, batch_scan, model, loss_weights=loss_weights,
                                icp_loss_only_iter=icp_loss_only_iter, gt_eye=gt_eye, epoch=epoch)
        del batch_T_gt, mask, T_pred, batch_scan
        # Compute the derivatives
        #scaler.scale(loss).backward()
        loss.backward()

        # Take step
        #scaler.step(opt)
        #scaler.update()
        opt.step()
        
        loss_hist += loss.detach()
        loss_comp_hist.append(loss_comp)
        del loss
        torch.cuda.empty_cache()

    mean_loss = loss_hist/len(iterator)
    # Compute mean of each loss component
    mean_loss_comp = {}
    for key in loss_comp_hist[0].keys():
        mean_loss_comp[key] = sum(d[key] for d in loss_comp_hist) / len(loss_comp_hist)

    return mean_loss, mean_loss_comp

def validate_policy(model, iterator, gt_eye=True, device='cpu', binary=False,
                    neptune_run=None, epoch=None):
    model.eval()
    val_acc = torch.zeros((1,3), device=device)
    mean_num_pc = 0.0
    max_w = 0.0
    min_w = 1000.0
    mean_w = 0.0

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            #print("Batch: ", i_batch)
            # Load in data
            batch_scan = batch['loc_data']
            batch_map = batch['map_data']
            batch_T = batch['transforms']
            batch_T_gt = batch_T['T_ml_gt'].to(device)
            batch_T_init = batch_T['T_ml_init'].to(device)

            if neptune_run is not None:# and i_batch == 0:
                T_pred, mask, _ = model(batch_scan, batch_map, batch_T_init, binary=binary, neptune_run=neptune_run, epoch=epoch, batch_idx=i_batch)
            else:
                T_pred, mask, _ = model(batch_scan, batch_map, batch_T_init, binary=binary)

            mean_num_pc += model.mean_num_pts

            if model.max_w > max_w:
                max_w = model.max_w
            if model.min_w < min_w:
                min_w = model.min_w

            mean_w += model.mean_w

            # Compute validation loss
            val_acc_i = eval_validation_loss(T_pred, batch_T_gt, gt_eye=gt_eye)
            val_acc += val_acc_i

            # Save first mask from this batch to neptune with name "learned_mask_#i_batch"
            if neptune_run is not None and epoch is not None and i_batch <= 10:
                #if model.network_output_type == 'polar':
                #    mask_cart = radar_polar_to_cartesian_diff(mask.detach().cpu(), batch_scan['azimuths'], model.res)
                #    mask_0 = mask_cart[0].numpy()
                #else:
                mask_0 = mask[0].detach().cpu().numpy()
                fig = plt.figure()
                plt.imshow(mask_0, cmap='gray')
                plt.colorbar(location='top', shrink=0.5)
                neptune_run["learned_mask"].append(fig, name=("batch " + str(i_batch) + ", epoch " + str(epoch)))
                plt.close()

                # If pre-training evaluation, save the 0'th raw scan for reference
                if epoch == -1:
                    fft_data = batch_scan['fft_data']
                    cfar_data = batch_scan['fft_cfar']
                    #mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
                    #fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))
                    #bev_data = radar_polar_to_cartesian_diff(fft_data, batch_scan['azimuths'], model.res)
                    bev_data = fft_data
                    #bev_fft_mask_data = radar_polar_to_cartesian_diff(fft_mask, batch_scan['azimuths'], model.res)
                    mean_bev_scan = torch.mean(bev_data, dim=(1,2), keepdim=True)
                    bev_fft_mask_data = torch.where(bev_data > 3.0*mean_bev_scan, torch.ones_like(bev_data), torch.zeros_like(bev_data))
                    scan_0 = fft_data[0].numpy()
                    bev_scan_0 = bev_data[0].numpy()
                    cfar_data_0 = cfar_data[0].numpy()
                    bev_fft_mask_0 = bev_fft_mask_data[0].numpy()

                    #fig = plt.figure()
                    #plt.imshow(scan_0, cmap='gray')
                    #plt.colorbar(location='top', shrink=0.5)
                    #neptune_run["raw_scan"].append(fig, name=("Polar Scan 0, batch " + str(i_batch)))
                    #plt.close()

                    fig = plt.figure()
                    plt.imshow(bev_scan_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("BEV Scan 0, batch " + str(i_batch)))
                    plt.close()

                    fig = plt.figure()
                    plt.imshow(cfar_data_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("CFAR Mask 0, batch " + str(i_batch)))
                    plt.close()

                    fig = plt.figure()
                    plt.imshow(bev_fft_mask_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("FFT Mask 0, batch " + str(i_batch)))
                    plt.close()

        mean_num_pc /= len(iterator)
        mean_w /= len(iterator)
        #print("Mean number of point clouds: ", mean_num_pc)

        val_acc /= len(iterator)

    return val_acc, mean_num_pc, mean_w, max_w, min_w

def eval_training_loss(T_pred, mask, num_non0, batch_T_gt, batch_scan, model, loss_weights=[],
                       icp_loss_only_iter=0, gt_eye=True, epoch=0):
    mask_criterion = torch.nn.BCELoss()
    loss_rot = torch.zeros(1, device=T_pred.device)
    loss_trans = torch.zeros(1, device=T_pred.device)
    loss_fft = torch.zeros(1, device=T_pred.device)
    loss_mask_pts = torch.zeros(1, dtype=T_pred.dtype, device=T_pred.device)
    loss_cfar = torch.zeros(1, dtype=T_pred.dtype, device=T_pred.device)
    loss_num_pts = torch.zeros(1, dtype=T_pred.dtype, device=T_pred.device)

    # Compute ICP loss
    if loss_weights['icp_rot'] > 0.0 or loss_weights['icp_trans'] > 0.0:
        # Compute ICP loss
        if gt_eye:
            xi_wedge = T_pred - torch.eye(4, dtype=T_pred.dtype, device=T_pred.device)
        else:
            xi_wedge = torch.matmul(T_pred, torch.inverse(batch_T_gt)) - torch.eye(4, dtype=T_pred.dtype, device=T_pred.device)
        # Extract xi components
        xi_r = xi_wedge[:, 0:2, 3]
        xi_theta = xi_wedge[:, 1, 0].unsqueeze(-1)
        loss_rot = torch.norm(xi_theta, dim=1).mean()
        loss_trans = torch.norm(xi_r, dim=1).mean()
    
    if icp_loss_only_iter <= 0 or (icp_loss_only_iter > 0 and epoch < icp_loss_only_iter) or \
        loss_weights['icp'] <= 0:
        # Compute FFT mask loss
        if loss_weights['fft'] > 0.0:
            # Find mean value of each fft azimuth
            fft_data = batch_scan['fft_data'].to(mask.device)
            #mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
            mean_azimuth = torch.mean(fft_data, dim=(1,2), keepdim=True)
            fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))

            #if model.network_output_type == "cartesian":
            #    azimuths = batch_scan['azimuths'].to(mask.device)
            #    fft_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)

            loss_fft = mask_criterion(mask, fft_mask)
        # Compute CFAR mask loss
        # CFAR image is loaded in polar or cartesian already
        if loss_weights['cfar'] > 0.0:
            fft_cfar = batch_scan['fft_cfar'].to(mask.device)
            loss_cfar = mask_criterion(mask, fft_cfar)

        # Compute mask pts loss
        if loss_weights['mask_pts'] > 0.0:
            temp = 1

        # Compute loss associated with number of points
        # This penalizes the network for ignoring too many points from those available
        if loss_weights['num_pts'] > 0.0:
            loss_num_pts = model.mean_all_pts - num_non0

    loss = loss_weights['icp_rot']*loss_rot + loss_weights['icp_trans']*loss_trans\
        + loss_weights['fft']*loss_fft + loss_weights['mask_pts']*loss_mask_pts\
        + loss_weights['cfar']*loss_cfar + loss_weights['num_pts']*loss_num_pts

    #print("Loss: ", loss.item(), " ICP: ", loss_weights['icp']*loss_icp, " FFT: ", loss_weights['fft']*loss_fft, \
    #    " Mask: ", loss_weights['mask_pts']*loss_mask_pts, " CFAR: ", loss_weights['cfar']*loss_cfar, \
    #    " Num pts: ", loss_weights['num_pts']*loss_num_pts)

    L_rot = (loss_weights['icp_rot']*loss_rot).detach()
    L_trans = (loss_weights['icp_trans']*loss_trans).detach()
    L_fft = (loss_weights['fft']*loss_fft).detach()
    L_mask_pts = (loss_weights['mask_pts']*loss_mask_pts).detach()
    L_cfar = (loss_weights['cfar']*loss_cfar).detach()
    L_num_pts = (loss_weights['num_pts']*loss_num_pts).detach()

    loss_components = {"rot": L_rot, "trans": L_trans, "fft": L_fft, "mask_pts": L_mask_pts,
                       "cfar": L_cfar, "num_pts": L_num_pts}
    
    return loss, loss_components

def eval_validation_loss(T_pred, batch_T_gt, gt_eye=True):
    
    # Compute ICP error
    if gt_eye:
        xi_wedge = T_pred - torch.eye(4, dtype=T_pred.dtype, device=T_pred.device)
    else:
        xi_wedge = torch.matmul(T_pred, torch.inverse(batch_T_gt)) - torch.eye(4, dtype=T_pred.dtype, device=T_pred.device)
    # Extract xi components
    xi_r = xi_wedge[:, 0:2, 3]
    xi_theta = xi_wedge[:, 1, 0].unsqueeze(-1)
    xi_stack = torch.cat((xi_theta, xi_r), dim=1)
    norm_err = torch.norm(xi_stack, dim=1).mean()
    rot_err = torch.norm(xi_theta, dim=1).mean()
    trans_err = torch.norm(xi_r, dim=1).mean()

    # Stack errors together
    tot_err = torch.hstack((norm_err, rot_err, trans_err))

    return tot_err

def generate_baseline(model, iterator, baseline_type="train", device='cpu',
                      loss_weights={'icp': 1.0, 'fft': 0.0, 'mask_pts': 0.0, 'cfar': 0.0},
                      binary=False, gt_eye=True):
    # We don't actually want to take steps, but want to see what the train baseline
    # is without anything being turned off/without taking additional ICP steps
    if baseline_type == "train":
        model.train()
    elif baseline_type == "val":
        model.eval()
    loss_init_hist = []
    loss_ones_hist = []

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            #print("Batch: ", i_batch)
            # Load in data
            batch_scan = batch['loc_data']
            batch_map = batch['map_data']
            batch_T = batch['transforms']
            batch_T_gt = batch_T['T_ml_gt'].to(device)
            batch_T_init = batch_T['T_ml_init'].to(device)

            # Form baseline masks
            fft_data = batch_scan['fft_data'].to(device)
            if loss_weights['cfar'] > 0.0:
                # CFAR image is loaded in polar or cartesian already
                fft_cfar = batch_scan['fft_cfar'].to(device)
                ones_mask = fft_cfar
            elif loss_weights['fft'] > 0.0:
                # Find mean value of each fft azimuth
                fft_data = batch_scan['fft_data'].to(device)
                #mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
                mean_azimuth = torch.mean(fft_data, dim=(1,2), keepdim=True)
                fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))
                #if model.network_input_type == "cartesian":
                #    azimuths = batch_scan['azimuths'].to(device)
                #    fft_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)
                ones_mask = fft_mask
            else:
                ones_mask = torch.ones_like(fft_data)

            # Compute training baselines
            T_pred_ones, mask_ones, num_non0 = model(batch_scan, batch_map, batch_T_init,
                                            binary=binary, override_mask=ones_mask)

            # Compute loss
            if baseline_type == "train":
                loss_init, _ = eval_training_loss(batch_T_init, mask_ones, num_non0, batch_T_gt, batch_scan, model, loss_weights=loss_weights, gt_eye=gt_eye)
                loss_ones, _ = eval_training_loss(T_pred_ones, mask_ones, num_non0, batch_T_gt, batch_scan, model, loss_weights=loss_weights, gt_eye=gt_eye)
                loss_init = loss_init.detach().cpu().numpy()
                loss_ones = loss_ones.detach().cpu().numpy()
            elif baseline_type == "val":
                loss_init = eval_validation_loss(batch_T_init, batch_T_gt, gt_eye=gt_eye)
                loss_ones = eval_validation_loss(T_pred_ones, batch_T_gt, gt_eye=gt_eye)
                loss_init = loss_init[0].detach().cpu().numpy()
                loss_ones = loss_ones[0].detach().cpu().numpy()
            # Save loss for full error
            loss_init_hist.append(loss_init)
            loss_ones_hist.append(loss_ones)

    del batch_scan, batch_map, batch_T, batch_T_gt, batch_T_init, T_pred_ones, mask_ones
    torch.cuda.empty_cache()

    # Compute mean losses
    mean_loss_ones = np.mean(loss_ones_hist)
    mean_loss_init = np.mean(loss_init_hist)

    return mean_loss_init, mean_loss_ones

def main(args):
    run = neptune.init_run(
        project="asrl/mm-icp",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjljOGQ1ZC1lNDE3LTQxYTQtOGNmMS1kMWY0NDcyY2IyODQifQ==",
        mode="async"
    )

    params = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # Dataset params
        "num_train": 16,
        "num_val": 16,
        "augment_factor": 1,        # Factor by which to augment dataset, this will
                                    # increase the number of train samples
        "random": False,
        "float_type": torch.float32,
        "use_gt": False,
        "pos_std": 2.0,             # Standard deviation of position initial guess
        "rot_std": 0.3,             # Standard deviation of rotation initial guess
        "gt_eye": True,             # Should ground truth transform be identity?
        "map_sensor": "lidar",
        "loc_sensor": "radar",
        "log_transform": False,      # True or false for log transform of fft data
        "normalize": ["minmax"],  # Options are "minmax", "standardize", and none
                                    # happens after log transform if log transform is true

        # Iterator params
        "batch_size": 8,
        "shuffle": False,

        # Training params
        "icp_type": "pt2pt", # Options are "pt2pt" and "pt2pl"
        "num_epochs": 1000,
        "learning_rate": 1e-5,#5*1e-5,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.0,   # Dropout rate, set 0 for no dropout
        "batch_norm": False, # True or false for batch norm
        "init_weights": True, # True or false for manually initializing weights
        "clip_value": 0.0, # Value to clip gradients at, set 0 for no clipping
        "a_thresh": 1.0, # Threshold for CFAR
        "b_thresh": 0.09, # Threshold for CFAR

        # Choose weights for loss function
        "loss_icp_rot_weight": 1.0, # Weight for icp rotation error loss
        "loss_icp_trans_weight": 1.0, # Weight for icp translation error loss
        "loss_fft_mask_weight": 0.3, # Weight for fft mask loss
        "loss_map_pts_mask_weight": 0.0, # Weight for map pts mask loss
        "loss_cfar_mask_weight": 0.0, # Weight for cfar mask loss
        "num_pts_weight": 0.0, # Weight for number of points loss
        "optimizer": "adam", # Options are "adam" and "sgd"
        "icp_loss_only_iter": -1, # Number of iterations after which to only use icp loss
        "max_iter": 8, # Maximum number of iterations for icp

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

    print("Using device: ", params['device'])

    loss_weights = {"icp_rot": params["loss_icp_rot_weight"], "icp_trans": params["loss_icp_trans_weight"],
                    "fft": params["loss_fft_mask_weight"],
                    "mask_pts": params["loss_fft_mask_weight"], "cfar": params["loss_cfar_mask_weight"],
                    "num_pts": params["num_pts_weight"]}

    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2020-12-04-14-00']]
    val_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2020-12-04-14-00']]
    """
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-01-13-26"],
                      ["boreas-2020-11-26-13-58", "boreas-2021-03-30-14-23"],
                      ["boreas-2020-11-26-13-58", "boreas-2021-04-29-15-55"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2021-01-26-10-59"]]
    
    train_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2021-06-17-17-52'],
                      ["boreas-2020-11-26-13-58", 'boreas-2021-03-02-13-38'],
                      ["boreas-2020-11-26-13-58", 'boreas-2021-09-07-09-35']]

    val_loc_pairs = [["boreas-2020-11-26-13-58", 'boreas-2021-05-06-13-19']]
    """
    tic = time.time()
    train_dataset = ICPWeightDataset(loc_pairs=train_loc_pairs, params=params, dataset_type='train')
    toc = time.time()
    print("Time to load train dataset: ", toc-tic)
    tic = time.time()
    test_dataset = ICPWeightDataset(loc_pairs=val_loc_pairs, params=params, dataset_type='test')
    toc = time.time()
    print("Time to load test dataset: ", toc-tic)
    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(test_dataset))

    #torch.autograd.set_detect_anomaly(True)

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=4, drop_last=True)
    validation_iterator = DataLoader(test_dataset, batch_size=2*params["batch_size"], shuffle=False, num_workers=4, drop_last=True)
    print("Dataloader created")

    # Initialize policy
    policy = LearnICPWeightPolicy(params = params)
    policy = policy.to(device=params["device"])

    if params["optimizer"] == "adam":
        opt = torch.optim.Adam(policy.parameters(), lr=params["learning_rate"])
    elif params["optimizer"] == "sgd":
        opt = torch.optim.SGD(policy.parameters(), lr=params["learning_rate"], nesterov=True, momentum=1.0)

    print("Policy and optimizer created")

    npt_logger = NeptuneLogger(
        run=run,
        model=policy,
    )
    run[npt_logger.base_namespace]["parameters"] = stringify_unsupported(
        params
    )

    # Form result directory
    result_dir = 'results/' + 'mask' + '/learn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_naming = result_dir + '/' + params["icp_type"]

    # Evaluate baselines for training and evaluation
    train_init_baseline, train_ones_baseline = generate_baseline(policy, training_iterator, baseline_type="train",
                                                    device=params["device"], binary=False,
                                                    loss_weights=loss_weights, gt_eye=params["gt_eye"])
    val_init_baseline, val_ones_baseline = generate_baseline(policy, validation_iterator, baseline_type="val", device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])

    print("Training init baseline: ", train_init_baseline)
    print("Training ones baseline: ", train_ones_baseline)
    print("Validation init baseline: ", val_init_baseline)
    print("Validation ones baseline: ", val_ones_baseline)

    # Train policy
    loss_hist = []
    # Eval policy before training
    print("Evaluating policy before training")
    avg_norm, _, _, _, _  = validate_policy(policy, validation_iterator, device=params["device"],
                                         binary=params["binary_inference"], gt_eye=params["gt_eye"],
                                         neptune_run=run, epoch=-1)
    #avg_norm = 1000
    # Compute best norm from total error
    best_norm = avg_norm[0, 0]

    print("Norm before training: ", avg_norm[0, 0])
    for epoch in range(params["num_epochs"]):
        tic_epoch = time.time()
        print ('EPOCH ', epoch)

        # Train the driving policy
        if epoch % 5 == 0 or epoch == params["num_epochs"] - 1 or epoch == 0:
            neptune_run = run
        else:
            neptune_run = None
        tic = time.time()
        mean_loss, mean_loss_comp = train_policy(policy, training_iterator, opt, loss_weights, device=params["device"],
                                 epoch=epoch, icp_loss_only_iter=params["icp_loss_only_iter"], gt_eye=params["gt_eye"])
        toc = time.time()
        epoch_train_time = toc-tic
        avg_sample_train_time = epoch_train_time/len(train_dataset)
        loss_hist.append(mean_loss)

        # Validate the driving policy
        print("Validating")
        tic = time.time()
        avg_norm, mean_num_pc, mean_w, max_w, min_w = validate_policy(policy, validation_iterator, neptune_run=neptune_run, epoch=epoch,
                                   device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
        toc = time.time()
        epoch_val_time = toc-tic
        avg_sample_val_time = epoch_val_time/len(test_dataset)
        if avg_norm[0, 0] < best_norm or epoch == 0:
            print("Saving best policy")
            best_norm = avg_norm[0, 0]
            torch.save(policy.state_dict(), result_naming + '_best_policy.pt')

        print("Average norm: ", avg_norm[0, 0])
        print("Best norm: ", best_norm)

        toc_epoch = time.time()
        epoch_time = toc_epoch - tic_epoch
        #print("Epoch time: ", epoch_time)

        #scheduler.step()
        # Log loss
        run[npt_logger.base_namespace]["epoch/loss"].append(mean_loss)
        run[npt_logger.base_namespace]["epoch/loss_rot"].append(mean_loss_comp["rot"])
        run[npt_logger.base_namespace]["epoch/loss_trans"].append(mean_loss_comp["trans"])
        run[npt_logger.base_namespace]["epoch/loss_fft"].append(mean_loss_comp["fft"])
        run[npt_logger.base_namespace]["epoch/loss_mask_pts"].append(mean_loss_comp["mask_pts"])
        run[npt_logger.base_namespace]["epoch/loss_cfar"].append(mean_loss_comp["cfar"])
        run[npt_logger.base_namespace]["epoch/loss_num_pts"].append(mean_loss_comp["num_pts"])

        # Log accuracy
        run[npt_logger.base_namespace]["epoch/acc"].append(avg_norm[0,0])
        run[npt_logger.base_namespace]["epoch/acc_rot"].append(avg_norm[0,1])
        run[npt_logger.base_namespace]["epoch/acc_trans"].append(avg_norm[0,2])
        run[npt_logger.base_namespace]["epoch/mean_num_pc"].append(mean_num_pc)
        run[npt_logger.base_namespace]["epoch/mean_w"].append(mean_w)
        run[npt_logger.base_namespace]["epoch/max_w"].append(max_w)
        run[npt_logger.base_namespace]["epoch/min_w"].append(min_w)
        run[npt_logger.base_namespace]["epoch/avg_sample_train_time"].append(avg_sample_train_time)
        run[npt_logger.base_namespace]["epoch/avg_sample_val_time"].append(avg_sample_val_time)
        run[npt_logger.base_namespace]["epoch/epoch_train_time"].append(epoch_train_time)
        run[npt_logger.base_namespace]["epoch/epoch_val_time"].append(epoch_val_time)
        run[npt_logger.base_namespace]["epoch/epoch_time"].append(epoch_time)

        # Save baseline for reference
        run[npt_logger.base_namespace]["epoch/train_init_baseline"].append(train_init_baseline)
        run[npt_logger.base_namespace]["epoch/train_ones_baseline"].append(train_ones_baseline)
        run[npt_logger.base_namespace]["epoch/val_init_baseline"].append(val_init_baseline)
        run[npt_logger.base_namespace]["epoch/val_ones_baseline"].append(val_ones_baseline)

    # Do final validation using the best policy
    policy.load_state_dict(torch.load(result_naming + '_best_policy.pt'))
    avg_norm, _, _, _, _ = validate_policy(policy, validation_iterator, neptune_run=neptune_run, epoch=epoch,
                                   device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
    print("Best average norm: ", avg_norm[0,0])
    
    run.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--radar_dir", help="directory of training data", default='../data/radar')

    args = parser.parse_args()

    main(args)