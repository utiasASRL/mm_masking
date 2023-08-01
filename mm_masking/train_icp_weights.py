import argparse
import torch
from icp_weight_dataset import ICPWeightDataset
from torch.utils.data import Dataset, DataLoader
from icp_weight_policy import LearnICPWeightPolicy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pylgmath import se3op
import os
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
import torch.nn as nn
from radar_utils import radar_polar_to_cartesian_diff


def train_policy(model, iterator, opt, loss_weights={'icp': 1.0, 'fft': 0.0, 'mask_pts': 0.0, 'cfar': 0.0, 'num_pts': 0.0},
                 device='cpu', neptune_run=None, epoch=None, clip_value=0.0,
                 icp_loss_only_iter=0, gt_eye=True):
    model.train()
    loss_hist = []
    encoder_norm = []
    decoder_norm = []
    final_layer_norm = []

    for i_batch, batch in enumerate(iterator):
        print("Batch: ", i_batch)
        # Load in data
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        batch_T = batch['transforms']
        batch_T_init = batch_T['T_ml_init'].to(device)

        # Zero grad
        opt.zero_grad()
        T_pred, mask, num_non0 = model(batch_scan, batch_map, batch_T_init)
        del batch_map, batch_T_init

        # Compute loss
        batch_T_gt = batch_T['T_ml_gt'].to(device)
        loss = eval_training_loss(T_pred, mask, num_non0, batch_T_gt, batch_scan, model, loss_weights=loss_weights,
                                  icp_loss_only_iter=icp_loss_only_iter, gt_eye=True, epoch=epoch)
        del batch_T_gt, mask, T_pred, batch_scan
        # Compute the derivatives
        loss.backward()

        if clip_value > 0.0:
            #nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

        # Store the norm of the gradients for each stage of unet
        for module in model.encoder:
            layer_norm = 0.0
            for layer in module.parameters():
                layer_norm += layer.grad.norm().detach().cpu().numpy()
            encoder_norm.append(layer_norm)
        for module in model.decoder:
            layer_norm = 0.0
            for layer in module.parameters():
                layer_norm += layer.grad.norm().detach().cpu().numpy()
            decoder_norm.append(layer_norm)
        for layer in model.final_layer:
            layer_norm = 0.0
            for layer in module.parameters():
                layer_norm += layer.grad.norm().detach().cpu().numpy()
            final_layer_norm.append(layer_norm)
        
        # Take step
        opt.step()
        
        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)
        del loss
        torch.cuda.empty_cache()

    mean_loss = np.mean(loss_hist)
    batch_grad_norm = {'encoder': encoder_norm, 'decoder': decoder_norm, 'final_layer': final_layer_norm}
    return mean_loss, batch_grad_norm


def validate_policy(model, iterator, gt_eye=True, device='cpu', verbose=False, binary=False,
                    neptune_run=None, epoch=None):
    model.eval()
    val_loss_list = []
    mean_num_pc = 0.0
    max_w = 0.0
    min_w = 1000.0
    mean_w = 0.0

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
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
            val_loss = eval_validation_loss(T_pred, batch_T_gt, gt_eye=gt_eye)

            # Compute RMSE
            val_loss_list.append(val_loss)

            # Save first mask from this batch to neptune with name "learned_mask_#i_batch"
            if neptune_run is not None and epoch is not None:
                if model.network_output == 'polar':
                    mask_cart = radar_polar_to_cartesian_diff(mask.detach().cpu(), batch_scan['azimuths'], model.res)
                    mask_0 = mask_cart[0].numpy()
                else:
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
                    bev_data = radar_polar_to_cartesian_diff(fft_data, batch_scan['azimuths'], model.res)
                    bev_cfar_data = radar_polar_to_cartesian_diff(cfar_data, batch_scan['azimuths'], model.res)
                    scan_0 = fft_data[0].numpy()
                    bev_scan_0 = bev_data[0].numpy()
                    bev_cfar_0 = bev_cfar_data[0].numpy()

                    fig = plt.figure()
                    plt.imshow(scan_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("Polar Scan 0, batch " + str(i_batch)))
                    plt.close()

                    fig = plt.figure()
                    plt.imshow(bev_scan_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("BEV Scan 0, batch " + str(i_batch)))
                    plt.close()

                    fig = plt.figure()
                    plt.imshow(bev_cfar_0, cmap='gray')
                    plt.colorbar(location='top', shrink=0.5)
                    neptune_run["raw_scan"].append(fig, name=("CFAR Mask 0, batch " + str(i_batch)))
                    plt.close()

        mean_num_pc /= len(iterator)
        mean_w /= len(iterator)
        print("Mean number of point clouds: ", mean_num_pc)

        val_loss = np.mean(val_loss_list)

    return val_loss, mean_num_pc, mean_w, max_w, min_w

def eval_training_loss(T_pred, mask, num_non0, batch_T_gt, batch_scan, model, loss_weights={'icp': 1.0, 'fft': 0.0, 'mask_pts': 0.0, 'cfar': 0.0, 'num_pts': 0.0},
                       icp_loss_only_iter=0, gt_eye=True, epoch=0):
    mask_criterion = torch.nn.BCELoss()
    loss_icp = 0.0
    loss_fft = 0.0
    loss_mask_pts = 0.0
    loss_cfar = 0.0
    loss_num_pts = 0.0

    # Compute ICP loss
    if loss_weights['icp'] > 0.0:
        # Compute ICP loss
        if gt_eye:
            xi_wedge = T_pred - torch.eye(4, dtype=torch.float64, device=T_pred.device)
        else:
            xi_wedge = torch.matmul(T_pred, torch.inverse(batch_T_gt)) - torch.eye(4, dtype=torch.float64, device=T_pred.device)
        # Extract xi components
        xi_r = xi_wedge[:, 0:2, 3]
        xi_theta = xi_wedge[:, 1, 0].unsqueeze(-1)
        # Stack the xi_theta and xi_r
        xi = torch.cat((xi_theta, xi_r), dim=1)
        loss_icp = torch.norm(xi, dim=1).mean()
    
    if icp_loss_only_iter <= 0 or (icp_loss_only_iter > 0 and epoch < icp_loss_only_iter) or \
        loss_weights['icp'] <= 0:
        # Compute FFT mask loss
        if loss_weights['fft'] > 0.0:
            # Find mean value of each fft azimuth
            fft_data = batch_scan['fft_data'].to(mask.device)
            mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
            fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))

            if model.network_input == "cartesian":
                azimuths = batch_scan['azimuths'].to(mask.device)
                fft_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)

            loss_fft = mask_criterion(mask, fft_mask)
        # Compute CFAR mask loss
        if loss_weights['cfar'] > 0.0:
            fft_cfar = batch_scan['fft_cfar'].to(mask.device)
            if model.network_input == "cartesian":
                azimuths = batch_scan['azimuths'].to(mask.device)
                fft_cfar = radar_polar_to_cartesian_diff(fft_cfar, azimuths, model.res)
            
            loss_cfar = mask_criterion(mask, fft_cfar)

        # Compute mask pts loss
        if loss_weights['mask_pts'] > 0.0:
            temp = 1

        # Compute loss associated with number of points
        # This penalizes the network for ignoring too many points from those available
        if loss_weights['num_pts'] > 0.0:
            loss_num_pts = model.mean_all_pts - num_non0

    loss = loss_weights['icp']*loss_icp + loss_weights['fft']*loss_fft + \
        loss_weights['mask_pts']*loss_mask_pts + loss_weights['cfar']*loss_cfar + \
        loss_weights['num_pts']*loss_num_pts

    print("Loss: ", loss.item(), " ICP: ", loss_weights['icp']*loss_icp, " FFT: ", loss_weights['fft']*loss_fft, \
        " Mask: ", loss_weights['mask_pts']*loss_mask_pts, " CFAR: ", loss_weights['cfar']*loss_cfar, \
        " Num pts: ", loss_weights['num_pts']*loss_num_pts)

    return loss

def eval_validation_loss(T_pred, batch_T_gt, gt_eye=True):
    err = np.zeros((T_pred.shape[0], 6))
    norm_err = np.zeros((T_pred.shape[0], 1))
    for jj in range(T_pred.shape[0]):
        if gt_eye:
            Err = T_pred[jj].detach().cpu().numpy()
        else:
            Err = T_pred[jj].detach().cpu().numpy() @ np.linalg.inv(batch_T_gt[jj].detach().cpu().numpy())
        err[jj] = se3op.tran2vec(Err).flatten()
        norm_err[jj] = np.linalg.norm(err[jj])

    return norm_err

def generate_baseline(model, iterator, baseline_type="train", device='cpu',
                      loss_weights={'icp': 1.0, 'fft': 0.0, 'mask_pts': 0.0, 'cfar': 0.0},
                      binary=False, gt_eye=True):
    model.eval()
    loss_init_hist = []
    loss_ones_hist = []

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            # Load in data
            batch_scan = batch['loc_data']
            batch_map = batch['map_data']
            batch_T = batch['transforms']
            batch_T_gt = batch_T['T_ml_gt'].to(device)
            batch_T_init = batch_T['T_ml_init'].to(device)

            # Form baseline masks
            fft_data = batch_scan['fft_data'].to(device)
            if loss_weights['cfar'] > 0.0:
                fft_cfar = batch_scan['fft_cfar'].to(device)
                if model.network_input == "cartesian":
                    azimuths = batch_scan['azimuths'].to(device)
                    fft_cfar = radar_polar_to_cartesian_diff(fft_cfar, azimuths, model.res)
                ones_mask = fft_cfar
            elif loss_weights['fft'] > 0.0:
                # Find mean value of each fft azimuth
                fft_data = batch_scan['fft_data'].to(device)
                mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
                fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))

                if model.network_input == "cartesian":
                    azimuths = batch_scan['azimuths'].to(device)
                    fft_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)
                ones_mask = fft_mask
            else:
                ones_mask = torch.ones_like(fft_data)

            # Compute training baselines
            T_pred_ones, mask_ones, num_non0 = model(batch_scan, batch_map, batch_T_init,
                                            binary=binary, override_mask=ones_mask)

            # Compute loss
            if baseline_type == "train":
                loss_init = eval_training_loss(batch_T_init, mask_ones, num_non0, batch_T_gt, batch_scan, model, gt_eye=gt_eye).detach().cpu().numpy()
                loss_ones = eval_training_loss(T_pred_ones, mask_ones, num_non0, batch_T_gt, batch_scan, model, loss_weights=loss_weights, gt_eye=gt_eye).detach().cpu().numpy()
            elif baseline_type == "val":
                loss_init = eval_validation_loss(batch_T_init, batch_T_gt, gt_eye=gt_eye)
                loss_ones = eval_validation_loss(T_pred_ones, batch_T_gt, gt_eye=gt_eye)

            # Save loss
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
        "num_train": 1,
        "num_test": 1,
        "random": False,
        "float_type": torch.float64,
        "use_gt": False,
        "pos_std": 1.5,             # Standard deviation of position initial guess
        "rot_std": 0.2,             # Standard deviation of rotation initial guess
        "gt_eye": True,             # Should ground truth transform be identity?
        "map_sensor": "radar",
        "loc_sensor": "radar",
        "log_transform": False,      # True or false for log transform of fft data
        "normalize": "none",  # Options are "minmax", "standardize", and none
                                    # happens after log transform if log transform is true

        # Iterator params
        "batch_size": 1,
        "shuffle": True,

        # Training params
        "icp_type": "pt2pt", # Options are "pt2pt" and "pt2pl"
        "num_epochs": 1000,
        "learning_rate": 1e-3,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.01,   # Dropout rate, set 0 for no dropout
        "batch_norm": False, # True or false for batch norm
        "init_weights": True, # True or false for manually initializing weights
        "clip_value": 0.0, # Value to clip gradients at, set 0 for no clipping
        "a_thresh": 0.7, # Threshold for CFAR
        "b_thresh": 0.09, # Threshold for CFAR
        # Choose weights for loss function
        "loss_icp_weight": 1.0, # Weight for icp loss
        "loss_fft_mask_weight": 0.0, # Weight for fft mask loss
        "loss_map_pts_mask_weight": 0.0, # Weight for map pts mask loss
        "loss_cfar_mask_weight": 0.5, # Weight for cfar mask loss
        "num_pts_weight": 0.001, # Weight for number of points loss
        "optimizer": "adam", # Options are "adam" and "sgd"
        "icp_loss_only_iter": -1, # Number of iterations after which to only use icp loss
        "max_iter": 8, # Maximum number of iterations for icp

        # Model setup
        "network_input": "cartesian", # Options are "cartesian" and "polar", what the network takes in
        "network_output": "cartesian", # Options are "cartesian" and "polar"
        "binary_inference": False, # Options are True and False, whether the mask is binary or not during inference
    }
    torch.set_default_dtype(params["float_type"])

    loss_weights = {"icp": params["loss_icp_weight"], "fft": params["loss_fft_mask_weight"],
                    "mask_pts": params["loss_fft_mask_weight"], "cfar": params["loss_cfar_mask_weight"],
                    "num_pts": params["num_pts_weight"]}

    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

    train_dataset = ICPWeightDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=train_loc_pairs,
                                        sensor=params["map_sensor"],
                                        random=params["random"],
                                        num_samples=params["num_train"],
                                        float_type=params["float_type"],
                                        use_gt=params["use_gt"],
                                        gt_eye=params["gt_eye"],
                                        pos_std=params["pos_std"],
                                        rot_std=params["rot_std"],
                                        a_thresh=params["a_thresh"],
                                        b_thresh=params["b_thresh"])
    test_dataset = ICPWeightDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=val_loc_pairs,
                                        sensor=params["map_sensor"],
                                        random=params["random"],
                                        num_samples=params["num_test"],
                                        float_type=params["float_type"],
                                        use_gt=params["use_gt"],
                                        gt_eye=params["gt_eye"],
                                        pos_std=params["pos_std"],
                                        rot_std=params["rot_std"],
                                        a_thresh=params["a_thresh"],
                                        b_thresh=params["b_thresh"])

    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(test_dataset))

    #torch.autograd.set_detect_anomaly(True)

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=4)
    validation_iterator = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)
    print("Dataloader created")

    # Initialize policy
    policy = LearnICPWeightPolicy(icp_type=params["icp_type"], 
                             network_input=params["network_input"],
                             network_output=params["network_output"],
                             leaky=params["leaky"], dropout=params["dropout"],
                             batch_norm=params["batch_norm"],
                             float_type=params["float_type"], device=params["device"],
                             init_weights=params["init_weights"],
                             normalize_type=params["normalize"],
                             log_transform=params["log_transform"],
                             fft_mean=train_dataset.fft_mean,
                             fft_std=train_dataset.fft_std,
                             fft_max=train_dataset.fft_max,
                             fft_min=train_dataset.fft_min,
                             a_threshold=params["a_thresh"],
                             b_threshold=params["b_thresh"],
                             icp_weight=loss_weights['icp'],
                             gt_eye=params["gt_eye"],
                             max_iter=params["max_iter"])
    policy = policy.to(device=params["device"])

    if params["optimizer"] == "adam":
        opt = torch.optim.Adam(policy.parameters(), lr=params["learning_rate"])
    elif params["optimizer"] == "sgd":
        opt = torch.optim.SGD(policy.parameters(), lr=params["learning_rate"], nesterov=True, momentum=1.0)

    print("Policy and optimizer created")

    # Set learning rate scheduler
    #scheduler = StepLR(opt, step_size=25, gamma=0.9)

    npt_logger = NeptuneLogger(
        run=run,
        model=policy,
        log_gradients=True,
        log_parameters=True,
        log_freq=1,
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
    val_hist = []
    # Eval policy before training
    print("Evaluating policy before training")
    avg_norm, _, _, _, _  = validate_policy(policy, validation_iterator, device=params["device"],
                                         binary=params["binary_inference"], gt_eye=params["gt_eye"],
                                         neptune_run=run, epoch=-1)
    #avg_norm = 1000
    best_norm = avg_norm
    val_hist.append(avg_norm)

    print("Norm before training: ", avg_norm)
    for epoch in range(params["num_epochs"]):
        print ('EPOCH ', epoch)

        # Train the driving policy
        if epoch % 10 == 0 or epoch == params["num_epochs"] - 1 or epoch == 0:
            neptune_run = run
        else:
            neptune_run = None
        mean_loss, batch_grad_norm = train_policy(policy, training_iterator, opt, loss_weights, device=params["device"],
                                 clip_value=params["clip_value"], epoch=epoch,
                                 icp_loss_only_iter=params["icp_loss_only_iter"], gt_eye=params["gt_eye"])
        loss_hist.append(mean_loss)

        # Validate the driving policy
        print("Validating")
        avg_norm, mean_num_pc, mean_w, max_w, min_w = validate_policy(policy, validation_iterator, neptune_run=neptune_run, epoch=epoch,
                                   device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
        val_hist.append(avg_norm)

        if avg_norm < best_norm or epoch == 0:
            print("Saving best policy")
            best_norm = avg_norm
            torch.save(policy.state_dict(), result_naming + '_best_policy.pt')

        print("Average norm: ", avg_norm)
        print("Best norm: ", best_norm)

        #scheduler.step()
        run[npt_logger.base_namespace]["epoch/loss"].append(mean_loss.item())
        run[npt_logger.base_namespace]["epoch/acc"].append(avg_norm.item())
        run[npt_logger.base_namespace]["epoch/mean_num_pc"].append(mean_num_pc)
        run[npt_logger.base_namespace]["epoch/mean_w"].append(mean_w)
        run[npt_logger.base_namespace]["epoch/max_w"].append(max_w)
        run[npt_logger.base_namespace]["epoch/min_w"].append(min_w)
        run[npt_logger.base_namespace]["epoch/encoder_grad_norm"].append(np.mean(batch_grad_norm["encoder"]))
        run[npt_logger.base_namespace]["epoch/decoder_grad_norm"].append(np.mean(batch_grad_norm["decoder"]))
        run[npt_logger.base_namespace]["epoch/final_layer_grad_norm"].append(np.mean(batch_grad_norm["final_layer"]))

        # Save baseline for reference
        run[npt_logger.base_namespace]["epoch/train_init_baseline"].append(train_init_baseline.item())
        run[npt_logger.base_namespace]["epoch/train_ones_baseline"].append(train_ones_baseline.item())
        run[npt_logger.base_namespace]["epoch/val_init_baseline"].append(val_init_baseline.item())
        run[npt_logger.base_namespace]["epoch/val_ones_baseline"].append(val_ones_baseline.item())

    # Do final validation using the best policy
    policy.load_state_dict(torch.load(result_naming + '_best_policy.pt'))
    avg_norm, _, _, _, _ = validate_policy(policy, validation_iterator, neptune_run=neptune_run, epoch=epoch,
                                   device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
    print("Best average norm: ", avg_norm)
    
    run.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--radar_dir", help="directory of training data", default='../data/radar')

    args = parser.parse_args()

    main(args)