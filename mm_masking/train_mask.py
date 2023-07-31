import argparse
import torch
from masking_dataset import MaskingDataset
from torch.utils.data import Dataset, DataLoader
from masking_policy import LearnMaskPolicy
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


def train_policy(model, iterator, opt, loss_weights={'icp': 1.0, 'ones': 0.0, 'zeros': 0.0, 'fft': 0.0},
                 device='cpu', neptune_run=None, epoch=None, clip_value=0.0,
                 icp_loss_only_iter=0, gt_eye=True):
    model.train()
    loss_hist = []
    encoder_norm = []
    decoder_norm = []
    final_layer_norm = []
    mean_num_pc = 0.0

    for i_batch, batch in enumerate(iterator):
        print("Batch: ", i_batch)
        # Load in data
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        batch_T = batch['transforms']
        batch_T_gt = batch_T['T_ml_gt'].to(device)
        batch_T_init = batch_T['T_ml_init'].to(device)

        # Zero grad
        opt.zero_grad()
        T_pred, mask, num_pc = model(batch_scan, batch_map, batch_T_gt, batch_T_init)
        mean_num_pc += num_pc

        # Compute loss
        loss = eval_training_loss(T_pred, mask, batch_T_gt, batch_scan, model, 
                                  loss_weights=loss_weights, device=device, epoch=epoch,
                                  icp_loss_only_iter=icp_loss_only_iter, gt_eye=gt_eye)
        
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

    mean_loss = np.mean(loss_hist)
    batch_grad_norm = {'encoder': encoder_norm, 'decoder': decoder_norm, 'final_layer': final_layer_norm}
    mean_num_pc /= len(iterator)
    print("Mean number of point clouds: ", mean_num_pc)
    return mean_loss, batch_grad_norm, mean_num_pc

def eval_training_loss(T_pred, mask, batch_T_gt, batch_scan, model, loss_weights={'icp': 1.0, 'ones': 0.0, 'zeros': 0.0, 'fft': 0.0},
                 device='cpu', epoch=None, icp_loss_only_iter=0, gt_eye=True):
    mask_criterion = torch.nn.BCELoss()
    loss = 0.0
    loss_icp = 0.0
    loss_ones = 0.0
    loss_zeros = 0.0
    loss_fft = 0.0
    # Compute ICP loss
    if loss_weights['icp'] > 0.0:
        if gt_eye:
            xi_wedge = T_pred - torch.eye(4, dtype=torch.float64, device=device)
        else:
            xi_wedge = torch.matmul(T_pred, torch.inverse(batch_T_gt)) - torch.eye(4, dtype=torch.float64, device=device)
        # Extract xi components
        xi_r = xi_wedge[:, 0:2, 3]
        xi_theta = xi_wedge[:, 1, 0].unsqueeze(-1)
        # Stack the xi_theta and xi_r
        xi = torch.cat((xi_theta, xi_r), dim=1)
        loss_icp = torch.norm(xi, dim=1).mean()
    if icp_loss_only_iter <= 0 or (icp_loss_only_iter > 0 and epoch < icp_loss_only_iter) or \
        loss_weights['icp'] <= 0:
        if loss_weights['ones'] > 0.0:
            # Want the mask to be all 1s if possible
            true_mask = torch.ones_like(mask)
            loss_ones = mask_criterion(mask, true_mask)
        if loss_weights['zeros'] > 0.0:
            # Want the mask to be all 0s if possible
            true_mask = torch.zeros_like(mask)
            loss_zeros = mask_criterion(mask, true_mask)
        if loss_weights['fft'] > 0.0:
            # Find mean value of each fft azimuth
            fft_data = batch_scan['fft_data'].to(device)
            mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)
            fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))

            if model.network_input == "cartesian":
                azimuths = batch_scan['azimuths'].to(device)
                fft_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)

            loss_fft = mask_criterion(mask, fft_mask)

    loss += loss_weights['icp'] * loss_icp + loss_weights['ones'] * loss_ones + \
            loss_weights['zeros'] * loss_zeros + loss_weights['fft'] * loss_fft

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

def validate_policy(model, iterator, gt_eye=True, device='cpu', verbose=False, binary=False,
                    neptune_run=None, epoch=None):
    model.eval()
    val_loss_list = []

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            # Load in data
            batch_scan = batch['loc_data']
            batch_map = batch['map_data']
            batch_T = batch['transforms']
            batch_T_gt = batch_T['T_ml_gt'].to(device)
            batch_T_init = batch_T['T_ml_init'].to(device)

            T_pred, mask, _ = model(batch_scan, batch_map, batch_T_gt, batch_T_init, binary=binary)

            # Compute validation loss
            val_loss = eval_validation_loss(T_pred, batch_T_gt, gt_eye=gt_eye)

            # Compute RMSE
            val_loss_list.append(val_loss)

            # Save first mask from this batch to neptune with name "learned_mask_#i_batch"
            if neptune_run is not None and epoch is not None:
                fig = plt.figure()
                plt.imshow(mask[0].detach().cpu().numpy(), cmap='gray')
                plt.colorbar(location='top', shrink=0.5)
                neptune_run["learned_mask"].append(fig, name=("batch" + str(i_batch) + ", epoch " + str(epoch)))
                plt.close()

    return np.mean(val_loss_list)

def generate_baseline(model, iterator, baseline_type="train", neptune_run=None, device='cpu', binary=False,
                      loss_weights={'icp': 1.0, 'ones': 0.0, 'zeros': 0.0, 'fft': 0.0}, gt_eye=True):
    model.eval()
    loss_fft_hist = []
    loss_ones_hist = []
    loss_zeros_hist = []

    for i_batch, batch in enumerate(iterator):
        # Load in data
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        batch_T = batch['transforms']
        batch_T_gt = batch_T['T_ml_gt'].to(device)
        batch_T_init = batch_T['T_ml_init'].to(device)

        # Form baseline masks
        fft_data = batch_scan['fft_data'].to(device)
        mean_azimuth = torch.mean(fft_data, dim=2).unsqueeze(-1)

        fft_mask = torch.where(fft_data > 3.0*mean_azimuth, torch.ones_like(fft_data), torch.zeros_like(fft_data))
        ones_mask = torch.ones_like(fft_data)
        zeros_mask = torch.zeros_like(fft_data)

        # Compute training baselines
        T_pred_fft, mask_fft, _ = model(batch_scan, batch_map, batch_T_gt, batch_T_init, override_mask=fft_mask, binary=binary)
        T_pred_ones, mask_ones, _ = model(batch_scan, batch_map, batch_T_gt, batch_T_init, override_mask=ones_mask, binary=binary)
        _, mask_zeros, _ = model(batch_scan, batch_map, batch_T_gt, batch_T_init, override_mask=zeros_mask, binary=binary)
        T_pred_zeros = batch_T_init

        # Compute loss
        if baseline_type == "train":
            loss_fft = eval_training_loss(T_pred_fft, mask_fft, batch_T_gt, batch_scan, model, 
                                    loss_weights=loss_weights, device=device, gt_eye=gt_eye).detach().cpu().numpy()
            loss_ones = eval_training_loss(T_pred_ones, mask_ones, batch_T_gt, batch_scan, model,
                                        loss_weights=loss_weights, device=device, gt_eye=gt_eye).detach().cpu().numpy()
            loss_zeros = eval_training_loss(T_pred_zeros, mask_zeros, batch_T_gt, batch_scan, model,
                                        loss_weights=loss_weights, device=device, gt_eye=gt_eye).detach().cpu().numpy()
        elif baseline_type == "val":
            loss_fft = eval_validation_loss(T_pred_fft, batch_T_gt, gt_eye=gt_eye)
            loss_ones = eval_validation_loss(T_pred_ones, batch_T_gt, gt_eye=gt_eye)
            loss_zeros = eval_validation_loss(T_pred_zeros, batch_T_gt, gt_eye=gt_eye)

        # Save loss
        loss_fft_hist.append(loss_fft)
        loss_ones_hist.append(loss_ones)
        loss_zeros_hist.append(loss_zeros)

        # Save fft_mask for reference
        if neptune_run is not None:
            #fig = plt.figure()
            #plt.imshow(fft_mask[0].detach().cpu().numpy(), cmap='gray')
            #plt.colorbar(location='top', shrink=0.5)
            #neptune_run["fft_mask"].append(fig, name=("batch " + str(i_batch) + " fft supervisory mask"))
            #plt.close()
            
            azimuths = batch_scan['azimuths'].to(device)
            bev_mask = radar_polar_to_cartesian_diff(fft_mask, azimuths, model.res)

            fig = plt.figure()
            plt.imshow(bev_mask[0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(location='top', shrink=0.5)
            neptune_run["fft_mask"].append(fig, name=("batch " + str(i_batch) + " bev supervisory mask"))
            plt.close()

    # Compute mean losses
    mean_loss_fft = np.mean(loss_fft_hist)
    mean_loss_ones = np.mean(loss_ones_hist)
    mean_loss_zeros = np.mean(loss_zeros_hist)

    return mean_loss_fft, mean_loss_ones, mean_loss_zeros


def main(args):
    run = neptune.init_run(
        project="asrl/mm-masking",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjljOGQ1ZC1lNDE3LTQxYTQtOGNmMS1kMWY0NDcyY2IyODQifQ==",
        name="masking",
        mode="debug"
    )

    params = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # Dataset params
        "num_train": 1,
        "num_test": 1,
        "random": False,
        "float_type": torch.float64,
        "use_gt": False,
        "pos_std": 2.0,             # Standard deviation of position initial guess
        "rot_std": 0.3,             # Standard deviation of rotation initial guess
        "gt_eye": True,             # Should ground truth transform be identity?
        "map_sensor": "radar",
        "loc_sensor": "radar",
        "log_transform": False,      # True or false for log transform of fft data
        "normalize": "standardize",        # Options are "minmax", "standardize", and none
                                    # happens after log transform if log transform is true

        # Iterator params
        "batch_size": 16,
        "shuffle": True,

        # Training params
        "icp_type": "pt2pt", # Options are "pt2pt" and "pt2pl"
        "num_epochs": 1000,
        "learning_rate": 1e-5,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.0,   # Dropout rate, set 0 for no dropout
        "batch_norm": True, # True or false for batch norm
        "init_weights": True, # True or false for manually initializing weights
        "clip_value": 0.0, # Value to clip gradients at, set 0 for no clipping
        "a_thresh": 0.7, # Threshold for CFAR
        "b_thresh": 0.09, # Threshold for CFAR
        # Choose weights for loss function
        "loss_icp_weight": 1.0, # Weight for icp loss
        "loss_ones_mask_weight": 0.0, # Weight for ones mask loss
        "loss_zeros_mask_weight": 0.0, # Weight for zeros mask loss
        "loss_fft_mask_weight": 0.3, # Weight for fft mask loss
        "optimizer": "adam", # Options are "adam" and "sgd"
        "icp_loss_only_iter": -1, # Number of iterations after which to only use icp loss
        "max_iter": 8, # Maximum number of iterations for icp

        # Model setup
        "network_input": "cartesian", # Options are "cartesian" and "polar", what the network takes in
        "network_output": "polar", # Options are "cartesian" and "polar"
        "binary_inference": True, # Options are True and False, whether the mask is binary or not during inference
    }

    loss_weights = {"icp": params["loss_icp_weight"], "ones": params["loss_ones_mask_weight"], 
                    "zeros": params["loss_zeros_mask_weight"], "fft": params["loss_fft_mask_weight"]}

    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

    train_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
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
                                        rot_std=params["rot_std"])
    test_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
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
                                        rot_std=params["rot_std"])

    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(test_dataset))

    torch.set_default_dtype(params["float_type"])
    #torch.autograd.set_detect_anomaly(True)

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=4)
    validation_iterator = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=4)
    print("Dataloader created")

    # Initialize policy
    policy = LearnMaskPolicy(icp_type=params["icp_type"], 
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
    """
    base_train_fft, base_train_ones, base_train_zeros = generate_baseline(policy, training_iterator, baseline_type="train",
                                                    device=params["device"], binary=False,
                                                    loss_weights=loss_weights, gt_eye=params["gt_eye"])

    base_val_fft, base_val_ones, base_val_zeros = generate_baseline(policy, validation_iterator, baseline_type="val", neptune_run=run,
                                                    device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
    """
    # Train policy
    loss_hist = []
    val_hist = []
    # Eval policy before training
    #avg_norm = validate_policy(policy, validation_iterator, device=params["device"], binary=params["binary_inference"], gt_eye=params["gt_eye"])
    avg_norm = 1000
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
        mean_loss, batch_grad_norm, mean_num_pc = train_policy(policy, training_iterator,opt, loss_weights, device=params["device"],
                                 clip_value=params["clip_value"], epoch=epoch,
                                 icp_loss_only_iter=params["icp_loss_only_iter"], gt_eye=params["gt_eye"])
        loss_hist.append(mean_loss)

        # Validate the driving policy
        print("Validating")
        avg_norm = validate_policy(policy, validation_iterator,neptune_run=neptune_run, epoch=epoch,
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
        run[npt_logger.base_namespace]["epoch/encoder_grad_norm"].append(np.mean(batch_grad_norm["encoder"]))
        run[npt_logger.base_namespace]["epoch/decoder_grad_norm"].append(np.mean(batch_grad_norm["decoder"]))
        run[npt_logger.base_namespace]["epoch/final_layer_grad_norm"].append(np.mean(batch_grad_norm["final_layer"]))

        # Save baseline for reference
        run[npt_logger.base_namespace]["epoch/base_fft_loss"].append(base_train_fft.item())
        run[npt_logger.base_namespace]["epoch/base_ones_loss"].append(base_train_ones.item())
        run[npt_logger.base_namespace]["epoch/base_zeros_loss"].append(base_train_zeros.item())
        run[npt_logger.base_namespace]["epoch/base_val_fft_loss"].append(base_val_fft.item())
        run[npt_logger.base_namespace]["epoch/base_val_ones_loss"].append(base_val_ones.item())
        run[npt_logger.base_namespace]["epoch/base_val_zeros_loss"].append(base_val_zeros.item())

        """
        if epoch % 5 == 0 or epoch == params["num_epochs"] - 1:
            plt.clf()
            plt.plot(loss_hist)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(result_naming + '_loss.png')
            plt.clf()
            plt.plot(val_hist)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Error')
            plt.savefig(result_naming + '_val.png')
            plt.clf()
        """

        #npt_logger.save_checkpoint()

    # Do final validation using the best policy
    policy.load_state_dict(torch.load(result_naming + '_best_policy.pt'))
    avg_norm = validate_policy(policy, validation_iterator, device=params["device"], verbose=True, binary=params["binary_inference"], gt_eye=params["gt_eye"])
    print("Best average norm: ", avg_norm)
    
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--radar_dir", help="directory of training data", default='../data/radar')

    args = parser.parse_args()

    main(args)