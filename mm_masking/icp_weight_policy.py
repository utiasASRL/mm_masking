import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
from dICP.ICP import ICP
from radar_utils import load_pc_from_file, cfar_mask, extract_pc, radar_polar_to_cartesian_diff, radar_cartesian_to_polar, radar_polar_to_cartesian, extract_weights, point_to_cart_idx, form_cart_range_angle_grid, form_polar_range_grid
from neptune.types import File
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias)
        except AttributeError:
            print("Skipping initialization of ", classname)

class LearnICPWeightPolicy(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Define constant params (need to move to config file)
        self.res = 0.0596   # This is the old resolution!

        icp_type=params["icp_type"]
        network_inputs = {"fft": params["fft_input"], "cfar": params["cfar_input"], "range": params["range_input"]}
        network_input_type=params["network_input_type"]
        network_output_type=params["network_output_type"]
        leaky=params["leaky"]
        dropout=params["dropout"]
        batch_norm=params["batch_norm"]
        float_type=params["float_type"]
        device=params["device"]
        init_weights=params["init_weights"]
        normalize_type=params["normalize"]
        log_transform=params["log_transform"]
        a_threshold=params["a_thresh"]
        b_threshold=params["b_thresh"]
        gt_eye=params["gt_eye"]
        max_iter=params["max_iter"]

        config_path = '../external/dICP/config/dICP_config.yaml'
        self.ICP_alg = ICP(icp_type=icp_type, config_path=config_path, differentiable=True, max_iterations=max_iter, tolerance=1e-5)
        self.ICP_alg_inference = ICP(icp_type=icp_type, config_path=config_path, differentiable=False, max_iterations=50, tolerance=1e-5)
        self.float_type = float_type
        self.device = device
        self.network_inputs = network_inputs
        if network_input_type == 'cartesian':
            self.range_mask, _ = form_cart_range_angle_grid(device=device)
        elif network_input_type == 'polar':
            self.range_mask = form_polar_range_grid(polar_resolution=self.res, device=device)
        
        self.network_input_type = network_input_type
        self.network_output_type = network_output_type
        self.leaky = leaky
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.normalize_type = normalize_type
        self.log_transform = log_transform
        self.a_thres = a_threshold
        self.b_thres = b_threshold
        self.gt_eye = gt_eye
        self.norm_weights = params['norm_weights']

        # Parameters saving
        self.mean_num_pts = 0.0
        self.max_w = 0.0
        self.min_w = 0.0
        self.mean_w = 0.0
        self.mean_all_pts = 0.0

        # Define network
        init_c_num = network_inputs['fft'] + network_inputs['cfar'] + network_inputs['range']
        enc_channels = [init_c_num, 8, 16, 32, 64, 128, 256]
        dec_channels = [256, 128, 64, 32, 16, 8]

        self.encoder = ModuleList(
			[self.conv_block(enc_channels[i], enc_channels[i + 1], i)
			 	for i in range(len(enc_channels) - 1)])

        self.decoder = ModuleList(
            [self.conv_block(dec_channels[i], dec_channels[i + 1])
                for i in range(len(dec_channels) - 1)])

        self.final_layer = nn.Sequential(
            nn.Conv2d(dec_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )

        if init_weights:
            self.apply(weights_init)

    def conv_block(self, in_channels, out_channels, i=0):
        if self.leaky:
            relu_layer = nn.LeakyReLU(0.1)
        else:
            relu_layer = nn.ReLU()
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        modules.append(relu_layer)
        if self.batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        modules.append(relu_layer)
        if self.batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        if self.dropout > 0.0:
            modules.append(nn.Dropout(p=self.dropout))
        
        # Add maxpooling layer after each conv block except the first in encoder
        if i > 0:
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*modules)

    def forward(self, batch_scan, batch_map, T_init, binary=False, override_mask=None, neptune_run=None, epoch=0, batch_idx=0):
        # If override_mask is not None, then don't use network to get mask, just use override_mask
        # Extract points
        fft_data = batch_scan['fft_data'].to(self.device)#.requires_grad_(True)
        fft_cfar = batch_scan['fft_cfar'].to(self.device)
        scan_pc_raw = batch_scan['raw_pc'].to(self.device)
        #map_pc_paths = batch_map['pc_path']
        map_pc = batch_map['pc'].to(self.device)

        # Form weight fft "scan", where each non-zero pixel is the weight for the 
        # corresponding pointcloud point at the same pixel
        fft_weights = torch.where(fft_cfar > 0.0, 1.0, 0.0)

        if override_mask is None:
            input_data = None
            # Convert input data to desired network input
            if self.network_inputs['fft']:
                # FFT image is already in polar or cartesian
                input_data = fft_data.unsqueeze(1)
            if self.network_inputs['cfar']:
                # CFAR image is already in polar or cartesian
                input_data = torch.cat([input_data, fft_cfar.unsqueeze(1)], dim=1)
            if self.network_inputs['range']:
                range_stack = torch.stack([self.range_mask for i in range(input_data.shape[0])], dim=0).unsqueeze(1)
                input_data = torch.cat([input_data, range_stack], dim=1)

            if self.log_transform:
                input_data = torch.log(input_data + 1e-6)
            for c in range(input_data.shape[1]):
                if "minmax" in self.normalize_type:
                    c_max = torch.max(input_data[:,c,:,:])
                    c_min = torch.min(input_data[:,c,:,:])
                    input_data[:,c,:,:] = (input_data[:,c,:,:] - c_min) / (c_max - c_min)
                elif "standardize" in self.normalize_type:
                    c_mean = torch.mean(input_data[:,c,:,:])
                    c_std = torch.std(input_data[:,c,:,:])
                    input_data[:,c,:,:] = (input_data[:,c,:,:] - c_mean) / c_std

            # Encoder
            enc_layers = []
            for i, layer in enumerate(self.encoder):
                enc_layers.append(input_data)
                input_data = layer(input_data)
            enc_layers.reverse()

            # Decoder
            for i, decoder_layer in enumerate(self.decoder):
                # Load in skip connection
                skip_con = enc_layers[i]
                # Upsample input data to match skip connection
                bi_w = skip_con.shape[2]
                bi_h = skip_con.shape[3]
                bi_upsample = nn.UpsamplingBilinear2d(size=(bi_w, bi_h))
                input_data = bi_upsample(input_data)
                # Now convolve using decoder to reduce channels
                input_data = decoder_layer(input_data)
                # Concatenate with skip connection
                input_data = torch.cat([enc_layers[i], input_data], dim=1)
                # Pass through decoder again
                input_data = decoder_layer(input_data)

            weight_mask = self.final_layer(input_data).squeeze(1)

            del input_data, enc_layers, bi_upsample, bi_w, bi_h, skip_con
            torch.cuda.empty_cache()
        else:
            weight_mask = override_mask

        # Normalize weights
        if self.norm_weights:
            weight_mask = weight_mask / torch.amax(weight_mask, dim=(1,2), keepdim=True)

        if binary:
            weight_mask = torch.where(weight_mask > 0.5, 1.0, 0.0)

        # Extract weights correcponding to scan_pc
        weights, diff_mean_num_non0, mean_num_non0, mean_w, max_w, min_w = extract_weights(weight_mask, scan_pc_raw)

        # Save params
        self.mean_num_pts = mean_num_non0
        self.max_w = max_w
        self.min_w = min_w
        self.mean_w = mean_w

        non0_x = scan_pc_raw[:,:,0] != 0.0
        non0_y = scan_pc_raw[:,:,1] != 0.0
        non0_pts = non0_x * non0_y
        self.mean_all_pts = torch.sum(non0_pts) / scan_pc_raw.shape[0]

        del fft_data, fft_cfar, fft_weights
        torch.cuda.empty_cache()

        if neptune_run is not None and batch_idx <= 10:
            # Plot the scan and map pointclouds
            map_pc_0 = map_pc[0].detach().cpu().numpy()
            # Only use map_pc_0 that are less than self.ICP_alg.target_pad_val
            map_pc_0 = map_pc_0[map_pc_0[:, 0] < self.ICP_alg.target_pad_val]
            scan_pc_0 = scan_pc_raw[0].detach().cpu().numpy()
            # Only use scan_pc_0 points that aren't exactly 0
            scan_w_0 = weights[0].detach().cpu().numpy()
            scan_w_0 = scan_w_0[np.abs(scan_pc_0[:,0]) > 0.05]
            # Normalize scan_w_0 since weights are relative
            scan_w_0 = scan_w_0 / np.max(scan_w_0)
            scan_pc_0 = scan_pc_0[np.abs(scan_pc_0[:,0]) > 0.05]

            # Also isolate the points for which weight is above 0.01
            scan_pc_0_used = scan_pc_0[scan_w_0 > 0.01]
            scan_w_0_used = scan_w_0[scan_w_0 > 0.01]
            scan_pc_0_w0 = scan_pc_0[scan_w_0 <= 0.01]
            scan_w_0_w0 = scan_w_0[scan_w_0 <= 0.01]

            fig = plt.figure()
            plt.scatter(map_pc_0[:, 0], map_pc_0[:, 1], s=1.0, c='r')
            plt.scatter(scan_pc_0[:, 0], scan_pc_0[:, 1], s=0.5, c='b', alpha=scan_w_0/max(scan_w_0))
            plt.legend(['map', 'scan'])
            plt.title("Pointclouds")
            neptune_run["extracted_pc"].append(fig, name=("epoch " + str(epoch) + ",batch " + str(batch_idx)))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.set_facecolor('black')
            sc = ax.scatter(scan_pc_0_used[:, 0], scan_pc_0_used[:, 1], c=scan_w_0_used, clim=(0.00, 1.0), cmap='spring', s=0.5)
            ax.scatter(scan_pc_0_w0[:, 0], scan_pc_0_w0[:, 1], c=0.5+scan_w_0_w0, clim=(0.0, 1.0), cmap='binary', s=0.5)
            cbar = plt.colorbar(sc, label='Weights')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title("Weighted Scan")
            neptune_run["weighted_pc"].append(fig, name=("epoch " + str(epoch) + ",batch " + str(batch_idx)))
            plt.close(fig)

        # Pass the modified fft_data through ICP
        del scan_pc_raw
        scan_pc_filt = batch_scan['filtered_pc'].to(self.device)
        T_est = self.icp(scan_pc_filt, map_pc, T_init, weights)

        return T_est, weight_mask, diff_mean_num_non0
    
    def icp(self, scan_pc, map_pc, T_init, weights):
        loss_fn = {"name": "cauchy", "metric": 1.0}
        trim_dist = 5.0
        if self.training:
            icp_result = self.ICP_alg.icp(scan_pc, map_pc, 
                                    T_init=T_init, weight=weights,
                                    trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
        else:
            icp_result = self.ICP_alg_inference.icp(scan_pc, map_pc, 
                                    T_init=T_init, weight=weights,
                                    trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
        return icp_result['T']