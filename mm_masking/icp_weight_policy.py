import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
import cv2
from torchvision import datasets, transforms
from dICP.ICP import ICP
from radar_utils import load_pc_from_file, cfar_mask, extract_pc, radar_polar_to_cartesian_diff, radar_cartesian_to_polar, radar_polar_to_cartesian, extract_weights, point_to_cart_idx
from neptune.types import File
import time

import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias)
        except AttributeError:
            print("Skipping initialization of ", classname)

class LearnICPWeightPolicy(nn.Module):
    def __init__(self, icp_type='pt2pl', network_input='cartesian', 
                 network_output='cartesian', leaky=False, dropout=0.0, batch_norm=False,
                 float_type=torch.float64, device='cpu', init_weights=True,
                 normalize_type='none', log_transform=False, fft_mean=0.0, fft_std=1.0, fft_min=0.0, fft_max=1.0,
                 a_threshold=0.7, b_threshold=0.09, icp_weight=1.0, gt_eye=True, max_iter=25):
        super().__init__()
        config_path = '../external/dICP/config/dICP_config.yaml'
        self.ICP_alg = ICP(icp_type=icp_type, config_path=config_path, differentiable=True, max_iterations=max_iter, tolerance=1e-5)
        self.float_type = float_type
        self.device = device
        self.network_input = network_input
        self.network_output = network_output
        self.leaky = leaky
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.normalize_type = normalize_type
        self.log_transform = log_transform
        self.a_thres = a_threshold
        self.b_thres = b_threshold
        self.icp_weight = icp_weight
        self.gt_eye = gt_eye

        # Parameters saving
        self.mean_num_pts = 0.0
        self.max_w = 0.0
        self.min_w = 0.0
        self.mean_w = 0.0
        self.mean_all_pts = 0.0

        # Load in normalization params
        self.fft_mean = fft_mean
        self.fft_std = fft_std
        self.fft_min = fft_min
        self.fft_max = fft_max

        # Define constant params (need to move to config file)
        self.res = 0.0596   # This is the old resolution!

        channels = [1, 8, 16, 32, 64, 128, 256]

        self.channels = channels

        self.encoder = ModuleList(
			[self.conv_block(channels[i], channels[i + 1], i)
			 	for i in range(len(channels) - 1)])

        self.decoder = ModuleList(
            [self.conv_block(channels[i+1], channels[i])
                for i in reversed(range(1,len(channels) - 1))])

        self.final_layer = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1),
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
        azimuths = batch_scan['azimuths'].to(self.device)
        #az_timestamps = batch_scan['az_timestamps'].to(self.device)
        fft_cfar = batch_scan['fft_cfar'].to(self.device)
        scan_pc = batch_scan['pc'].to(self.device)
        #map_pc_paths = batch_map['pc_path']
        map_pc = batch_map['pc'].to(self.device)

        # Form weight fft "scan", where each non-zero pixel is the weight for the 
        # corresponding pointcloud point at the same pixel
        fft_weights = torch.where(fft_cfar > 0.0, 1.0, 0.0)

        if override_mask is None:
            # Convert input data to desired network input
            if self.network_input == 'polar':
                input_data = fft_data
            elif self.network_input == 'cartesian':
                input_data = radar_polar_to_cartesian_diff(fft_data, azimuths, self.res)
            
            if self.log_transform:
                input_data = torch.log(input_data + 1e-6)
            if self.normalize_type == "minmax":
                input_data = (input_data - self.fft_min) / (self.fft_max - self.fft_min)
            elif self.normalize_type == "standardize":
                input_data = input_data - self.fft_mean
                input_data = input_data / self.fft_std

            input_data = input_data.unsqueeze(1)
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

            mask = self.final_layer(input_data).squeeze(1)

            del input_data, enc_layers, bi_upsample, bi_w, bi_h, skip_con
            torch.cuda.empty_cache()
        else:
            mask = override_mask
            # Make sure override mask matches self.network_input
            if mask.shape == fft_data.shape and self.network_input == 'cartesian':
                mask = radar_polar_to_cartesian_diff(mask, azimuths, self.res)
            elif mask.shape != fft_data.shape and self.network_input == 'polar':
                mask = radar_cartesian_to_polar(mask, azimuths, self.res)

        if binary:
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        # This needs to be cleaned up
        if self.network_input == 'polar':
            # If polar input, original output mask is polar
            # Apply mask to desired output form
            if self.network_output == 'polar':
                output_data = fft_weights * mask

            elif self.network_output == 'cartesian':
                mask = radar_polar_to_cartesian_diff(mask, azimuths, self.res)
                bev_data = radar_polar_to_cartesian_diff(fft_weights, azimuths, self.res)
                output_data = bev_data * mask

        elif self.network_input == 'cartesian':
            # If cartesian input, original output mask is cartesian
            # Apply mask to desired output form
            if self.network_output == 'polar':
                mask = radar_cartesian_to_polar(mask, azimuths, self.res)
                output_data = fft_weights * mask
            
            elif self.network_output == 'cartesian':
                bev_data = radar_polar_to_cartesian_diff(fft_weights, azimuths, self.res)
                output_data = bev_data * mask

        # output_data is now the weights for each non-zero pixel in fft_cfar
        # Extract weights correcponding to scan_pc
        weights, diff_mean_num_non0, mean_num_non0, mean_w, max_w, min_w = extract_weights(mask, scan_pc)

        # Save params
        self.mean_num_pts = mean_num_non0
        self.max_w = max_w
        self.min_w = min_w
        self.mean_w = mean_w
        self.mean_all_pts = torch.sum(scan_pc[:,:,0] > 0.0) / scan_pc.shape[0]

        del fft_data, fft_cfar, fft_weights, output_data
        torch.cuda.empty_cache()

        if neptune_run is not None:
            # Plot the scan and map pointclouds
            map_pc_0 = map_pc[0].detach().cpu().numpy()
            # Only use map_pc_0 that are less than self.ICP_alg.target_pad_val
            map_pc_0 = map_pc_0[map_pc_0[:, 0] < self.ICP_alg.target_pad_val]
            scan_pc_0 = scan_pc[0].detach().cpu().numpy()
            # Only use scan_pc_0 points that aren't exactly 0
            scan_w_0 = weights[0].detach().cpu().numpy()
            scan_w_0 += 0.001
            scan_w_0[scan_w_0 > 1.0] = 1.0
            scan_w_0 = scan_w_0[np.abs(scan_pc_0[:,0]) > 0.01]
            scan_pc_0 = scan_pc_0[np.abs(scan_pc_0[:,0]) > 0.01]

            # Also isolate the points for which weight is above 0.01
            scan_pc_0_used = scan_pc_0[scan_w_0 > 0.01]
            scan_w_0_used = scan_w_0[scan_w_0 > 0.01]
            scan_pc_0_w0 = scan_pc_0[scan_w_0 <= 0.01]
            scan_w_0_w0 = scan_w_0[scan_w_0 <= 0.01]

            fig = plt.figure()
            plt.scatter(map_pc_0[:, 0], map_pc_0[:, 1], s=1.0, c='r')
            plt.scatter(scan_pc_0[:, 0], scan_pc_0[:, 1], s=0.5, c='b', alpha=scan_w_0)
            plt.legend(['map', 'scan'])
            plt.title("Pointclouds")
            neptune_run["extracted_pc"].append(fig, name=("epoch " + str(epoch) + ",batch " + str(batch_idx)))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.set_facecolor('black')
            sc = ax.scatter(scan_pc_0_used[:, 0], scan_pc_0_used[:, 1], c=scan_w_0_used, clim=(0.01, 1.0), cmap='spring', s=0.5)
            ax.scatter(scan_pc_0_w0[:, 0], scan_pc_0_w0[:, 1], c='grey', s=0.5)
            cbar = plt.colorbar(sc, label='Weights')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title("Weighted Scan")
            neptune_run["weighted_pc"].append(fig, name=("epoch " + str(epoch) + ",batch " + str(batch_idx)))
            #plt.savefig("weighted_pc.png")
            plt.close(fig)

        # Pass the modified fft_data through ICP
        if self.icp_weight > 0.0:
            T_est = self.icp(scan_pc, map_pc, T_init, weights)
        else:
            T_est = T_init

        return T_est, mask, diff_mean_num_non0
    
    def icp(self, scan_pc, map_pc, T_init, weights):
        loss_fn = {"name": "cauchy", "metric": 1.0}
        trim_dist = 5.0
        _, T_ms = self.ICP_alg.icp(scan_pc, map_pc, 
                                   T_init=T_init, weight=weights,
                                   trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
        return T_ms