import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
import cv2
from torchvision import datasets, transforms
from dICP.ICP import ICP
from radar_utils import load_pc_from_file, cfar_mask, extract_pc, radar_polar_to_cartesian_diff, radar_cartesian_to_polar, radar_polar_to_cartesian
from neptune.types import File

import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias)
        except AttributeError:
            print("Skipping initialization of ", classname)

class LearnMaskPolicy(nn.Module):
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

        # Load in normalization params
        self.fft_mean = fft_mean
        self.fft_std = fft_std
        self.fft_min = fft_min
        self.fft_max = fft_max

        # Define constant params (need to move to config file)
        self.res = 0.0596   # This is the old resolution!

        channels = [1, 8, 16, 32, 64, 128, 256]
        #channels = [1, 64, 128, 256, 512, 1024]

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

    def forward(self, batch_scan, batch_map, T_gt, T_init, binary=False, override_mask=None):
        # If override_mask is not None, then don't use network to get mask, just use override_mask
        # Extract points
        scan_pc_paths = batch_scan['pc_path']
        fft_data = batch_scan['fft_data'].to(self.device)#.requires_grad_(True)
        azimuths = batch_scan['azimuths'].to(self.device)
        az_timestamps = batch_scan['az_timestamps'].to(self.device)
        #radar_paths = batch_scan['radar_path']
        map_pc_paths = batch_map['pc_path']

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
            
            logits = self.final_layer(input_data).squeeze(1)
            mask = logits
        else:
            mask = override_mask
            # Make sure override mask matches self.network_input
            if mask.shape == fft_data.shape and self.network_input == 'cartesian':
                mask = radar_polar_to_cartesian_diff(mask, azimuths, self.res)
            elif mask.shape != fft_data.shape and self.network_input == 'polar':
                mask = radar_cartesian_to_polar(mask, azimuths, self.res)

        if binary:
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        mask_output = mask

        #print("Max and min of mask: " + 
        #      str(np.round(torch.max(mask.detach()).item(), 3)) + ", " + 
        #      str(np.round(torch.min(mask.detach()).item(), 3)))

        # This needs to be cleaned up
        if self.network_input == 'polar':
            # If polar input, original output mask is polar
            # Apply mask to desired output form
            if self.network_output == 'polar':
                output_data = fft_data * mask

            elif self.network_input == 'cartesian':
                mask = radar_polar_to_cartesian_diff(mask, azimuths, self.res)
                bev_data = radar_polar_to_cartesian_diff(fft_data, azimuths, self.res)
                output_data = bev_data * mask

        elif self.network_input == 'cartesian':
            # If cartesian input, original output mask is cartesian
            # Apply mask to desired output form
            if self.network_output == 'polar':
                mask = radar_cartesian_to_polar(mask, azimuths, self.res)
                output_data = fft_data * mask
            
            elif self.network_output == 'cartesian':
                bev_data = radar_polar_to_cartesian_diff(fft_data, azimuths, self.res)
                output_data = bev_data * mask

        # Extract source pointclouds
        scan_pc_cfar_mask = cfar_mask(output_data, self.res, a_thresh=self.a_thres, b_thresh=self.b_thres, diff=True)
        scan_pc_list = extract_pc(scan_pc_cfar_mask, self.res, azimuths, az_timestamps, diff=True)

        # Extract target pointclouds
        map_pc_list = []
        mean_num_points = 0
        for ii in range(T_init.shape[0]):
            map_pc_ii = load_pc_from_file(map_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=True)
            map_pc_list.append(map_pc_ii)
            # Also extract pc pointcloud for statistics and potential transform
            scan_ii = scan_pc_list[ii]
            mean_num_points += scan_ii.shape[0]
            if self.gt_eye:
                # Also want to transform scan_pc into map_pc frame
                # since we want an ICP ground truth transform of eye for simplicity
                
                # Transform scan_pc to map frame using gt
                scan_ii = (T_gt[ii, :3, :3] @ scan_ii.T).T + T_gt[ii, :3, 3]
                # Add fake ones point to scan_ii
                #fake_pt = torch.tensor([[1000.0, 1.0, 0.0]], dtype=self.float_type, device=self.device)
                #scan_ii = torch.cat([scan_ii, fake_pt], dim=0)
                #scan_ii.retain_grad()
                scan_pc_list[ii] = scan_ii
        mean_num_points = mean_num_points / T_init.shape[0]

        # Pass the modified fft_data through ICP
        if self.icp_weight > 0.0:
            T_est = self.icp(scan_pc_list, map_pc_list, T_init)
        else:
            T_est = T_init

        return T_est, mask_output, mean_num_points
    
    def icp(self, scan_pc_list, map_pc_list, T_init):
        loss_fn = {"name": "cauchy", "metric": 1.0}
        trim_dist = 5.0
        #print(scan_pc_list[0])
        #print(scan_pc_list[0].grad)
        
        _, T_ms = self.ICP_alg.icp(scan_pc_list, map_pc_list, T_init=T_init, trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
        #T_ms.sum().backward()
        #torch.set_printoptions(threshold=10_000)
        #print(scan_pc_list[0])
        #print(scan_pc_list[0].grad[:,0])
        #afdsdfs
        return T_ms

class LearnScalePolicy(nn.Module):
    def __init__(self, size_pc=20, icp_type='pt2pl', float_type=torch.float64, device='cpu', scale_init=1.0, true_scale=1.2):
        super().__init__()
        config_path = '../external/dICP/config/dICP_config.yaml'
        self.ICP_alg = ICP(icp_type=icp_type, config_path=config_path, differentiable=True, max_iterations=50, tolerance=1e-5)
        self.size_pc = size_pc
        self.float_type = float_type
        self.device = device
        # Define a single constant parameter
        self.true_scale = torch.tensor(true_scale, dtype=float_type, device=device)
        self.params = torch.nn.Parameter(torch.tensor(scale_init, dtype=float_type, device=device))

    def forward(self, batch_scan, batch_map, batch_T_init):
        # Extract points
        scan_pc_paths = batch_scan['pc_path']
        map_pc_paths = batch_map['pc_path']
        T_init = batch_T_init
        # Run ICP
        T_est = self.icp(scan_pc_paths, map_pc_paths, T_init)

        return T_est
    
    def icp(self, scan_pc_paths, map_pc_paths, T_init):
        T_ms = torch.zeros((T_init.shape[0], 4, 4), dtype=self.float_type, device=self.device)
        for ii in range(T_init.shape[0]):
            # Load in point clouds
            scan_pc = load_pc_from_file(scan_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=False)
            map_pc = load_pc_from_file(map_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=True)

            # Apply inverse of true scale to scan_pc. This avoids having to apply this during data loading.
            # The parameter is trying to counter this constant transform
            scan_pc_corrupt = scan_pc / self.true_scale

            # Apply relu to parameter to ensure it is positive
            scale = torch.relu(self.params)

            # Apply scale to scan_pc
            scan_pc_unscaled = scan_pc_corrupt * scale

            loss_fn = {"name": "huber", "metric": 1.0}
            _, T_ii = self.ICP_alg.icp(scan_pc_unscaled, map_pc, T_init=T_init[ii], trim_dist=5.0, loss_fn=loss_fn, dim=2)
            T_ms[ii, :, :] = T_ii

        return T_ms

class LearnICPPolicy(nn.Module):
    def __init__(self, size_pc=20, icp_type='pt2pl', float_type=torch.float64, device='cpu', trim_init=5.0, huber_init=1.0):
        super().__init__()
        config_path = '../external/dICP/config/dICP_config.yaml'
        self.ICP_alg = ICP(icp_type=icp_type, config_path=config_path, differentiable=True, max_iterations=50, tolerance=1e-5)
        self.size_pc = size_pc
        self.float_type = float_type
        self.device = device
        # Define a single constant parameter
        self.params = torch.nn.Parameter(torch.tensor([trim_init, huber_init], dtype=float_type, device=device))

    def forward(self, batch_scan, batch_map, batch_T_init):
        # Extract points
        scan_pc_paths = batch_scan['pc_path']
        map_pc_paths = batch_map['pc_path']
        T_init = batch_T_init
        # Run ICP
        T_est = self.icp(scan_pc_paths, map_pc_paths, T_init)

        return T_est
    
    def icp(self, scan_pc_paths, map_pc_paths, T_init):
        T_ms = torch.zeros((T_init.shape[0], 4, 4), dtype=self.float_type, device=self.device)
        for ii in range(T_init.shape[0]):
            scan_pc = load_pc_from_file(scan_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=False)
            map_pc = load_pc_from_file(map_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=True)

            # Apply relu to parameters to ensure they are positive
            trim_dist = torch.relu(self.params[0])
            huber_delta = torch.relu(self.params[1])

            loss_fn = {"name": "huber", "metric": huber_delta}
            _, T_ii = self.ICP_alg.icp(scan_pc, map_pc, T_init=T_init[ii], trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
            T_ms[ii, :, :] = T_ii

        return T_ms
    
class LearnBFARPolicy(nn.Module):
    def __init__(self, size_pc=20, icp_type='pt2pl', float_type=torch.float64, device='cpu', a_init=1.0, b_init=0.09):
        super().__init__()
        config_path = '../external/dICP/config/dICP_config.yaml'
        self.ICP_alg = ICP(icp_type=icp_type, config_path=config_path, differentiable=True, max_iterations=50, tolerance=1e-5)
        self.size_pc = size_pc
        self.float_type = float_type
        self.device = device

        # Define constant params (need to move to config file)
        self.res = 0.0596   # This is the old resolution!

        # Define a single constant parameter
        a_torch = torch.tensor([a_init], dtype=float_type, device=device)
        b_torch = torch.tensor([b_init], dtype=float_type, device=device)
        self.params = torch.nn.Parameter(torch.tensor([a_torch, b_torch], dtype=float_type, device=device))
        #self.b = torch.nn.Parameter(torch.tensor([b_init], dtype=float_type, device=device))
        #self.a = torch.nn.Parameter(torch.tensor([0.9605702757835388]))
        #self.b = torch.nn.Parameter(torch.tensor([0.0376703217625618]))

    def forward(self, batch_scan, batch_map, batch_T_init):
        # Extract points
        scan_pc_paths = batch_scan['pc_path']
        fft_data = batch_scan['fft_data'].to(self.device)
        azimuths = batch_scan['azimuths'].to(self.device)
        az_timestamps = batch_scan['az_timestamps'].to(self.device)
        #radar_paths = batch_scan['radar_path']
        map_pc_paths = batch_map['pc_path']
        T_init = batch_T_init

        # Print params
        #print("Extract params: ", self.a.item(), self.b.item())

        #vis_scan_pc = scan_pc.detach().clone()     # Save for visualization

        # Pass the modified point cloud and the map point cloud through ICP
        T_est = self.icp(scan_pc_paths, map_pc_paths, T_init, fft_data, azimuths, az_timestamps)

        del fft_data, azimuths, az_timestamps, T_init, scan_pc_paths, map_pc_paths

        return T_est
    
    def icp(self, scan_pc_paths, map_pc_paths, T_init, fft_data, azimuths, az_timestamps):
        T_ms = torch.zeros((T_init.shape[0], 4, 4), dtype=self.float_type, device=self.device)

        # Apply relu to parameters to ensure they are positive
        a_thresh = torch.relu(self.params[0])
        b_thresh = torch.relu(self.params[1])

        scan_pc_cfar_mask = cfar_mask(fft_data, self.res, a_thresh=a_thresh, b_thresh=b_thresh)
        scan_pc_list = extract_pc(scan_pc_cfar_mask, self.res, azimuths, az_timestamps)

        for ii in range(T_init.shape[0]):
            """
            loc_radar_img = cv2.imread(radar_paths[ii], cv2.IMREAD_GRAYSCALE)
            loc_radar_mat = np.asarray(loc_radar_img)

            fft_data, azimuths, az_timestamps = load_radar(loc_radar_mat)
            fft_data = torch.tensor(fft_data, dtype=self.float_type)
            azimuths = torch.tensor(azimuths, dtype=self.float_type)
            az_timestamps = torch.tensor(az_timestamps, dtype=self.float_type)
            """

            scan_pc = scan_pc_list[ii]

            #scan_pc_file = load_pc_from_file(scan_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=False)
            map_pc = load_pc_from_file(map_pc_paths[ii], to_type=self.float_type, to_device=self.device, flip_y=True)

            #scan_pc = extract_pc(fft_data[ii], self.res, az_timestamps[ii], azimuths[ii], a_thresh=self.a, b_thresh=self.b)
            #scan_pc = extract_pc(fft_data, self.res, az_timestamps, azimuths, a_thresh=1.0, b_thresh=0.09)
            #scan_pc = pol_2_cart(scan_pc)

            """
            scan_pc = scan_pc.detach().cpu().numpy()
            scan_pc_file = scan_pc_file.detach().cpu().numpy()


            print(scan_pc.shape)
            import matplotlib.pyplot as plt
            plt.figure()
            
            plt.scatter(scan_pc_file[:, 0], scan_pc_file[:, 1], s=0.5, c='b')
            plt.scatter(scan_pc[:, 0], scan_pc[:, 1], s=0.5, c='r')
            plt.savefig("peak_points_pc.png")

            dsaffds
            """
            

            #pc_ii_file, T_ii_file = pt2pl_dICP(scan_pc_file, map_pc, T_init=T_init[ii], max_iterations=20, tolerance=1e-9, trim_dist=25.0, huber_delta=1.0, dim=2)
            
            # If less than 10 points, don't try ICP
            #if len(scan_pc) > 10:
            loss_fn = {"name": "huber", "metric": 1.0}
            _, T_ii = self.ICP_alg.icp(scan_pc, map_pc, T_init=T_init[ii], trim_dist=5.0, loss_fn=loss_fn, dim=2)
            #else:
            #    T_ii = T_init[ii]

            T_ms[ii, :, :] = T_ii

            del scan_pc, T_ii, map_pc
            #pcs.append(pc_ii)

        del scan_pc_list, a_thresh, b_thresh

        return T_ms#, pcs
    


""" 
# THIS WAS CODE USED TO TEST POLAR TO CARTESIAN CONVERSION
# KEEPING IT IN CASE I NEED IT LATER, IDEALLY THIS GETS TURNED INTO A TEST
#fft_data = fft_data * 0.0
#fft_data[:, :, 90:111] = 1.0
#fft_data[:, 90:111, :] = 1.0

# DELETE
fft_data_use = fft_data[0].detach().cpu().numpy()
azimuths_use = azimuths[0].detach().cpu().numpy()
cart_og = radar_polar_to_cartesian(fft_data_use, azimuths_use, self.res)
cart_diff = radar_polar_to_cartesian_diff(fft_data, azimuths, self.res)

#cart_og =cart_og * 0.0
#leng = int((cart_og.shape[0]-1)/2)
#del_l = 10
#cart_og[leng-del_l:leng+del_l, leng-del_l:leng+del_l] = 1.0
#cart_diff = cart_diff * 0.0
#cart_diff[:, leng-del_l:leng+del_l, leng-del_l:leng+del_l] = 1.0

cart_og_diff = torch.from_numpy(cart_og).to(self.device)
cart_og_diff = cart_og_diff.unsqueeze(0)

polar_og = fft_data_use
polar_diff = radar_cartesian_to_polar(cart_diff, azimuths, self.res, polar_pixel_width=fft_data.shape[2])

cart_regen = radar_polar_to_cartesian_diff(polar_diff, azimuths, self.res)

plt.figure()
plt.imshow(cart_og, cmap='gray')
plt.colorbar()
plt.savefig("results/cart_og.png")
plt.close()

plt.figure()
plt.imshow(cart_regen[0].detach().cpu().numpy(), cmap='gray')
plt.colorbar()
plt.savefig("results/cart_regen.png")
plt.close()

plt.figure()
plt.imshow(cart_diff[0].detach().cpu().numpy(), cmap='gray')
plt.colorbar()
plt.savefig("results/cart_diff.png")
plt.close()

cart_og_vs_diff = cart_og-cart_diff[0].detach().cpu().numpy()
cart_og_vs_diff[cart_og_vs_diff < 1e-2] = 0.0
cart_diff_vs_regen = cart_diff[0].detach().cpu().numpy()-cart_regen[0].detach().cpu().numpy()
cart_diff_vs_regen[cart_diff_vs_regen < 1e-2] = 0.0

plt.figure()
plt.imshow(cart_og_vs_diff, cmap='gray')
plt.colorbar()
plt.savefig("results/cart_og_vs_diff.png")
plt.close()

plt.figure()
plt.imshow(cart_diff_vs_regen, cmap='gray')
plt.colorbar()
plt.savefig("results/cart_diff_vs_regen.png")
plt.close()

plt.figure()
plt.imshow(polar_og[:,:], cmap='gray')
plt.colorbar()
plt.savefig("results/polar_og.png")
plt.close()

plt.figure()
plt.imshow(polar_diff[0,:,:].detach().cpu().numpy(), cmap='gray')
plt.colorbar()
plt.savefig("results/polar_diff.png")
plt.close()

asdfasdf
"""