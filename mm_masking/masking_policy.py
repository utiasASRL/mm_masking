import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import datasets, transforms
from dICP.ICP import ICP
from RadarUtils import load_pc_from_file, extract_pc, pol_2_cart, load_radar, extract_pc_parallel

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

            
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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

        scan_pc_list = extract_pc_parallel(fft_data, self.res, az_timestamps, azimuths, a_thresh=a_thresh, b_thresh=b_thresh)

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