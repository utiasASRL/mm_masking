import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn.functional as F
import cv2
import time

def load_pc_from_file(file_path, to_type=None, to_device='cpu'):
    pc = np.fromfile(file_path, dtype=np.float32)
    pc = pc.reshape((len(pc) // 6, 6))
    # Send to device
    pc = torch.from_numpy(pc).to(to_device)
    # Convert to type
    if to_type is not None:
        pc = pc.type(to_type)
    return pc

def load_radar(raw_img):
    raw_data = np.asarray(raw_img)
    time_convert = 1000
    encoder_conversion = 2 * np.pi / 5600
    timestamps = np.frombuffer(raw_data[:,:8].tobytes(), dtype=np.int64) * time_convert
    azimuths = np.frombuffer(raw_data[:,8:10].tobytes(), dtype=np.uint16) * encoder_conversion
    fft_data = np.divide(raw_data[:,11:], 255.0, dtype=np.float32)
    return fft_data, azimuths, timestamps

def cfar_mask(raw_scans, res, width=101, minr=2.0, maxr=80.0, guard=5, 
                    a_thresh=1.0, b_thresh=0.09, diff=True, steep_fact=10.0):
    assert(raw_scans.ndim == 3), "raw_scans must be 3D"
    device = raw_scans.device
    # Handle odd width
    width = width + 1 if width % 2 == 0 else width
    w2 = width // 2
    
    # Compute column range based on minimum/maximum range
    mincol = max(0, int(minr / res + w2 + guard + 1))
    maxcol = min(raw_scans.shape[2], int(maxr / res - w2 - guard))
    col_range = torch.arange(mincol, maxcol)

    left_start_idx = col_range - w2 - guard
    left_end_idx = col_range - guard
    left = torch.zeros((raw_scans.shape[0], raw_scans.shape[1], len(left_start_idx)), device=device)

    for idx, (start, end) in enumerate(zip(left_start_idx, left_end_idx)):
        left[:, :, idx] = torch.sum(raw_scans[:, :, start:end], axis=2)

    right_start_idx = col_range + guard + 1
    right_end_idx = col_range + w2 + guard + 1
    right = torch.zeros((raw_scans.shape[0], raw_scans.shape[1], len(right_start_idx)), device=device)
    for idx, (start, end) in enumerate(zip(right_start_idx, right_end_idx)):
        right[:, :, idx] = torch.sum(raw_scans[:, :,start:end], axis=2)
    
    stat = torch.maximum(left, right) / w2  # GO-CFAR
    thres = a_thresh * stat + b_thresh

    # Save full sized threshold map
    thres_full = 1000*torch.ones(raw_scans.shape, device=device)
    thres_full[:, :, col_range] = thres

    # Compute threshold mask
    if diff:
        thres_mask_raw = 0.5 * torch.tanh(steep_fact * (raw_scans - thres_full) + 2.5) + 0.5
        thres_mask = torch.hardshrink(thres_mask_raw, lambd=0.99)
    else:
        thres_mask = torch.where(raw_scans > thres_full, 1.0, 0.0)

    return thres_mask

def extract_pc(thres_mask, res, azimuth_angles, azimuth_times, T_ab=None, diff=True, steep_fact=10.0):
    device = thres_mask.device
    # Threshold the raw scans
    thres_scan = res * torch.arange(thres_mask.shape[2], device=device) * thres_mask

    # Find peaks of thresholded points
    peak_points = mean_peaks_parallel_fast(thres_scan, diff=diff, steep_fact=steep_fact)

    # Assemble all points, with azimuth angles and azimuth shapes as the other two dimensions
    azimuth_angles_mat = azimuth_angles.unsqueeze(2).repeat(1, 1, thres_mask.shape[2])
    azimuth_times_mat = azimuth_times.unsqueeze(2).repeat(1, 1, thres_mask.shape[2])
    peak_pt_mat = torch.cat((peak_points.unsqueeze(-1), azimuth_angles_mat.unsqueeze(-1), azimuth_times_mat.unsqueeze(-1)), dim=-1)

    # Flatten peak_points along second and third dimensions
    peak_pt_vec = peak_pt_mat.reshape(peak_pt_mat.shape[0], -1, peak_pt_mat.shape[-1])
    pc_list = []

    for ii in range(peak_pt_vec.shape[0]):
        peak_pt_vec_ii = peak_pt_vec[ii]
        nonzero_indices = peak_pt_vec_ii[:,0].nonzero(as_tuple=True)
        # Find the mean between every two consecutive peaks in nonzero_ii
        nonzero_odd_indices = nonzero_indices[0][1::2]
        nonzero_even_indices = nonzero_indices[0][0::2]
        nonzero_ii_start = peak_pt_vec_ii[nonzero_odd_indices]
        nonzero_ii_end = peak_pt_vec_ii[nonzero_even_indices]

        avg_peak_pt_vec_ii = (nonzero_ii_start + nonzero_ii_end) / 2.0

        pc_ii = pol_2_cart(avg_peak_pt_vec_ii)
        if T_ab is not None:
            T_ii = T_ab[ii]
            pc_ii = (T_ii[:3, :3] @ pc_ii.T).T + T_ii[:3, 3]

        pc_list.append(pc_ii)

    return pc_list

def extract_weights(mask, scan_pc):
    # Extract weights from mask corresponding to scan_pc points
    mask_c = mask.unsqueeze(1)
    scan_pc = scan_pc.type(mask_c.dtype)
    grid_pc = point_to_cart_idx(scan_pc, min_to_plus_1=True)

    # scan_pc has filled in (0, 0) points for batch shape matching
    # We want weights corresponding to these points to be 0
    # Therefore, set the indices for these points to be out of bounds
    # and pad the grid_sample function with 0's
    scan_x_0 = scan_pc[:,:,0] == 0.0
    scan_y_0 = scan_pc[:,:,1] == 0.0
    fake_scan_idx = scan_x_0 * scan_y_0

    grid_pc[fake_scan_idx] = -100.0 * torch.ones(2, dtype=grid_pc.dtype, device=grid_pc.device)

    # Finally, unsqueeze the grid_pc to give vector x 1 shape as output from grid_sample
    grid_pc = grid_pc.unsqueeze(2)
    weights = F.grid_sample(mask_c, grid_pc, mode='bilinear', padding_mode='zeros', align_corners=True)
    # Squeeze out the extra dimensions
    weights = weights.squeeze(1).squeeze(-1)

    # Compute stats
    # Weights below 0.05 are considered non-impactful/0
    mean_num_non0 = (torch.sum(weights[~fake_scan_idx] > 0.05)/weights.shape[0]).detach()

    mean_w = torch.mean(weights[~fake_scan_idx]).detach()
    max_w = torch.max(weights[~fake_scan_idx]).detach()
    min_w = torch.min(weights[~fake_scan_idx]).detach()
    # Compute number of non 0 weights in a differentiable way for potential backprop
    diff_mean_num_non0 = torch.sum(0.5 * torch.tanh(5*weights[~fake_scan_idx]) + 0.5) / weights.shape[0]

    return weights, diff_mean_num_non0, mean_num_non0, mean_w, max_w, min_w

def extract_bev_from_pts(pc, cart_pixel_width=640):
    # Find cartesian indeces of the pointcloud
    pc_idx = point_to_cart_idx(pc)

    # Set all out of range indices to midpoint
    pc_idx[pc_idx < 0] = cart_pixel_width // 2
    pc_idx[pc_idx > (cart_pixel_width-1)] = cart_pixel_width // 2

    # Form the BEV with all 0's for now
    pc_bev = torch.zeros((pc.shape[0], cart_pixel_width, cart_pixel_width), dtype=pc.dtype, device=pc.device)
    
    # Fill in 1's for all nearby pixels from each pc_idx
    pc_idx_floor = torch.floor(pc_idx).type(torch.long)
    pc_idx_ceil = torch.ceil(pc_idx).type(torch.long)
    pc_bev[torch.arange(pc_idx_ceil.shape[0]).unsqueeze(1), pc_idx_ceil[:,:,0], pc_idx_floor[:,:,1]] = 1
    pc_bev[torch.arange(pc_idx_ceil.shape[0]).unsqueeze(1), pc_idx_ceil[:,:,0], pc_idx_ceil[:,:,1]] = 1
    pc_bev[torch.arange(pc_idx_ceil.shape[0]).unsqueeze(1), pc_idx_floor[:,:,0], pc_idx_floor[:,:,1]] = 1
    pc_bev[torch.arange(pc_idx_ceil.shape[0]).unsqueeze(1), pc_idx_floor[:,:,0], pc_idx_ceil[:,:,1]] = 1

    # All out of bound and fake points generated for batching
    # are stored in centermost pixel, zero it out
    pc_bev[:, cart_pixel_width // 2, cart_pixel_width // 2] = 0.0

    return pc_bev

def mean_peaks_parallel_fast(arr, diff, steep_fact):
    res = torch.zeros_like(arr)
    
    # Find non-zero values
    if diff:
        zero_detect_arr = 1 - torch.tanh(steep_fact * arr)
    else:
        zero_detect_arr = arr == 0

    # Get first non-zero values in a given blob
    res_forward = arr[:, :, :-1] * (zero_detect_arr[:, :, 1:])

    # Get last non-zero values in a given blob
    res_backward = arr[:, :, 1:] * (zero_detect_arr[:, :, :-1])

    # Store both values as "peaks", their averages will be taken after points are extracted
    res[:, :, :-1] = res_forward + res_backward

    return res

def pol_2_cart(pointcloud):
    rho = pointcloud[:, 0]
    phi = pointcloud[:, 1]
    
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    z = torch.zeros_like(rho)
    
    return torch.stack((x, y, z), axis=1)

def radar_polar_to_cartesian(fft_data, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=640,
                             interpolate_crossover=False, fix_wobble=True):
    # TAKEN FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # Compute the range (m) captured by pixels in cartesian scan
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step
    # This fixes the wobble in the old CIR204 data from Boreas
    M = azimuths.shape[0]
    azms = azimuths.squeeze()
    if fix_wobble:
        c3 = np.searchsorted(azms, sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azms[c3]
        diff = sample_angle.squeeze() - a3
        a2 = azms[c2]
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)

def radar_polar_to_cartesian_diff(fft_data, azimuths, radar_resolution, cart_resolution=0.2384, cart_pixel_width=640,
                                  interpolate_crossover=True, fix_wobble=True):
    # ADAPTED FROM PYBOREAS
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_width (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    """
    # Compute the range (m) and angle (rad) values for each cartesian pixel
    sample_range, sample_angle = form_cart_range_angle_grid(cart_pixel_width=cart_pixel_width, dtype=fft_data.dtype, device=fft_data.device)

    # So far sample_range and sample_angle are the same for each item in batch
    # Now, expand them to match batch size
    sample_range = sample_range.unsqueeze(0).expand(fft_data.shape[0], -1, -1)
    sample_angle = sample_angle.unsqueeze(0).expand(fft_data.shape[0], -1, -1)

    # Interpolate radar data pixel coordinates
    # sample_u contains the range (pixel) value of each cartesian pixel
    # sample_v contains the angle (pixel) value of each cartesian pixel
    azimuth_step = (azimuths[:, -1] - azimuths[:, 0]) / (azimuths.shape[1] - 1)
    azimuth_step = azimuth_step.unsqueeze(1).unsqueeze(1).expand(sample_angle.shape)
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    azi_0 = azimuths[:, 0].unsqueeze(1).unsqueeze(1).expand(sample_angle.shape)
    sample_v = (sample_angle - azi_0) / azimuth_step

    # This fixes the wobble in the old CIR204 data from Boreas
    # This wobble is present when azimuth_step computed above isn't actually constant
    # Honestly, not entirely sure how this works... but did manage to get it working in batch
    M = azimuths.shape[1]
    if fix_wobble:
        azms_batch = azimuths.unsqueeze(1).expand(azimuths.shape[0], sample_angle.shape[1], azimuths.shape[1])
        # Searchsorted complains if the arrays are not contiguous
        c3 = torch.searchsorted(azms_batch.contiguous(), sample_angle.contiguous())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a = azimuths.unsqueeze(1).unsqueeze(3)
        az_idx = torch.arange(azimuths.shape[0])[:, None, None]
        a3 = a[az_idx, :, c3].squeeze(-1).squeeze(-1)
        diff = sample_angle - a3
        a2 = a[az_idx, :, c2].squeeze(-1).squeeze(-1)
        delta = diff * (diff < 0) * (c3 > 0) / (a3 - a2 + 1e-14)
        sample_v = (c3 + delta)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    # Add top/bottom padding for interpolation purposes, since azimuths should roll over
    if interpolate_crossover:
        fft_data = torch.concatenate((fft_data[:, -1:], fft_data, fft_data[:, :1]), 1)
        sample_v = sample_v + 1

    # Normalize sample_u and sample_v to be in the range [-1, 1] (this is needed for grid_sample)
    sample_u = sample_u / (fft_data.shape[2] - 1) * 2 - 1
    sample_v = sample_v / (fft_data.shape[1] - 1) * 2 - 1

    # polar_to_cart_wrap is a tensor of shape (N, cart_pixel_height, cart_pixel_width, 2)
    # polar_to_cart_wrap[N, i, j, :] contains the (u, v) polar coordinates of the cartesian pixel (i, j)
    polar_to_cart_warp = torch.stack((sample_u, sample_v), -1)

    # Set up dimensions for grid_sample
    fft_data = fft_data.unsqueeze(1)

    # Compute mapping, padding with zeros since we want to ignore out of bounds values, corners aligned as this corresponds to
    # having pixe values defined in the center of the pixel    
    remapped_data = F.grid_sample(fft_data, polar_to_cart_warp, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return remapped_data.squeeze(1)

def radar_cartesian_to_polar(cart, azimuths, radar_resolution, cart_resolution=0.2384, polar_pixel_shape=(400, 3360)):
    """Convert a cartesian radar scan to polar.
    """
    # Compute the range (m) captured by pixels in each azimuth
    range_grid = form_polar_range_grid(polar_resolution=radar_resolution, polar_pixel_shape=polar_pixel_shape, dtype=cart.dtype, device=cart.device)
    range_coords = range_grid[0]

    # For each azimuth and range, what is the cartesian coordinate?
    # Project the range along each azimuth onto the cartesian coordinate system
    sample_X = torch.sin(azimuths.unsqueeze(-1)) @ range_coords.unsqueeze(0)
    sample_Y = torch.cos(azimuths.unsqueeze(-1)) @ range_coords.unsqueeze(0)

    # Compute the pixel values of each cartesian coordinate
    # Note, sample_X and sample_Y are already centered at 0, which is what is needed for grid_sample
    # Thus, we simply need to convert from meters to pixels
    # Minus in front of sample_v since the pixel numbering is from top to bottom
    sample_u = sample_X / cart_resolution
    sample_v = -sample_Y / cart_resolution

    # Normalize sample_u and sample_v to be in the range [-1, 1] (this is needed for grid_sample)
    sample_u = sample_u / (cart.shape[2] - 1) * 2
    sample_v = sample_v / (cart.shape[1] - 1) * 2

    # cart_to_polar_wrap is a tensor of shape (N, polar_pixel_height, polar_pixel_width, 2)
    # cart_to_polar_wrap[N, i, j, :] contains the (u, v) cartesian coordinates of the polar pixel (i, j)
    cart_to_polar_warp = torch.stack((sample_u, sample_v), -1)

    # Set up dimensions for grid_sample
    cart = cart.unsqueeze(1)

    # Compute mapping, padding with zeros since we want to ignore out of bounds values, corners aligned as this corresponds to
    # having pixe values defined in the center of the pixel
    remapped_data = F.grid_sample(cart, cart_to_polar_warp.double(), mode='bilinear', padding_mode='zeros', align_corners=True)

    return remapped_data.squeeze(1)

def point_to_cart_idx(pc, cart_resolution=0.2384, cart_pixel_width=640, min_to_plus_1=False):
    # Compute the cartesian pixel coordinates of each point in the pointcloud pc
    # pc is a tensor of shape (N, m, 2/3)

    # First, isolate the x and y coordinates of the scan_pc points
    # and convert to pixels. Note, we want the x axis to point up
    # and the y axis to point right. Since indexing is from top to bottom,
    # flip the x/u indexing
    grid_pc_u = -pc[:,:,0] / cart_resolution
    grid_pc_v = pc[:,:,1] / cart_resolution

    if min_to_plus_1:
        # grid_sample requires the grid to list x/y coordinates in the range [-1, 1]
        # not 100% sure why, but need to stack the x/v and y/u coordinates in the opposite order
        grid_pc = torch.stack((grid_pc_v, grid_pc_u), axis=2)
        # Normalize grid_pc to be in the range [-1, 1] (this is needed for grid_sample)
        # grid_pc is already centered at 0, so we simply need to normalize
        grid_pc = grid_pc / (cart_pixel_width - 1) * 2
    else:
        grid_pc = torch.stack((grid_pc_u, grid_pc_v), axis=2)
        # Align (0, 0) with the center of the pixel grid
        grid_pc = grid_pc + cart_pixel_width / 2

    return grid_pc

def form_cart_range_angle_grid(cart_resolution=0.2384, cart_pixel_width=640, dtype=None, device='cpu'):
    # Compute the range (m) and angle (rad) value of each cartesian pixel
    # A pixels coordinates are the center of the pixel
    # If even number of pixels, 0 m coordinate falls on edges of pixels (hence the -0.5)
    # Otherwise, 0 m falls on the middle of a pixel and so form is simplified
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width / 2 * cart_resolution
    
    # Compute the value of each cartesian pixel, centered at 0
    if dtype is None:
        coords = torch.linspace(-cart_min_range, cart_min_range, cart_pixel_width, device=device)
    else:
        coords = torch.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=dtype, device=device)
    Y, X = torch.meshgrid(coords, -1 * coords, indexing='xy')
    sample_range = torch.sqrt(Y * Y + X * X)
    sample_angle = torch.arctan2(Y, X)
    sample_angle += torch.where(sample_angle < 0, 2. * torch.pi, 0.0) # wrap to 0-2pi

    return sample_range, sample_angle

def form_polar_range_grid(polar_resolution=0.2384, polar_pixel_shape=(400, 3360), dtype=None, device='cpu'):
    # Compute the range (m) captured by pixels in each azimuth
    # A pixels coordinates are the center of the pixel
    # Azimuths already contain known angles camputed in polar. Also means we don't need to deal with wobble
    # Pixel 0 in range is 0 m, so range is
    polar_range = (polar_pixel_shape[1] - 1) * polar_resolution

    # Compute the range value of each polar pixel
    # These values are the same for each batch item
    if dtype is None:
        range_coords = torch.linspace(0.0, polar_range, polar_pixel_shape[1], device=device)
    else:
        range_coords = torch.linspace(0.0, polar_range, polar_pixel_shape[1], dtype=dtype, device=device)

    # Return grid where each row is the range_coords
    range_grid = range_coords.unsqueeze(0).expand(polar_pixel_shape[0], -1)

    return range_grid