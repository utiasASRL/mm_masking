import numpy as np
import torch
from scipy.signal import find_peaks
from dICP.diff_nn import diff_nn

def load_pc_from_file(file_path, to_type=torch.float64, to_device='cpu', flip_y=False):
    pc = np.fromfile(file_path, dtype=np.float32)
    pc = pc.reshape((len(pc) // 6, 6))
    # For some dumb reason, the map point cloud is flipped in the y-axis
    if flip_y:
        pc[:, 1] = -pc[:, 1]
        pc[:, 4] = -pc[:, 4]
    # Send to device
    pc = torch.from_numpy(pc).to(to_device)
    # Convert to type
    pc = pc.type(to_type)
    return pc

def load_radar(raw_data):
    time_convert = 1000
    encoder_conversion = 2 * np.pi / 5600
    N = raw_data.shape[0]
    timestamps = np.frombuffer(raw_data[:,:8].tobytes(), dtype=np.int64) * time_convert
    azimuths = np.frombuffer(raw_data[:,8:10].tobytes(), dtype=np.uint16) * encoder_conversion
    range_bins = raw_data.shape[1] - 11
    fft_data = np.divide(raw_data[:,11:], 255.0, dtype=np.float32)
    return fft_data, azimuths, timestamps

def extract_pc_parallel(raw_scans, res, azimuth_times, azimuth_angles, width=101, minr=2.0, maxr=80.0, guard=5, a_thresh=1.0, b_thresh=0.09):
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

    # Precompute peaks
    #raw_scan_peaks = find_scan_peaks(raw_scans)
    raw_scan_peaks = raw_scans

    """
    thres_mask = raw_scans > thres_full

    raw_scan_thres = raw_scans * thres_mask
    raw_scan_peaks_thres = raw_scan_peaks * thres_mask

    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(thres_mask[0].detach().cpu().numpy(), cmap='gray')
    plt.savefig("thres_mask.png")
    plt.figure()
    plt.imshow(raw_scans[0].detach().cpu().numpy(), cmap='gray')
    plt.savefig("raw_scans.png")
    plt.figure()
    plt.imshow(raw_scan_peaks[0].detach().cpu().numpy(), cmap='gray')
    plt.savefig("raw_scan_peaks.png")
    plt.figure()
    plt.imshow(raw_scan_thres[0].detach().cpu().numpy(), cmap='gray')
    plt.savefig("raw_scan_thres.png")
    plt.figure()
    plt.imshow(raw_scan_peaks_thres[0].detach().cpu().numpy(), cmap='gray')
    plt.savefig("raw_scan_peaks_thres.png")
    dsafdafs
    """


    #raw_scan_peaks = raw_scans

    # Compute threshold mask
    steep_fact = 10.0
    thres_mask_raw = 0.5 * torch.tanh(steep_fact * (raw_scan_peaks - thres_full) + 2.5) + 0.5
    thres_mask = torch.hardshrink(thres_mask_raw, lambd=0.99)
    #thres_mask = thres_mask/thres_mask

    #thres_mask = raw_scan_peaks > thres_full
    #thres_mask = thres_mask > 0.0

    # Peak points now contains the rho values of all peaks
    #peak_points = res * torch.ones(raw_scan.shape) * torch.arange(raw_scan.shape[1]) * thres_mask
    thres_points = res * torch.ones(raw_scans.shape, device=device) * torch.arange(raw_scans.shape[2], device=device) * thres_mask

    # Compute mean of non zero blobs
    #peak_points = mean_peaks_parallel(thres_points)
    peak_points = mean_peaks_parallel_fast(thres_points)
    #peak_points = thres_points

    # Assemble all points, with azimuth angles and azimuth shapes as the other two dimensions

    azimuth_angles_mat = azimuth_angles.unsqueeze(2).repeat(1, 1, raw_scans.shape[2])
    azimuth_times_mat = azimuth_times.unsqueeze(2).repeat(1, 1, raw_scans.shape[2])
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
        pc_list.append(pc_ii)

    return pc_list

def extract_pc(raw_scan, res, azimuth_times, azimuth_angles, width=101, minr=2.0, maxr=80.0, guard=5, a_thresh=1.0, b_thresh=0.09):
    pointcloud = []
    width = width + 1 if width % 2 == 0 else width
    w2 = width // 2
    
    mincol = max(0, int(minr / res + w2 + guard + 1))
    maxcol = min(raw_scan.shape[1], int(maxr / res - w2 - guard))
    col_range = torch.arange(mincol, maxcol)


    left_start_idx = col_range - w2 - guard
    left_end_idx = col_range - guard
    left = torch.zeros(raw_scan.shape[0], len(left_start_idx))
    for idx, (start, end) in enumerate(zip(left_start_idx, left_end_idx)):
        left[:, idx] = torch.sum(raw_scan[:,start:end], axis=1)

    #left = np.array([np.sum(raw_scan[:,start:end], axis=1) for start, end in zip(left_start_idx, left_end_idx)]).T
    right_start_idx = col_range + guard + 1
    right_end_idx = col_range + w2 + guard + 1
    right = torch.zeros(raw_scan.shape[0], len(right_start_idx))
    for idx, (start, end) in enumerate(zip(right_start_idx, right_end_idx)):
        right[:, idx] = torch.sum(raw_scan[:,start:end], axis=1)
    
    #right = torch.array([torch.sum(raw_scan[:,start:end], axis=1) for start, end in zip(right_start_idx, right_end_idx)]).T
    stat = torch.maximum(left, right) / w2  # GO-CFAR
    thres = a_thresh * stat + b_thresh

    # Save full sized threshold map
    thres_full = 1000*torch.ones(raw_scan.shape)
    thres_full[:, col_range] = thres

    # Compute threshold mask
    #thres_mask1 = raw_scan > thres_full

    #thres_mask = torch.logical_and(raw_scan, thres_mask1)
    steep_fact = 5.0
    thres_mask_raw = 0.5 * torch.tanh(steep_fact * (raw_scan - thres_full) - 2.0) + 0.5
    # Normalize raw mask
    #thres_mask = torch.relu(thres_mask_raw - 0.95) / torch.max(thres_mask_raw - 0.95)

    thres_mask = torch.hardshrink(thres_mask_raw, lambd=0.95)


    
    """
    # Initialize tensors
    my_tensor = torch.tensor([0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 4, 5, 8, 0, 0], dtype=torch.float, requires_grad=True)
    out_tensor = torch.zeros_like(my_tensor)

    # Set parameters
    slope = 100
    window_size = 3
    res_col_range = torch.arange(2, my_tensor.shape[0]-2)

    # Compute which means to keep
    tanh_slope = torch.tanh(slope * my_tensor[res_col_range-1]) * torch.tanh(slope * my_tensor[res_col_range]) * torch.tanh(slope * my_tensor[res_col_range+1])
    # Compute means and isolate desired indices
    out_tensor[res_col_range] = tanh_slope * torch.mean( my_tensor[res_col_range].unfold(0, window_size, 1), dim=1)

    print(tanh_slope.shape)
    print(torch.mean( my_tensor.unfold(0, window_size, 1), dim=1).shape)
    dfsads

    # Print results and check that grad exists
    out_tensor.sum().backward()

    print(out_tensor)
    print(my_tensor.grad is not None)


    dsfaasd
    """




    #thres_mask1 = thres_mask > 0.1

    #thres_mask2 = torch.logical_and(thres_mask, thres_mask)

    """
    thres_mask2.sum().backward()
    print(torch.max(raw_scan.grad))
    faddfas

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(thres_mask1.detach().numpy(), cmap='gray')
    plt.savefig("thres_mask1.png")
    plt.figure()
    plt.imshow(thres_mask.detach().numpy(), cmap='gray')
    plt.savefig("thres_mask.png")
    dsafdafs

    """

    # Peak points now contains the rho values of all peaks
    #peak_points = res * torch.ones(raw_scan.shape) * torch.arange(raw_scan.shape[1]) * thres_mask
    thres_points = res * torch.ones(raw_scan.shape) * torch.arange(raw_scan.shape[1]) * thres_mask

    # Compute mean of non zero blobs
    peak_points = mean_peaks(thres_points)

    #peak_points = torch.zeros_like(thres_points)
    #for ii in range(peak_points.shape[0]):
    #    peak_points[ii] = mean_peaks(thres_points[ii])

    """
    # Set parameters for grouping peaks
    slope = 100
    window_size = 9

    assert window_size % 2 == 1, "Window size must be odd"

    win_pad = (window_size-1)//2
    col_range_pad = torch.arange(mincol-win_pad, maxcol+win_pad)
    # Compute which means to keep
    tanh_slope = torch.ones(peak_points[:, col_range].shape)

    for ii in range(window_size):
        tanh_slope = tanh_slope * torch.tanh(slope * peak_points[:, col_range+ii-(window_size-1)//2])

    #tanh_slope = torch.tanh(slope * peak_points[:, col_range-1]) * torch.tanh(slope * peak_points[:, col_range]) * torch.tanh(slope * peak_points[:, col_range+1])
    # Compute means and isolate desired indices
    peak_points[:,col_range] = tanh_slope * torch.mean( peak_points[:,col_range_pad].unfold(1, window_size, 1), dim=2)

    
    with open('peak_pts.npy', 'wb') as f:
        np.save(f, peak_points.detach().numpy())
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(peak_points.detach().numpy(), cmap='gray')
    plt.savefig("peak_points.png")
    """

    # Assemble all points, with azimuth angles and azimuth shapes as the other two dimensions
    azimuth_angles_mat = azimuth_angles.unsqueeze(1).repeat(1, raw_scan.shape[1])
    azimuth_times_mat = azimuth_times.unsqueeze(1).repeat(1, raw_scan.shape[1])
    peak_pt_mat = torch.cat((peak_points.unsqueeze(-1), azimuth_angles_mat.unsqueeze(-1), azimuth_times_mat.unsqueeze(-1)), dim=-1)

    # Flatten peak_points along first two dimensions
    peak_pt_vec = peak_pt_mat.reshape(-1, 3)

    nonzero_indices = peak_pt_vec[:,0].nonzero(as_tuple=True)

    #new_tensor = torch.gather(peak_points, 0, nonzero_indices.squeeze())
    #print(nonzero_indices.shape)

    return peak_pt_vec[nonzero_indices]
    faddfas

    #thres_mask.sum().backward()
    #print(torch.max(raw_scan.grad))
    #faddfas

    for ii in range(peak_points.shape[1]):
        if peak_points[8, ii] > 0:
            print("peak_points[8, ii]: ", peak_points[8, ii])
    afsdfs

    i = 8
    peak_points = 0
    num_peak_points = 0

    for j in range(mincol, maxcol):
        if raw_scan[i, j] > thres[i, j-mincol]:
            peak_points += j
            num_peak_points += 1
        elif num_peak_points > 0:
            p0 = res * peak_points / num_peak_points
            p1 = azimuth_angles[i]
            p2 = azimuth_times[i]

            p = torch.tensor((p0, p1, p2), dtype=torch.float32, requires_grad=True).reshape(1, 3)
            #p = torch.tensor((1, 3), dtype=torch.float32, requires_grad=True)
            pointcloud.append(p)
            print("peak_points: ", peak_points)
            print("num_peak_points: ", num_peak_points)
            peak_points = 0
            num_peak_points = 0

    print("pc: ", pointcloud)

    # Assemble peak points with each cell valued at row number
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(thres_mask)
    #plt.savefig("thres_mask.png")
    #dsafdafs

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(peak_points)
    #plt.savefig("peak_points.png")
    #dsafdafs

    #num_peak_points = torch.zeros(raw_scan.shape[0])

    masked_scan = raw_scan * thres_mask

    masked_scan_i = masked_scan[8, :].reshape(1, masked_scan.shape[1])

    #masked_scan.sum().backward()
    #print(torch.max(raw_scan.grad))
    #faddfas

    for i in range(raw_scan.shape[0]):
        for j in range(mincol, maxcol):
            peak_points += j * (raw_scan[i, j] > thres[i, j-mincol])
            num_peak_points += (raw_scan[i, j] > thres[i, j-mincol])
            print(num_peak_points)
            fdasdfas
            if num_peak_points > 0:
                p0 = res * peak_points / num_peak_points
                p1 = azimuth_angles[i]
                p2 = azimuth_times[i]

                p = torch.tensor((p0, p1, p2), dtype=torch.float32, requires_grad=True).reshape(1, 3)
                pointcloud.append(p)
                peak_points = 0
                num_peak_points = 0

    assert len(pointcloud) > 0

    return torch.concatenate(pointcloud, axis=0)


def find_scan_peaks(raw_scan):

    # Apply a 1D convolution to the raw scan to find peaks
    #kernel = torch.ones((raw_scan.shape[0], raw_scan.shape[1], 1), device=raw_scan.device)
    #conv_scan = torch.nn.functional.conv1d(raw_scan, kernel, padding='same', stride=1)

    conv_scan = raw_scan.clone()
    
    """
    peaks = torch.zeros_like(raw_scan)
    count = 0
    for ii in range(raw_scan.shape[0]):
        for jj in range(raw_scan.shape[1]):
            for kk in range(1, raw_scan.shape[2] - 1):
                if raw_scan[ii, jj, kk] > raw_scan[ii, jj, kk-1] and raw_scan[ii, jj, kk] > raw_scan[ii, jj, kk+1]:
                    peaks[ii, jj, kk-5:kk+5] = raw_scan[ii, jj, kk-5:kk+5]

    return peaks
    """
    peak_mask = torch.zeros_like(raw_scan)

    peak_mask = (conv_scan > torch.roll(conv_scan, 1, dims=2)) * (conv_scan > torch.roll(conv_scan, -1, dims=2))
    peak_mask = peak_mask.bool()

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(peak_mask[0].detach().cpu().numpy(), cmap='gray')
    #plt.savefig("peak_mask.png")

    return raw_scan * peak_mask

def mean_peaks_parallel_fast(arr):
    res = torch.zeros_like(arr)
    
    zero_detect_arr = zero_detect(arr)

    #for ii in range(arr.shape[2]-1):
    #    res[:, :,ii] = arr[:,:,ii] * (1-zero_detect_arr[:, :,ii+1])
    res_forward = arr[:, :, :-1] * (1 - zero_detect_arr[:, :, 1:])

    # Get non-zero values in the array
    res_backward = arr[:, :, 1:] * (1 - zero_detect_arr[:, :, :-1])

    res[:, :, :-1] = res_forward + res_backward
    return res


def mean_peaks_parallel(arr):
    res = torch.zeros_like(arr)
    s = torch.zeros((arr.shape[0], arr.shape[1], arr.shape[2]-1), device=arr.device)
    n = torch.zeros((arr.shape[0], arr.shape[1], arr.shape[2]-1), device=arr.device)


    """
    z = np.array([1, 2, 0, 3]) # replace z1, z2, z3, ..., zn with your values
    n = np.zeros(len(z)) # initialize an array of zeros with length equal to z

    c = np.flip(np.cumprod(1 - z[:-1][::-1]))

    # Pad the cumulative product with 1 and reverse it
    c = np.concatenate(([1], c))[::-1]

    # Compute the element-wise product of z and c
    n = z * c

    # Compute the cumulative sum of n in reverse order
    n = np.flip(np.cumsum(np.flip(n)))

    print(z)
    print(n)


    fadsad
    """
    
    zero_detect_arr = zero_detect(arr)

    for ii in range(arr.shape[2]-1):
        s_next = zero_detect_arr[:, :,ii] * (s[:,:,ii-1] + arr[:, :,ii])
        n_next = zero_detect_arr[:, :,ii] * (n[:,:,ii-1] + 1)
        s[:,:,ii] = s_next
        n[:,:,ii] = n_next

    #short_zero_detect_arr = zero_detect_arr[:, :, :-1]
    #n[:, :, 0] = short_zero_detect_arr[:, :, 0]
    #n[:, :, 1:] = torch.cumsum(short_zero_detect_arr[:, :, 1:] * torch.flip(torch.cumprod(1 - short_zero_detect_arr[:, :, :-1], dim=-1), dims=[-1]), dim=-1)
    #n = n[:, :, :-1]

    #n[1:] = np.cumsum(z[1:] * np.flip(np.cumprod(1 - z[:-1]), axis=0))
    res[:, :, :-1] = s / (n + 1e-6) * (1-zero_detect_arr[:, :, 1:])
    
    return res

def mean_peaks_parallel_old(arr):
    res = torch.zeros_like(arr)
    s = torch.zeros(arr.shape[0], arr.shape[1])
    n = torch.zeros(arr.shape[0], arr.shape[1])
    
    zero_detect_arr = zero_detect(arr)

    for ii in range(arr.shape[2]-1):
        s = zero_detect_arr[:, :,ii] * (s + arr[:, :,ii])
        n = zero_detect_arr[:, :,ii] * (n + 1)
        res[:, :,ii] = s / (n+1e-6) * (1-zero_detect_arr[:, :,ii+1])
    return res

def mean_peaks(arr):
    res = torch.zeros_like(arr)
    s = torch.zeros(arr.shape[0])
    n = torch.zeros(arr.shape[0])
    
    for i in range(arr.shape[1]-1):
        s = zero_detect(arr[:,i]) * (s + arr[:,i])
        n = zero_detect(arr[:,i]) * (n + 1)
        res[:,i] = s / (n+1e-6) * (1-zero_detect(arr[:,i+1]))
    return res

def zero_detect(x, slope=5.0):
    return torch.tanh(slope*x)

def extract_pc2(raw_scan, res, azimuth_times, azimuth_angles, width=101, minr=2.0, maxr=80.0, guard=5, a_thresh=1.0, b_thresh=0.09):
    pointcloud = []
    rows, cols = raw_scan.shape
    if width % 2 == 0:
        width += 1
    w2 = np.floor(width / 2).astype(int)
    mincol = int(minr / res + w2 + guard + 1)
    if mincol > cols or mincol < 0:
        mincol = 0
    maxcol = int(maxr / res - w2 - guard)
    if maxcol > cols or maxcol < 0:
        maxcol = cols

    thres_array = np.zeros((rows, maxcol - mincol))

    for i in range(rows):
        azimuth = azimuth_angles[i]
        time = azimuth_times[i]
        polar_time = []

        peak_points = 0
        num_peak_points = 0

        for j in range(mincol, maxcol):
            left = 0
            right = 0
            for k in range(-w2 - guard, -guard):
                left += raw_scan[i, j + k]
            for k in range(guard + 1, w2 + guard + 1):
                right += raw_scan[i, j + k]
            # (statistic) estimate of clutter power
            # stat = (left + right) / (2 * w2)
            stat = max(left, right) / w2  # GO-CFAR
            thres = a_thresh * stat + b_thresh

            thres_array[i, j - mincol] = thres

            if raw_scan[i, j] > thres:
                peak_points += j
                num_peak_points += 1
            elif num_peak_points > 0:
                p = np.zeros((1, 3))
                p[0, 0] = res * peak_points / num_peak_points
                p[0, 1] = azimuth
                p[0, 2] = 0
                p = p.astype(np.float32)
                polar_time.append(p)
                peak_points = 0
                num_peak_points = 0

        if len(polar_time) > 0:
            polar_time = np.concatenate(polar_time, axis=0)
            polar_time[:, 2] = time
            pointcloud.append(polar_time)

    print(thres_array)
    print(thres_array.shape)
    fdsaads

    if len(pointcloud) > 0:
        pointcloud = np.concatenate(pointcloud, axis=0)
    else:
        pointcloud = np.zeros((0, 3), dtype=np.float32)
    return pointcloud

def pol_2_cart(pointcloud):
    rho = pointcloud[:, 0]
    phi = pointcloud[:, 1]
    
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    z = torch.zeros_like(rho)
    
    return torch.stack((x, y, z), axis=1)
