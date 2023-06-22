import argparse
import torch
from masking_dataset import MaskingDataset
from torch.utils.data import Dataset, DataLoader
from masking_policy import LearnICPPolicy, LearnBFARPolicy, LearnScalePolicy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pylgmath import se3op
from  matplotlib.colors import LinearSegmentedColormap
import os


def param_sweep(model, iterator, params, device='cpu', float_type=torch.float64):
    model.eval()
    #print("Current params: ", params)
    model.params = torch.nn.Parameter(torch.tensor(params, dtype=float_type, device=device))

    err_hist = []
    norm_hist = []
    err = 0.0

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            # Load in data
            batch_scan = batch['loc_data']
            batch_map = batch['map_data']
            batch_T = batch['transforms']
            batch_T_gt = batch_T['T_ml_gt'].to(device)
            batch_T_init = batch_T['T_ml_init'].to(device)

            T_pred = model(batch_scan, batch_map, batch_T_init)

            err = np.zeros((T_pred.shape[0], 6))
            norm_err = np.zeros((T_pred.shape[0], 1))
            for jj in range(T_pred.shape[0]):
                Err = batch_T_gt[jj].detach().cpu().numpy() @ np.linalg.inv(T_pred[jj].detach().cpu().numpy())
                err[jj] = se3op.tran2vec(Err).flatten()
                norm_err[jj] = np.linalg.norm(err[jj])

            #print("Batch: ", i_batch, " Norm error: ", np.mean(norm_err))
            norm_hist.append(np.mean(norm_err))

            del err, norm_err, T_pred

    avg_norm = np.mean(norm_hist)
    return avg_norm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    # Dataset params
    num_train = 10
    #num_test = 1
    size_pc = -1
    random = False
    float_type = torch.float64
    # Iterator params
    batch_size = 10
    shuffle = True
    # Training params
    icp_type = "pt2pl" # Options are "pt2pt" and "pt2pl"
    task = "icp" # Options are "scale", "icp", "bfar"

    # Param init
    # SCALE TASK
    if task == "scale":
        true_scale = 1.2
        p_names = ["scale"]
        scale_sweep = {"lims": [0.5, 2.0], "step": 0.01}
        # Form list of all possible parameter combinations
        scale_sweep = np.arange(scale_sweep["lims"][0], scale_sweep["lims"][1], scale_sweep["step"])
        param_list = []
        for scale in scale_sweep:
            param_list.append([scale])

    # ICP TASK
    elif task == "icp":
        p_names = ["trim", "huber delta"]
        trim_sweep = {"lims": [0.0, 20.0], "step": 0.5}
        huber_sweep = {"lims": [0.0, 10.0], "step": 0.2}
        # Form list of all possible parameter combinations
        trim_sweep = np.arange(trim_sweep["lims"][0], trim_sweep["lims"][1], trim_sweep["step"])
        huber_sweep = np.arange(huber_sweep["lims"][0], huber_sweep["lims"][1], huber_sweep["step"])
        param_list = []
        for trim in trim_sweep:
            for huber in huber_sweep:
                param_list.append([trim, huber])

    # BFAR TASK
    elif task == "bfar":
        p_names = ["a", "b"]
        a_sweep = {"lims": [0.0, 1.3], "step": 0.1}
        b_sweep = {"lims": [0.0, 0.2], "step": 0.02}
        # Form list of all possible parameter combinations
        a_sweep = np.arange(a_sweep["lims"][0], a_sweep["lims"][1], a_sweep["step"])
        b_sweep = np.arange(b_sweep["lims"][0], b_sweep["lims"][1], b_sweep["step"])
        param_list = []
        for a in a_sweep:
            for b in b_sweep:
                param_list.append([a, b])    

    print("Parameters set")
    print("Total number of parameter combinations: ", len(param_list))

    # Load in all ground truth data based on the localization pairs provided in 
    sensor = 'radar'
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    

    train_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=train_loc_pairs,
                                        sensor=sensor,
                                        random=random,
                                        num_samples=num_train,
                                        size_pc=size_pc,
                                        float_type=float_type,
                                        use_gt=False)

    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))

    torch.set_default_dtype(float_type)

    # Form result directory
    result_dir = 'results/' + task + '/sweep'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_naming = result_dir + '/' + icp_type

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    print("Dataloader created")

    # Initialize policy
    if task == "scale":
        policy = LearnScalePolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device, true_scale=true_scale)
    elif task == "icp":
        policy = LearnICPPolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device)
    elif task == "bfar":
        policy = LearnBFARPolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device)
    policy = policy.to(device=device)
    print("Policy created")

    sweep_result = []
    err_stack = []
    p1_stack = []
    p2_stack = []
    min_err = 10000.0
    for params in param_list:
        err = param_sweep(policy, training_iterator, params=params, device=device, float_type=float_type)
        print("Error for params: " + str(params) + " is: " + str(err))
        sweep_result.append((params, err))
        err_stack.append(err)
        p1_stack.append(params[0])
        if len(p_names) > 1:
            p2_stack.append(params[1])

        if err < min_err:
            min_err = err
            min_err_params = params

        if len(sweep_result) % 5 == 0 or len(sweep_result) == len(param_list) - 1:
            # Save results
            #if len(np.unique(np.array(p1_stack))) > 1 and len(p1_stack) % 2 == 0:
                #plot_sweep(p1_stack, p2_stack, err_stack, result_naming, p_names)
            with open(result_naming + '_sweep_result.pkl', 'wb') as f:
                pickle.dump(sweep_result, f)

    with open(result_naming + '_sweep_result.pkl', 'wb') as f:
        pickle.dump(sweep_result, f)

    # Plot final result
    print("Min err. norm: " + str(min_err))
    print("For params: " + str(min_err_params))
    plot_sweep(p1_stack, p2_stack, err_stack, result_naming, p_names)

def plot_sweep(p1_stack, p2_stack, err_stack, result_naming, p_names=["p1", "p2"]):

    # If there is only one parameter, plot a 2D line
    if len(p_names) == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(p1_stack, err_stack)
        plt.xlabel(p_names[0])
        plt.ylabel("Error (%)")
        plt.savefig(result_naming + "_sweep.png")
        return

    # Convert the lists to numpy arrays
    p1_arr = np.array(p1_stack)
    p2_arr = np.array(p2_stack)
    res_arr = np.array(err_stack)

    # Reshape the res array to create a grid
    p1_unique = np.unique(p1_arr)
    p2_unique = np.unique(p2_arr)
    res_grid = res_arr.reshape(len(p1_unique), len(p2_unique)).T

    res_grid[res_grid == 100] = None

    # Create a 3D plot of the result surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p1_grid, p2_grid = np.meshgrid(p1_unique, p2_unique)

    c = ["darkgreen", "palegreen", "orange", "lightcoral", "red", "darkred"]
    v = [0,.15,.4,0.6,.9,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    surf = ax.plot_surface(p1_grid, p2_grid, res_grid, cmap=cmap)
    ax.set_xlabel(p_names[0])
    ax.set_ylabel(p_names[1])
    ax.set_zlabel('Error (%)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=90, azim=-90)
    # Save figure
    plt.savefig(result_naming + '_sweep_result_surf.png')

    # Create a contour plot of the result surface
    fig = plt.figure()
    ax = fig.add_subplot(111)
    contour = ax.contourf(p1_unique, p2_unique, res_grid, cmap=cmap)
    ax.set_xlabel(p_names[0])
    ax.set_ylabel(p_names[1])
    cbar = plt.colorbar(contour)
    plt.savefig(result_naming + '_sweep_result_contour.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--radar_dir", help="directory of training data", default='../data/radar')

    args = parser.parse_args()

    main(args)