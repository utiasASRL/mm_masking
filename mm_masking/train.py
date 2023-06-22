import argparse
import torch
from masking_dataset import MaskingDataset
from torch.utils.data import Dataset, DataLoader
from masking_policy import LearnScalePolicy, LearnICPPolicy, LearnBFARPolicy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pylgmath import se3op
import os


def train_policy(model, iterator, opt, device='cpu'):
    model.train()
    loss_hist = []

    # Define type of loss function
    #loss_class = torch.nn.MSELoss()

    for i_batch, batch in enumerate(iterator):
        # Load in data
        batch_scan = batch['loc_data']
        batch_map = batch['map_data']
        batch_T = batch['transforms']
        batch_T_gt = batch_T['T_ml_gt'].to(device)
        batch_T_init = batch_T['T_ml_init'].to(device)

        # Zero grad
        opt.zero_grad()

        T_pred = model(batch_scan, batch_map, batch_T_init)

        # Compute loss
        #xi_wedge = torch.matmul(batch_T_gt, torch.inverse(T_pred)) - torch.eye(4, dtype=torch.float64, device=device)
        xi_wedge = torch.matmul(T_pred, torch.inverse(batch_T_gt)) - torch.eye(4, dtype=torch.float64, device=device)
        # Extract xi components
        xi_r = xi_wedge[:, 0:2, 3]
        xi_theta = xi_wedge[:, 1, 0].unsqueeze(-1)
        # Stack the xi_theta and xi_r
        xi = torch.cat((xi_theta, xi_r), dim=1)
        loss = torch.norm(xi, dim=1).mean()

        #loss = loss_class(T_pred, batch_T_gt)
        # Compute the derivatives
        loss.backward()

        #print("Loss: ", loss.item())
        print("Param grad: ", model.params.grad)
        print("Extract params at batch: ", i_batch, " ", model.params.data)

        # Take step
        opt.step()
        
        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)

        del loss, T_pred, xi, xi_r, xi_theta, xi_wedge, batch_T_gt, batch_T_init

    # Print params
    print("Extract params: ", model.params.data)

    mean_loss = np.mean(loss_hist)
    return mean_loss

def validate_policy(model, iterator, device='cpu', verbose=False):
    model.eval()
    norm_err_list = []

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

            if verbose:
                print("T_init: ", batch_T_init[0,:,:])
                print("T_pred: ", T_pred[0,:,:])
                print("T_gt: ", batch_T_gt[0,:,:])
                print("Error: ", err[0,:])
                print("Norm error: ", norm_err[0])

            # Compute RMSE
            norm_err_list.append(norm_err)

            del norm_err, T_pred
    
    avg_norm = np.mean(norm_err_list)
    return avg_norm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    # Dataset params
    #num_train = 1000
    #num_test = 50
    num_train = 10
    num_test = 10
    size_pc = -1
    random = False
    float_type = torch.float64
    # Iterator params
    batch_size = 10
    shuffle = True
    # Training params
    icp_type = "pt2pl" # Options are "pt2pt" and "pt2pl"
    task = "bfar" # Options are "scale", "icp", "bfar"
    num_epochs = 1

    # Param init
    # SCALE TASK
    if task == "scale":
        scale_init = 1.4
        true_scale = 1.2
        learning_rate = 0.001
        p_names = ['scale']

    # ICP TASK
    elif task == "icp":
        trim_init = 5.0
        huber_init = 2.0
        learning_rate = 0.001
        p_names = ['trim', 'huber']

    # BFAR TASK
    elif task == "bfar":
        a_init = 0.5
        b_init = 0.2

        a_init = 1.0
        b_init = 0.09

        learning_rate = 0.0001
        p_names = ['a', 'b']
    

    print("Parameters set")

    # Load in all ground truth data based on the localization pairs provided in 
    sensor = 'radar'
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

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
    test_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=val_loc_pairs,
                                        sensor=sensor,
                                        random=random,
                                        num_samples=num_test,
                                        size_pc=size_pc,
                                        float_type=float_type,
                                        use_gt=False)

    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(test_dataset))

    torch.set_default_dtype(float_type)
    #torch.autograd.set_detect_anomaly(True)

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    validation_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    print("Dataloader created")

    # Initialize policy
    if task == "scale":
        policy = LearnScalePolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device, scale_init=scale_init, true_scale=true_scale)
    elif task == "icp":
        policy = LearnICPPolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device, trim_init=trim_init, huber_init=huber_init)
    elif task == "bfar":
        policy = LearnBFARPolicy(size_pc=size_pc, icp_type=icp_type, float_type=float_type, device=device, a_init=a_init, b_init=b_init)
    policy = policy.to(device=device)

    #opt = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0, maximize=False)

    print("Policy and optimizer created")

    #opt = torch.optim.SGD(policy.parameters(), lr=learning_rate)
    # Set learning rate scheduler
    #scheduler = StepLR(opt, step_size=25, gamma=0.9)

    # Form result directory
    result_dir = 'results/' + task + '/learn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_naming = result_dir + '/' + icp_type

    # Train policy
    loss_hist = []
    val_hist = []
    param_hist = []
    best_param = policy.params.detach().cpu().numpy()
    # Eval policy before training
    avg_norm = validate_policy(policy, validation_iterator, device=device)
    best_norm = avg_norm
    val_hist.append(avg_norm)

    print("Norm before training: ", avg_norm)
    for epoch in range(num_epochs):
        print ('EPOCH ', epoch)

        # Train the driving policy
        mean_loss = train_policy(policy, training_iterator, opt, device=device)
        loss_hist.append(mean_loss)

        # Validate the driving policy
        avg_norm = validate_policy(policy, validation_iterator, device=device)
        val_hist.append(avg_norm)
        curr_params = policy.params.detach().cpu().numpy()
        if len(p_names) == 1:
            curr_params = np.array([curr_params])
        param_hist_val = np.concatenate((curr_params, np.array([avg_norm])), axis=0)
        param_hist.append(param_hist_val)

        if avg_norm < best_norm or epoch == 0:
            best_norm = avg_norm
            best_param = curr_params
            torch.save(policy.state_dict(), result_naming + '_best_policy.pt')
        print("Average norm: ", avg_norm)
        print("Current params: ", curr_params)
        print("Best norm: ", best_norm)
        print("Best params: ", best_param)

        #scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs - 1:
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

            plot_params(param_hist, p_names, result_naming)

            # Save the ab values
            np.save(result_naming + '_param_hist.npy', param_hist)

    # Do final validation using the best policy
    policy.load_state_dict(torch.load(result_naming + '_best_policy.pt'))
    avg_norm = validate_policy(policy, validation_iterator, device=device, verbose=True)
    print("Best average norm: ", avg_norm)
    plot_params(param_hist, p_names, result_naming)


def plot_params(param_hist, p_names, result_naming):
    p_hist = param_hist.copy()
    # Extract param history
    p_hist = np.array(p_hist)
    p_hist = p_hist[:,:-1]
    # If single dimensional param, add a dimension
    if len(p_names) == 1:
        p_hist = np.expand_dims(p_hist, axis=1)

    # Plot params over epoch
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(p_hist.shape[0]), p_hist[:,0], color='black', linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(p_names[0])
    plt.savefig(result_naming + '_p1_learn.png')

    if len(p_names) > 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(p_hist.shape[0]), p_hist[:,1], color='black', linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(p_names[1])
        plt.savefig(result_naming + '_p2_learn.png')

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