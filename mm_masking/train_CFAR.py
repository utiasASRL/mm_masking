import argparse
import torch
from cfar_dataset import CFARDataset
from torch.utils.data import Dataset, DataLoader
from cfar_policy import LearnCFARPolicy
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pylgmath import se3op
import os
import neptune
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported

def train_policy(model, iterator, opt, regularizer=0.0, device='cpu', neptune_run=None, epoch=None):
    model.train()
    loss_hist = []

    for i_batch, batch in enumerate(iterator):
        print("Batch: ", i_batch)
        # Load in data
        fft_data = batch['fft_data'].to(device)
        cfar_mask = batch['cfar_mask'].to(device)

        if neptune_run is not None and epoch == 0:
            fig = plt.figure()
            plt.imshow(cfar_mask[0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(location='top', shrink=0.5)
            neptune_run["train/gt_mask"].append(fig, step=epoch, name=("gt mask " + str(epoch)))
            plt.close()

        # Zero grad
        opt.zero_grad()

        # Get prediction
        mask_pred = model(fft_data, neptune_run=neptune_run, epoch=epoch)

        # Compute loss
        criterion = torch.nn.BCELoss()
        loss = criterion(mask_pred, cfar_mask)

        # Compute the derivatives
        loss.sum().backward()

        # Take step
        opt.step()
        
        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)

    mean_loss = np.mean(loss_hist)
    return mean_loss

def validate_policy(model, iterator, device='cpu', verbose=False, binary=False):
    model.eval()
    norm_err_list = []

    with torch.no_grad():
        for i_batch, batch in enumerate(iterator):
            # Load in data
            fft_data = batch['fft_data'].to(device)
            cfar_mask = batch['cfar_mask'].to(device)

            mask_pred = model(fft_data, binary=binary)

            criterion = torch.nn.BCELoss(reduction='none')
            loss = criterion(mask_pred, cfar_mask)
            #loss.mean().backward()

            # Compute RMSE
            norm_err_list.append(loss.detach().cpu().numpy())

    avg_norm = np.mean(norm_err_list)
    return avg_norm

def main(args):
    run = neptune.init_run(
        project="lisusdaniil/cfar-masking",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjljOGQ1ZC1lNDE3LTQxYTQtOGNmMS1kMWY0NDcyY2IyODQifQ==",
        name="CFAR",
        mode="async",
    )

    params = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # Dataset params
        "num_train": 1,
        "num_test": 1,
        "random": False,
        "float_type": torch.float64,
        "use_gt": False,
        "map_sensor": "radar",
        "loc_sensor": "radar",
        "log_transform": False,      # True or false for log transform of fft data
        "normalize": "none",        # Options are "minmax", "standardize", and none
                                    # happens after log transform if log transform is true

        # Iterator params
        "batch_size": 1,
        "shuffle": True,

        # Training params
        "num_epochs": 500,
        "learning_rate": 1e-5,
        "leaky": False,   # True or false for leaky relu
        "dropout": 0.0,   # Dropout rate, set 0 for no dropout
        "regularizer": 0.0, # Regularization parameter, set 0 for no regularization
        "init_weights": True, # True or false for manually initializing weights
        
        # Model setup
        "network_input": "cartesian", # Options are "cartesian" and "polar", what the network takes in
        "network_output": "polar", # Options are "cartesian" and "polar"
        "binary_inference": False, # Options are True and False, whether the mask is binary or not during inference
    }

    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

    train_dataset = CFARDataset(gt_data_dir=args.gt_data_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=train_loc_pairs,
                                        random=params["random"],
                                        num_samples=params["num_train"],
                                        float_type=params["float_type"],
                                        normalization_type=params["normalize"],
                                        log_transform=params["log_transform"],
                                        use_gt=params["use_gt"],)
    test_dataset = CFARDataset(gt_data_dir=args.gt_data_dir,
                                        radar_dir=args.radar_dir,
                                        loc_pairs=train_loc_pairs,
                                        random=params["random"],
                                        num_samples=params["num_train"],
                                        float_type=params["float_type"],
                                        normalization_type=params["normalize"],
                                        log_transform=params["log_transform"],
                                        use_gt=params["use_gt"],)

    print("Dataset created")
    print("Number of training examples: ", len(train_dataset))
    print("Number of validation examples: ", len(test_dataset))

    torch.set_default_dtype(params["float_type"])

    # Form iterators
    training_iterator = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=2)
    validation_iterator = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=2)
    print("Dataloader created")

    # Initialize policy
    policy = LearnCFARPolicy(network_input=params["network_input"],
                             network_output=params["network_output"],
                             leaky=params["leaky"], dropout=params["dropout"],
                             float_type=params["float_type"], device=params["device"],
                             init_weights=params["init_weights"])
    policy = policy.to(device=params["device"])

    opt = torch.optim.Adam(policy.parameters(), lr=params["learning_rate"])
    #opt = torch.optim.SGD(policy.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0, maximize=False)

    print("Policy and optimizer created")

    #opt = torch.optim.SGD(policy.parameters(), lr=learning_rate)
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
    result_dir = 'results/' + 'cfar_mask' + '/learn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Train policy
    loss_hist = []
    val_hist = []
    # Eval policy before training
    avg_norm = validate_policy(policy, validation_iterator, device=params["device"], binary=params["binary_inference"])
    #avg_norm = 1000
    best_norm = avg_norm
    val_hist.append(avg_norm)

    print("Norm before training: ", avg_norm)
    for epoch in range(params["num_epochs"]):
        print ('EPOCH ', epoch)

        # Train the driving policy
        if epoch % 5 == 0 or epoch == params["num_epochs"] - 1 or epoch == 0:
            neptune_run = run
        else:
            neptune_run = None
        mean_loss = train_policy(policy, training_iterator, opt, 
                                 params["regularizer"], device=params["device"],
                                 neptune_run=neptune_run, epoch=epoch)
        loss_hist.append(mean_loss)

        # Validate the driving policy
        print("Validating")
        avg_norm = validate_policy(policy, validation_iterator,
                                   device=params["device"], binary=params["binary_inference"])
        val_hist.append(avg_norm)

        if avg_norm < best_norm or epoch == 0:
            print("Saving best policy")
            best_norm = avg_norm
            torch.save(policy.state_dict(), 'best_policy.pt')
        print("Average norm: ", avg_norm)
        print("Best norm: ", best_norm)

        #scheduler.step()
        run[npt_logger.base_namespace]["epoch/loss"].append(mean_loss.item())
        run[npt_logger.base_namespace]["epoch/acc"].append(avg_norm.item())

        #npt_logger.save_checkpoint()

    # Do final validation using the best policy
    policy.load_state_dict(torch.load('best_policy.pt'))
    avg_norm = validate_policy(policy, validation_iterator, device=params["device"], verbose=True, binary=params["binary_inference"])
    print("Best average norm: ", avg_norm)
    
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--radar_dir", help="directory of training data", default='../data/radar')

    args = parser.parse_args()

    main(args)