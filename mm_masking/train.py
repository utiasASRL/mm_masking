import argparse
import torch
from masking_dataset import MaskingDataset
from torch.utils.data import Dataset, DataLoader
from masking_policy import MaskingPolicy
import time

def main(args):

    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

    training_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        cart_dir=args.cart_dir,
                                        loc_pairs=train_loc_pairs,
                                        batch_size=args.batch_size,
                                        shuffle=False)
    validation_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        cart_dir=args.cart_dir,
                                        loc_pairs=val_loc_pairs,
                                        batch_size=args.batch_size,
                                        shuffle=False)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    masking_policy = MaskingPolicy().to(DEVICE)

    opt = torch.optim.Adam(masking_policy.parameters(), lr=args.lr)
    args.start_time = time.time()
    
    print (masking_policy)
    print (opt)
    print (args)

    args.class_dist = get_class_distribution(training_dataset, args)
    #print(np.ceil(args.batch_size*args.class_dist))
    #print(sum(np.ceil(args.batch_size*args.class_dist)))

    if args.weighted_loss:
        # Tell iterator desired distribution to be enforced in batches
        training_dataset.class_distribution = args.class_dist
        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    else:
        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    

    best_val_accuracy = 0 
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)

        # Train the driving policy
        train_discrete(masking_policy, training_iterator, opt, args)

        # Evaluate the driving policy on the validation set
        # Reshuffle validation set each epoch
        validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        avg_acc = test_discrete(masking_policy, validation_iterator, opt, args)
        
        # If the accuracy on the validation set is a new high then save the network weights 
        if avg_acc > best_val_accuracy:
            best_val_accuracy = avg_acc
            torch.save(masking_policy.state_dict(), args.weights_out_file)
        
    return masking_policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='./data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='./data/pointclouds')
    parser.add_argument("--cart_dir", help="directory of training data", default='./data/cart')

    args = parser.parse_args()

    main(args)