import argparse
import torch
from masking_dataset import MaskingDataset
from torch.utils.data import Dataset, DataLoader
from masking_policy import MaskingPolicy
import time
import os.path as osp
import cv2
import numpy as np

def main(args):
    # Load in all ground truth data based on the localization pairs provided in 
    train_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]
    val_loc_pairs = [["boreas-2020-11-26-13-58", "boreas-2020-12-04-14-00"]]

    training_dataset = MaskingDataset(gt_data_dir=args.gt_data_dir,
                                        pc_dir=args.pc_dir,
                                        radar_dir=args.cart_dir,
                                        loc_pairs=train_loc_pairs,
                                        batch_size=args.batch_size,
                                        shuffle=False)

    # Load in a single training example
    for i, (loc_data, map_data, T_lm_gt) in enumerate(training_dataset):
        if i < 9:
            continue
        map_cart = map_data['cart']
        map_pc = map_data['pc'][:,0:3]
        loc_cart = loc_data['cart']
        loc_pc = loc_data['pc'][:,0:3]
        loc_time = loc_data['timestamp']
        map_time = map_data['timestamp']

        lidar_dir = '/raid/dli/mm_masking/data/pointclouds/lidar/boreas-2020-11-26-13-58'
        # Load in lidar pointcloud
        lidar_file = osp.join(lidar_dir, '1606417109739990.bin')
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 6)[:,0:3]

        # Define 4x4 transformation matrix from radar to lidar as random
        T_radar_lidar = np.array([[0.6829738634663442554, 0.7304428121501747029, 0.0, 0.0], [0.7304428121501747029, -0.6829738634663442554, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # Transform lidar pointcloud to radar frame
        lidar = np.matmul(T_radar_lidar[:3, :3], lidar.T).T + T_radar_lidar[:3, 3]

        lidar_vis = visualize_lidar_pointcloud(lidar.copy(), "red")
        
        # Load in mask file corresponding to localization
        map_masked = cv2.imread(osp.join(args.mask_dir, train_loc_pairs[0][0], str(map_time) + '.png'))
        loc_masked = cv2.imread(osp.join(args.mask_dir, train_loc_pairs[0][1], str(loc_time) + '.png'))
        # Convert mask array to only 0's and 1's, where 0's are masked out
        loc_masked = np.where(loc_masked > 0, 1, 0)
        map_masked = np.where(map_masked > 0, 1, 0)
        # Updated masked cart images
        loc_masked = np.multiply(np.array(loc_cart), loc_masked)
        map_masked = np.multiply(np.array(map_cart), map_masked)

        # Extract mask that would generate masked from cart images
        loc_mask = loc_masked - loc_cart
        loc_mask = np.where(np.abs(loc_mask) > 0, 255, 0)
        map_mask = map_masked - map_cart
        map_mask = np.where(np.abs(map_mask) > 0, 255, 0)

        # Represent pointclouds as images
        loc_pc_vis = visualize_pointcloud(loc_pc.copy(), "green")
        map_pc_vis = visualize_pointcloud(map_pc.copy(), "red")

        icp_result = gen_fake_icp(loc_pc, map_pc, T_lm_gt)

        lidar_icp_result = gen_fake_radar_lidar_icp(loc_pc, lidar, T_lm_gt)

        # Save data
        # Save original cart images
        cv2.imwrite(osp.join(args.output_dir, 'loc_cart_orig.png'), loc_cart)
        cv2.imwrite(osp.join(args.output_dir, 'map_cart_orig.png'), map_cart)
        # Save masked cart images
        cv2.imwrite(osp.join(args.output_dir, 'loc_cart_masked.png'), loc_masked)
        cv2.imwrite(osp.join(args.output_dir, 'map_cart_masked.png'), map_masked)
        # Save masks
        cv2.imwrite(osp.join(args.output_dir, 'loc_mask.png'), loc_mask)
        cv2.imwrite(osp.join(args.output_dir, 'map_mask.png'), map_mask)
        # Save pointclouds
        cv2.imwrite(osp.join(args.output_dir, 'loc_pc.png'), loc_pc_vis)
        cv2.imwrite(osp.join(args.output_dir, 'map_pc.png'), map_pc_vis)
        # Save lidar pointcloud
        cv2.imwrite(osp.join(args.output_dir, 'lidar.png'), lidar_vis)
        # Save ICP result
        cv2.imwrite(osp.join(args.output_dir, 'icp_result.png'), icp_result)
        cv2.imwrite(osp.join(args.output_dir, 'lidar_icp_result.png'), lidar_icp_result)

        break

def visualize_lidar_pointcloud(pc, pc_color, ref_pc=None):
    # Save x and y coordinates of pointcloud to image
    # pc: (N, 3) array of pointcloud 
    # Returns: (H, W, 3) array of image with color values at x and y coordinates of pointcloud
    H = 640
    W = 640
    img = np.zeros((H, W, 3))
    # Map pointcloud to image bounds
    buffer = 125
    max_x = np.max(pc[:, 0]) - buffer
    min_x = np.min(pc[:, 0]) + buffer
    max_y = np.max(pc[:, 1]) - buffer
    min_y = np.min(pc[:, 1]) + buffer

    pc[:, 0] = (pc[:, 0] - min_x) / (max_x - min_x) * W
    pc[:, 1] = (pc[:, 1] - min_y) / (max_y - min_y) * H

    for i in range(pc.shape[0]):
        x = int(pc[i, 0])
        y = int(pc[i, 1])
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        # To improve visibility, plot x,y coordinate and the 4 surrounding pixels
        if pc_color == "green":
            img[y-1:y+1, x-1:x+1] = [0, 255, 0]
        elif pc_color == "blue":
            # Assign an azure blue colour
            img[y-1:y+1, x-1:x+1] = [255, 255, 0]
        elif pc_color == "red":
            img[y-1:y+1, x-1:x+1] = [0, 0, 255]
    
    # Rotate image 90 degrees counter-clockwise
    img = np.rot90(img, 3)

    return img

def visualize_pointcloud(pc, pc_color, ref_pc=None):
    # Save x and y coordinates of pointcloud to image
    # pc: (N, 3) array of pointcloud 
    # Returns: (H, W, 3) array of image with color values at x and y coordinates of pointcloud
    H = 640
    W = 640
    img = np.zeros((H, W, 3))
    # Map pointcloud to image bounds
    if ref_pc is not None:
        max_x = np.max(ref_pc[:, 0])
        min_x = np.min(ref_pc[:, 0])
        max_y = np.max(ref_pc[:, 1])
        min_y = np.min(ref_pc[:, 1])
    else:
        max_x = np.max(pc[:, 0])
        min_x = np.min(pc[:, 0])
        max_y = np.max(pc[:, 1])
        min_y = np.min(pc[:, 1])

    # Center pointcloud to the average of the x and y coordinates
    pc[:, 0] = pc[:, 0] - np.mean(pc[:, 0])
    pc[:, 1] = pc[:, 1] - np.mean(pc[:, 1])

    pc[:, 0] = (pc[:, 0] - min_x) / (max_x - min_x) * W
    pc[:, 1] = (pc[:, 1] - min_y) / (max_y - min_y) * H

    for i in range(pc.shape[0]):
        x = int(pc[i, 0])
        y = int(pc[i, 1])
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        # To improve visibility, plot x,y coordinate and the 4 surrounding pixels
        if pc_color == "green":
            img[y-2:y+2, x-2:x+2] = [0, 255, 0]
        elif pc_color == "blue":
            # Assign an azure blue colour
            img[y-2:y+2, x-2:x+2] = [255, 255, 0]
        elif pc_color == "red":
            img[y-2:y+2, x-2:x+2] = [0, 0, 255]
    
    # Rotate image 90 degrees counter-clockwise
    img = np.rot90(img, 1)

    return img

def gen_fake_icp(loc_pc, map_pc, T_lm_gt):
    # Generate fake ICP result
    # loc_pc: (N, 3) array of pointcloud 
    # map_pc: (M, 3) array of pointcloud 
    # T_lm_gt: (4, 4) array of ground truth transformation matrix
    # Returns: (4, 4) array of fake ICP result

    # Generate a slightly off transformation matrix
    T_ml_fake = np.linalg.inv(T_lm_gt.copy())
    #T_ml_fake[:3, 3] += np.random.normal(0, 0.01, 3)
    # Translate T_ml_fake by 0.1m in x and y
    #T_ml_fake[:3, 3] += np.array([100.0, 100.0, 0])

    # Apply ground truth transformation to localization pointcloud
    loc_pc = np.matmul(T_ml_fake[:3, :3], loc_pc.T).T + T_ml_fake[:3, 3] - np.array([-7.0, 0, 0])
    # Save pointclouds
    loc_pc_vis = visualize_pointcloud(loc_pc.copy(), "green")
    map_pc_vis = visualize_pointcloud(map_pc.copy(), "red")

    # Overlay pointclouds
    overlay = cv2.addWeighted(loc_pc_vis, 1.0, map_pc_vis, 1.0, 0)

    return overlay

def gen_fake_radar_lidar_icp(radar_pc, lidar_pc, T_lm_gt):
    # Generate fake ICP result
    # loc_pc: (N, 3) array of pointcloud 
    # map_pc: (M, 3) array of pointcloud 
    # T_lm_gt: (4, 4) array of ground truth transformation matrix
    # Returns: (4, 4) array of fake ICP result

    # Generate a slightly off transformation matrix
    T_ml_fake = np.linalg.inv(T_lm_gt.copy())
    #T_ml_fake[:3, 3] += np.random.normal(0, 0.01, 3)
    # Translate T_ml_fake by 0.1m in x and y
    #T_ml_fake[:3, 3] += np.array([100.0, 100.0, 0])
    # Apply ground truth transformation to localization pointcloud
    radar_pc = np.matmul(T_ml_fake[:3, :3], radar_pc.T).T + T_ml_fake[:3, 3] - np.array([-12.0, -14.0, 0])
    # Save pointclouds
    loc_pc_vis = visualize_pointcloud(radar_pc.copy(), "green")
    map_pc_vis = visualize_lidar_pointcloud(lidar_pc.copy(), "red")
    

    # Overlay pointclouds
    overlay = cv2.addWeighted(loc_pc_vis, 1.0, map_pc_vis, 1.0, 0)

    return overlay


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='../dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='../dataset/val')
    
    parser.add_argument("--gt_data_dir", help="directory of training data", default='../data/localization_gt')
    parser.add_argument("--pc_dir", help="directory of training data", default='../data/pointclouds')
    parser.add_argument("--cart_dir", help="directory of training data", default='../data/cart')
    parser.add_argument("--mask_dir", help="directory of training data", default='../data/mask')

    parser.add_argument("--output_dir", help="directory of training data", default='../data/diagram_imgs')

    args = parser.parse_args()

    main(args)