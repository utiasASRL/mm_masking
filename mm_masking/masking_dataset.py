from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os.path as osp
from pyboreas.utils.odometry import read_traj_file2


class InvalidDatasetException(Exception):
    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )

class MaskingDataset():

    def __init__(self, gt_data_dir, pc_dir, cart_dir, loc_pairs, batch_size, shuffle):
        self.loc_pairs = loc_pairs

        # Loop through loc_pairs and load in ground truth loc poses
        self.gt_loc_poses = []
        self.loc_cart_paths = []
        self.loc_pc_paths = []
        self.map_cart_paths = []
        self.map_pc_paths = []
        for pair in loc_pairs:
            map_seq = pair[0]
            loc_seq = pair[1]
            
            # Load in ground truth localization to map poses
            gt_file = osp.join(gt_data_dir, map_seq + '_' + loc_seq + '.txt')
            gt_poses, loc_times, map_times, _, _ = read_traj_file2(gt_file)

            # Find localization cartesian images with names corresponding to pred_times
            loc_cart_paths = []
            loc_pc_paths = []
            map_cart_paths = []
            map_pc_paths = []
            incomplete_loc_times = []
            for idx, loc_time in enumerate(loc_times):
                # Load in localization paths
                loc_cart_path = osp.join(cart_dir, loc_seq, str(loc_time) + '.png')
                loc_pc_path = osp.join(pc_dir, loc_seq, str(loc_time) + '.bin')

                # Load in map paths
                map_cart_path = osp.join(cart_dir, map_seq, str(map_times[idx]) + '.png')
                map_pc_path = osp.join(pc_dir, map_seq, str(map_times[idx]) + '.bin')

                # Check if the paths exist, if not then save timestamps that don't exist
                if not osp.exists(loc_cart_path) or not osp.exists(loc_pc_path) or \
                     not osp.exists(map_cart_path) or not osp.exists(map_pc_path):
                    print('WARNING: Images or point clouds don\'t exist')
                    print('Localization time: ' + str(loc_time))
                    print('Map time: ' + str(map_times[idx]))

                    # Save timestamp that does not exist
                    incomplete_loc_times.append(loc_time)
                    continue
                else:
                    # Load in cartesian images
                    loc_cart_paths.append(loc_cart_path)
                    map_cart_paths.append(map_cart_path)
                    # Load in point cloud binaries
                    loc_pc_paths.append(loc_pc_path)
                    map_pc_paths.append(map_pc_path)

            # Remove ground truth localization to map poses that do not have corresponding images
            if len(incomplete_loc_times) != 0:
                print('WARNING: Number of localization cartesian images does not match number of localization poses')
                # Remove ground truth localization to map poses that do not have corresponding images
                for time in incomplete_loc_times:
                    index = loc_times.index(time)
                    loc_times.pop(index)
                    map_times.pop(index)
                    gt_poses.pop(index)

            self.loc_cart_paths += loc_cart_paths
            self.loc_pc_paths += loc_pc_paths
            self.map_cart_paths += map_cart_paths
            self.map_pc_paths += map_pc_paths
            self.gt_loc_poses += gt_poses

        # Assert that the number of all lists are the same
        assert len(self.gt_loc_poses) == len(self.loc_cart_paths) \
             == len(self.loc_pc_paths) == len(self.map_cart_paths) \
                == len(self.map_pc_paths)
        
        # Print number of total data points
        print('Number of total data points: ' + str(len(self.gt_loc_poses)))

    def __len__(self):
        return len(self.gt_loc_poses)

    def __getitem__(self, index):
        # Load in localization and map cartesian image
        loc_cart = Image.open(self.loc_cart_paths[index])
        map_cart = Image.open(self.map_cart_paths[index])
        # Load in localization and map point cloud
        loc_pc = np.fromfile(self.loc_pc_paths[index])
        map_pc = np.fromfile(self.map_pc_paths[index])

        # Load in ground truth localization to map pose
        T_lm_gt = self.gt_loc_poses[index]
        
        loc_data = {'cart' : loc_cart, 'pc' : loc_pc}
        map_data = {'cart' : map_cart, 'pc' : map_pc}

        return loc_data, map_data, T_lm_gt