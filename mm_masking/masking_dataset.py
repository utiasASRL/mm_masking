from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

class InvalidDatasetException(Exception):
    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )

class MaskingDataset():

    def __init__(self, img_paths, img_labels, size_of_images):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.size_of_images = size_of_images
        if len(self.img_paths) != len(self.img_labels):
            raise InvalidDatasetException(self.img_paths, self.img_labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Load in localization cartesian image
        
        
        loc_data = {'cart' : cart_loc, 'pc' : pc_loc}
        map_data = {'cart' : cart_map, 'pc' : pc_map}

        return loc_data, map_data, T_lm_gt