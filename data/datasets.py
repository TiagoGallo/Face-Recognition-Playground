import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class Validation_Dataset(Dataset):
    '''
        Dataset subclass for loading validation datasets
    '''
    def __init__(self, root_dir, dataset_name, transforms=None):
        self.data_dir = root_dir
        self.name = dataset_name

        self.get_paths()
        self.transforms = transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_file = self.path_list[index]
        img = Image.open(img_file)

        im_out = self.transforms(img)
        return im_out

    def get_paths(self):
        self.path_list = []
        self.issame_list = []
        skipped_pairs = 0
        
        all_images = os.listdir(self.data_dir)
        positive_pairs_images = [img for img in all_images if 'True' in img]
        negative_pairs_images = [img for img in all_images if 'True' in img]

        # Get all positive pairs
        positive_loaded = set()
        for img in positive_pairs_images:
            _, _, pair_num, _ = img.split('_')

            if pair_num in positive_loaded:
                continue

            positive_loaded.add(pair_num)

            path0 = os.path.join(self.data_dir, f'True_pair_{pair_num}_1.jpg')
            path1 = os.path.join(self.data_dir, f'True_pair_{pair_num}_2.jpg')
            
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                self.path_list += (path0, path1)
                self.issame_list.append(True)
            else:
                skipped_pairs += 1

        # Get all positive pairs
        negative_loaded = set()
        for img in positive_pairs_images:
            _, _, pair_num, _ = img.split('_')

            if pair_num in negative_loaded:
                continue

            negative_loaded.add(pair_num)

            path0 = os.path.join(self.data_dir, f'False_pair_{pair_num}_1.jpg')
            path1 = os.path.join(self.data_dir, f'False_pair_{pair_num}_2.jpg')
            
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                self.path_list += (path0, path1)
                self.issame_list.append(False)
            else:
                skipped_pairs += 1
        
        if skipped_pairs > 0:
            print(f'Skipped {skipped_pairs} image pairs for {self.data_dir}')

class Traininig_Dataset(Dataset):
    """Training Face Recognition dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_list = self.process_dir()

    def process_dir(self):
        images_list = []
        self.num_classes = len(os.listdir(self.root_dir))

        for person in os.listdir(self.root_dir):
            person_num = int(person)

            person_dir = os.path.join(self.root_dir, person)
            for filename in os.listdir(person_dir):
                images_list.append((person_num, os.path.join(person_dir, filename)))

        return images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label, img_name = self.images_list[idx]

        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_name