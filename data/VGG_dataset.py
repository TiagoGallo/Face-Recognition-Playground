import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VGGDataset(Dataset):
    """VGG Faces dataset."""

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