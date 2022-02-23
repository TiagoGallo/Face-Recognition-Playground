from .datasets import Traininig_Dataset, Validation_Dataset

from torchvision import transforms as T
from torch.utils.data import DataLoader

import os

def load_data(Train_dataset_path, Test_dataset_path,
              train_batch_size, test_batch_size):
    # Load training data
    train_transforms = T.Compose([
                T.Resize((112, 112)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    train_dataset = Traininig_Dataset(Train_dataset_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                            shuffle=True, num_workers=4)

    num_classes = train_dataset.num_classes

    test_transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_loaders = []

    for dataset_name in os.listdir(Test_dataset_path):
        test_dataset = Validation_Dataset(os.path.join(Test_dataset_path, dataset_name), dataset_name, transforms=test_transform)
        test_loaders.append(DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False))

    return train_loader, test_loaders, num_classes