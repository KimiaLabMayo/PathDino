import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
import os
import random

class CustomDatasetFolders(Dataset):
    def __init__(self, root_folders, transform=None):
        self.root_folders = root_folders
        self.transform = transform
        self.samples = self._get_samples()
        print(f"CustomDatasetFolders: {len(self.samples)} samples found.")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, 0

    def _get_samples(self):
        samples = []
        for folder_idx, folder_path in enumerate(self.root_folders):
            class_folders = os.listdir(folder_path)
            for class_folder in class_folders:
                print(f"CustomDatasetFolders: {folder_idx} {class_folder}")
                class_path = os.path.join(folder_path, class_folder)
                image_files = os.listdir(class_path)
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    samples.append(image_path)
        random.shuffle(samples)
        return samples