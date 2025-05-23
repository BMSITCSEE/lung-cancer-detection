import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
from torchvision import transforms

class LungDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        image = np.expand_dims(image, axis=0)  # [1, D, H, W]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
