import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os, glob
import numpy as np
import nibabel as nib
from segmentation_models_pytorch import Unet
from tqdm import tqdm

# === Dataset ===
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

# === Config ===
img_dir = "data/seg/images/"
mask_dir = "data/seg/masks/"
epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_paths = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))

dataset = SegmentationDataset(img_paths, mask_paths)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# === Model ===
model = Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Train ===
model.train()
for epoch in range(epochs):
    running_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss / len(loader):.4f}")

torch.save(model, "models/unet_model.pth")
