import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scripts.train_unet import SegmentationDataset
import os, glob

model = torch.load("models/unet_model.pth", map_location=torch.device('cpu'))
model.eval()

img_paths = sorted(glob.glob("data/seg/images/*.nii.gz"))
mask_paths = sorted(glob.glob("data/seg/masks/*.nii.gz"))

test_dataset = SegmentationDataset(img_paths, mask_paths)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for x, y in test_loader:
    pred = torch.sigmoid(model(x)).detach().numpy()[0, 0]
    x = x.numpy()[0, 0]
    y = y.numpy()[0, 0]

    plt.subplot(1, 3, 1)
    plt.imshow(x, cmap='gray')
    plt.title("Input")
    plt.subplot(1, 3, 2)
    plt.imshow(y, cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(pred > 0.5, cmap='gray')
    plt.title("Prediction")
    plt.show()
    break
