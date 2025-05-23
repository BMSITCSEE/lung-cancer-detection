import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import models
from src.dataloader import LungDataset
from src.augment import get_transforms
from src.train import train_model
import os
import glob

# === CONFIG ===
data_dir = "data/train/"
epochs = 10
batch_size = 8
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATASET ===
image_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
labels = [0 if "benign" in path else 1 for path in image_paths]  # example labeling rule

dataset = LungDataset(image_paths, labels, transform=None)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === MODEL ===
model = models.densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model = model.to(device)

# === TRAIN ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_model(model, dataloader, epochs, criterion, optimizer, device)

# === SAVE ===
torch.save(model, "models/densenet_model.pth")
print("Model saved.")
