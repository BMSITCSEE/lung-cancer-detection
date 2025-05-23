import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from src.dataloader import LungDataset
from src.train import train_model
from src.models import Simple3DCNN
import os, glob

# === CONFIG ===
data_dir = "data/train_3d/"
epochs = 10
batch_size = 2
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA ===
image_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
labels = [0 if "benign" in path else 1 for path in image_paths]

dataset = LungDataset(image_paths, labels)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === MODEL ===
model = Simple3DCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_model(model, loader, epochs, criterion, optimizer, device)
torch.save(model, "models/3dcnn_model.pth")
