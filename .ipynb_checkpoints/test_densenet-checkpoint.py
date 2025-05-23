import torch
from torch.utils.data import DataLoader
from src.dataloader import LungDataset
from src.evaluate import evaluate_model
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("models/densenet_model.pth", map_location=device)

# === TEST DATA ===
test_dir = "data/test/"
image_paths = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
labels = [0 if "benign" in path else 1 for path in image_paths]

test_dataset = LungDataset(image_paths, labels, transform=None)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

evaluate_model(model, test_loader, device)
