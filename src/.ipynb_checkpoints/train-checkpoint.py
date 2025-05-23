import torch
from torch import nn, optim
from tqdm import tqdm

def train_model(model, dataloader, epochs, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
