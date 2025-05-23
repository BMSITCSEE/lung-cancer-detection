# Lung Cancer Detection using CNNs (Advanced)

This project detects lung cancer from CT scan images using deep learning techniques like:
- 3D CNNs
- UNet for segmentation
- DenseNet121 (Transfer Learning)
- Grad-CAM & SHAP for explainability

## Features
- Data preprocessing (HU normalization, resampling)
- Data augmentation using Albumentations
- Dockerized environment
- REST API with Flask
- Fully modular folder structure

## Dataset
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [LUNA16](https://luna16.grand-challenge.org)

## How to Run

```bash
docker build -t lung-cancer .
docker run -p 5000:5000 lung-cancer
