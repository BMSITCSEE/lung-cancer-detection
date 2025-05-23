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
- [LUNA16](https://luna16.grand-challenge.org)

## How to Run

```bash
1. Clone the Repository
git clone https://github.com/BMSITCSEE/lung-cancer-detection.git
cd lung-cancer-detection

2. Create and Activate Conda Environment
conda create -n lung python=3.8 -y
conda activate lung

3. Install Dependencies
pip install -r requirements.txt
If you don't have a requirements.txt, generate it with:

pip freeze > requirements.txt

4. Run these files
Train UNet (for segmentation preprocessing)
python src/train_unet.py

Train DenseNet (for 2D classification)
python src/train_densenet.py

Train 3D CNN (for 3D classification)
python src/train_3dcnn.py

Run evaluation:
python src/evaluate.py

Run explainability (e.g., Grad-CAM/SHAP):
python src/explain.py

5.Optional: If Using Jupyter Notebook
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
