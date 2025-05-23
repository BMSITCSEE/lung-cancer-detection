from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from src.explain import generate_gradcam
import io

app = Flask(__name__)

# Load model
model = torch.load('models/densenet_model.pth', map_location=torch.device('cpu'))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_file = request.files["image"]
        image = Image.open(image_file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        tensor = transform(image).unsqueeze(0)
        output = model(tensor)
        _, predicted = torch.max(output, 1)

        return jsonify({"prediction": int(predicted.item())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
