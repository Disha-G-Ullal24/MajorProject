import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

MODEL_PATH = "../models/grape_mobilenetv2.pth"
IMG_PATH = "../data/train/good/grape_0.jpg"  # Change to any test image
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

img = Image.open(IMG_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = torch.sigmoid(model(img_tensor))
pred = output.item()
label = "GOOD" if pred > 0.5 else "BAD"
print(f"Prediction: {label} (Confidence: {pred:.2f})")
