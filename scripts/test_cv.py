import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Paths (relative to THIS file) ---
THIS_DIR = os.path.dirname(__file__)                 # .../MajorProject/scripts
ROOT_DIR = os.path.dirname(THIS_DIR)                # .../MajorProject
MODEL_PATH = os.path.join(ROOT_DIR, "models", "grape_mobilenetv2.pth")

# ✅ Change this to the captured image path
IMG_PATH = os.path.join(THIS_DIR, "captured", "grape_0.jpg")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # Binary output
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       # Scale to [0,1]
])

# --- Load & preprocess image ---
if not os.path.exists(IMG_PATH):
    print(f"⚠️ Image not found: {IMG_PATH}")
    exit()

img = Image.open(IMG_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim

# --- Predict ---
with torch.no_grad():
    output = torch.sigmoid(model(img_tensor))
    confidence = output.item()
    pred_idx = 1 if confidence > 0.5 else 0

# --- Class labels ---
class_labels = {0: "Bad Grape", 1: "Good Grape"}
print(f"Predicted label: {class_labels[pred_idx]} (Confidence: {confidence:.3f})")
