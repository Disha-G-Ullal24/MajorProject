import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import serial
import time
import numpy as np
import joblib
import os
import csv
from datetime import datetime
import pandas as pd

# ---------------- PARAMETERS ----------------
IMG_SIZE = (224, 224)
CNN_PATH = r"models/cnn_extractor.pth"
RF_PATH = r"models/rf_model.pkl"
IMAGE_PATH = "captured/grape_0.jpg"
NIR_FILE = "data/nir/nir_data.csv"  # contains 13 numeric columns
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
RESULTS_FILE = "grape_results.csv"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD CNN ----------------
base_model = models.mobilenet_v2(weights=None)
base_model.classifier = nn.Identity()
base_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
base_model.to(device)
base_model.eval()
print("✅ CNN loaded")

# ---------------- LOAD RANDOM FOREST ----------------
rf_model = joblib.load(RF_PATH)
print("✅ RF loaded")

# ---------------- LOAD NIR DATA ----------------
if not os.path.exists(NIR_FILE):
    raise FileNotFoundError(f"NIR file not found: {NIR_FILE}")
nir_df = pd.read_csv(NIR_FILE)
nir_features = nir_df.select_dtypes(include=np.number).values  # numeric columns only
print(f"✅ NIR loaded, shape={nir_features.shape}")

# ---------------- IMAGE TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- CONNECT TO ARDUINO ----------------
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("✅ Arduino connected")
except:
    arduino = None
    print("⚠️ Arduino not connected")

# ---------------- LOAD IMAGE ----------------
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found:", IMAGE_PATH)
    exit()

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ---------------- EXTRACT CNN FEATURES ----------------
with torch.no_grad():
    cnn_features = base_model(img_tensor).cpu().numpy()  # shape (1,1280)

# ---------------- COMBINE CNN + NIR ----------------
# Pick first row of NIR (adjust as needed)
nir_row = nir_features[0].reshape(1, -1)  # shape (1,13)
features = np.concatenate([cnn_features, nir_row], axis=1)  # shape (1,1293)

# ---------------- CHECK FEATURE SIZE ----------------
if features.shape[1] != rf_model.n_features_in_:
    raise ValueError(f"Feature size mismatch! RF expects {rf_model.n_features_in_}, got {features.shape[1]}")

# ---------------- PREDICTION ----------------
pred_label = rf_model.predict(features)[0]
class_names = ['BadGrapes', 'GoodGrapes']
result = class_names[pred_label]
print("Prediction:", result)

# ---------------- SAVE CSV ----------------
file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['ImageName', 'Prediction', 'Timestamp'])
    writer.writerow([os.path.basename(IMAGE_PATH), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
print("✅ Result saved")

# ---------------- SEND TO ARDUINO ----------------
if arduino:
    arduino.write(b'G' if result=="GoodGrapes" else b'B')
    print("✅ Sent to Arduino:", "G" if result=="GoodGrapes" else "B")
