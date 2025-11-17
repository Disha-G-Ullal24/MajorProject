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

# ---------------- PARAMETERS ----------------
IMG_SIZE = (224, 224)  # matches training
CNN_PATH = r"models/cnn_extractor.pth"
RF_PATH = r"models/rf_model.pkl"
IMAGE_PATH = "captured/grape_0.jpg"  # can be updated dynamically
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
RESULTS_FILE = "grape_results.csv"  # stores predictions

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD CNN FEATURE EXTRACTOR ----------------
base_model = models.mobilenet_v2(weights=None)
base_model.classifier = nn.Identity()
base_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
base_model.to(device)
base_model.eval()

# ---------------- LOAD RANDOM FOREST ----------------
rf_model = joblib.load(RF_PATH)

# ---------------- TRANSFORMS ----------------
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
    print("✅ Arduino connected.")
except:
    arduino = None
    print("⚠️ Arduino not connected!")

# ---------------- LOAD AND PROCESS IMAGE ----------------
if not os.path.exists(IMAGE_PATH):
    print("❌ Error: Image not found at", IMAGE_PATH)
    exit()

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ---------------- EXTRACT FEATURES ----------------
with torch.no_grad():
    features = base_model(img_tensor).cpu().numpy()

# ---------------- PREDICT USING RANDOM FOREST ----------------
pred_label = rf_model.predict(features)[0]
class_names = ['BadGrapes', 'GoodGrapes']  # must match dataset folder names
result = class_names[pred_label]

print(f"Result: {result}")

# ---------------- SAVE RESULT TO CSV ----------------
file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['ImageName', 'Prediction', 'Timestamp'])
    writer.writerow([os.path.basename(IMAGE_PATH), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

print(f"✅ Result saved to {RESULTS_FILE}")

# ---------------- SEND TO ARDUINO ----------------
if arduino:
    arduino.write(b'G' if result == "GoodGrapes" else b'B')
    print("✅ Sent to Arduino:", "G" if result == "GoodGrapes" else "B")
