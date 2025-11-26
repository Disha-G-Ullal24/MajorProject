import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv
import joblib
from datetime import datetime
import serial
import time
import numpy as np
import pandas as pd

# ---------------- PARAMETERS -----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MajorProject/
CNN_PATH = os.path.join(BASE_DIR, "models", "cnn_extractor.pth")
RF_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
NIR_FILE = os.path.join(BASE_DIR, "data", "nir", "nir_data.csv")
IMG_SIZE = (224, 224)
RESULTS_FILE = os.path.join(BASE_DIR, "grape_results.csv")
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# ---------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD CNN FEATURE EXTRACTOR -----------------
base_model = models.mobilenet_v2(weights=None)
base_model.classifier = nn.Identity()
base_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
base_model.to(device)
base_model.eval()
print("✅ CNN loaded successfully.")

# ---------------- LOAD RANDOM FOREST -----------------
rf_model = joblib.load(RF_PATH)
print("✅ Random Forest loaded successfully.")

# ---------------- LOAD NIR DATA -----------------
if not os.path.exists(NIR_FILE):
    raise FileNotFoundError(f"NIR data file not found: {NIR_FILE}")

nir_df = pd.read_csv(NIR_FILE)

# Ensure NIR data is numeric and exclude non-numeric columns
nir_features_array = nir_df.select_dtypes(include=np.number).values
print(f"✅ NIR features loaded, shape: {nir_features_array.shape}")

# ---------------- IMAGE TRANSFORM -----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- CONNECT TO ARDUINO -----------------
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("✅ Arduino connected.")
except:
    arduino = None
    print("⚠️ Arduino not connected!")

# ---------------- FUNCTION TO CLASSIFY FRAME -----------------
def classify_frame(frame, nir_features):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_features = base_model(img_tensor).cpu().numpy()
        nir_features = np.array(nir_features, dtype=float).reshape(1, -1)
        features = np.concatenate([cnn_features, nir_features], axis=1)

        # Verify feature length matches RF training
        if features.shape[1] != rf_model.n_features_in_:
            raise ValueError(f"Feature size mismatch! RF expects {rf_model.n_features_in_}, got {features.shape[1]}")

        pred_label = rf_model.predict(features)[0]

    class_names = ['BadGrapes', 'GoodGrapes']
    return class_names[pred_label]

# ---------------- START CAMERA -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("📷 Press 'SPACE' or 'S' to capture image and classify. Press 'Q' to quit.")

nir_index = 0  # to select corresponding NIR row for each frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    h, w, _ = frame.shape
    x1, y1 = w//2 - 100, h//2 - 100
    x2, y2 = w//2 + 100, h//2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("Grape Classification", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') or key == ord('s'):
        roi = frame[y1:y2, x1:x2]

        if nir_index >= len(nir_features_array):
            print("⚠️ No more NIR data available, using last row")
            nir_features = nir_features_array[-1]
        else:
            nir_features = nir_features_array[nir_index]
            nir_index += 1

        result = classify_frame(roi, nir_features)
        print(f"Prediction: {result}")

        # Save result to CSV
        file_exists = os.path.isfile(RESULTS_FILE)
        with open(RESULTS_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Prediction'])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result])

        # Send to Arduino
        if arduino:
            arduino.write(b'G' if result == "GoodGrapes" else b'B')

        # Display prediction on frame temporarily
        cv2.putText(frame, result, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if result=="GoodGrapes" else (0,0,255), 2)
        cv2.imshow("Grape Classification", frame)
        cv2.waitKey(1000)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
