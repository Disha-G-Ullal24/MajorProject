import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import serial
import time
import joblib
import json
import os

# ---------------- LOAD CONFIG ----------------
if not os.path.exists("config.json"):
    raise FileNotFoundError("config.json not found!")

with open("config.json") as f:
    cfg = json.load(f)

# ---------------- SERIAL CONNECTION ----------------
try:
    ser = serial.Serial(cfg["serial_port"], cfg["baud_rate"], timeout=1)
    time.sleep(2)  # wait for Arduino
    print(f"✅ Serial connected on {cfg['serial_port']}")
except:
    ser = None
    print("⚠️ Serial connection failed! Arduino commands will not be sent.")

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD CV MODEL ----------------
CV_MODEL_PATH = "models/grape_mobilenetv2.pth"
cv_model = models.mobilenet_v2(weights=None)
cv_model.classifier[1] = nn.Linear(cv_model.last_channel, 1)  # Binary output
cv_model.load_state_dict(torch.load(CV_MODEL_PATH, map_location=device))
cv_model.to(device)
cv_model.eval()

# Feature extractor (penultimate layer)
feature_extractor = nn.Sequential(*list(cv_model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# ---------------- LOAD HYBRID MODEL ----------------
HYBRID_MODEL_PATH = "models/hybrid_model.pkl"
hybrid_model = joblib.load(HYBRID_MODEL_PATH)

# ---------------- IMAGE TRANSFORMS ----------------
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("📷 Press 'q' to quit realtime inference.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # ---------------- CV MODEL PREPROCESS ----------------
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # CV Prediction
    with torch.no_grad():
        output = torch.sigmoid(cv_model(img_tensor))
    pred = output.item()  # scalar
    label = 1 if pred > 0.5 else 0  # GOOD=1, BAD=0
    conf = pred

    # ---------------- DECISION ----------------
    if conf >= cfg["skip_nir_confidence"]:
        # CV-only decision
        if ser:
            cmd = b'A\n' if label == 1 else b'B\n'
            ser.write(cmd)
    else:
        # Hybrid decision (CV features + NIR)
        # Replace simulated NIR with actual sensor readings
        brix, poly = 17, 155
        with torch.no_grad():
            feat = feature_extractor(img_tensor)
            feat = torch.flatten(feat, 1).cpu().numpy()  # flatten to 1D

        X = np.hstack((feat, [[brix, poly]]))
        hybrid_pred = hybrid_model.predict(X)[0]

        if hybrid_pred == 1:  # GOOD
            if brix >= cfg["nir_thresholds"]["brix_min"] and poly >= cfg["nir_thresholds"]["polyphenol_min"]:
                if ser:
                    ser.write(b'A1\n')
            else:
                if ser:
                    ser.write(b'B1\n')
        else:
            if ser:
                ser.write(b'B1\n')

    # ---------------- DISPLAY ----------------
    label_text = "GOOD" if (label == 1 or hybrid_pred == 1) else "BAD"
    color = (0, 255, 0) if label_text == "GOOD" else (0, 0, 255)
    cv2.putText(frame, f"Prediction: {label_text}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Realtime Inference", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
