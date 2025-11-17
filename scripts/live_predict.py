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

# ---------------- PARAMETERS -----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MajorProject/
CNN_PATH = os.path.join(BASE_DIR, "models", "cnn_extractor.pth")
RF_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
IMG_SIZE = (224, 224)
RESULTS_FILE = os.path.join(BASE_DIR, "grape_results.csv")
SERIAL_PORT = 'COM5'
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
def classify_frame(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = base_model(img_tensor).cpu().numpy()
        pred_label = rf_model.predict(features)[0]
    
    class_names = ['BadGrapes', 'GoodGrapes']
    result = class_names[pred_label]
    return result

# ---------------- START CAMERA -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("📷 Press 'SPACE' or 'S' to capture image and classify. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Draw a rectangle in the center (ROI for grape)
    h, w, _ = frame.shape
    x1, y1 = w//2 - 100, h//2 - 100
    x2, y2 = w//2 + 100, h//2 + 100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("Grape Classification", frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture & classify when SPACE or 'S' is pressed
    if key == ord(' ') or key == ord('s'):
        roi = frame[y1:y2, x1:x2]
        result = classify_frame(roi)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result=="GoodGrapes" else (0,0,255), 2)
        cv2.imshow("Grape Classification", frame)
        cv2.waitKey(1000)  # Show prediction for 1 second

    # Quit when 'Q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
