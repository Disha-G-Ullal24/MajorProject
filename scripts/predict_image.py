import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import serial
import time

# ---------------- PARAMETERS ----------------
IMG_SIZE = (128, 128)
MODEL_PATH = r"D:\Major_project\Project_code\MajorProject\models\grape_mobilenetv2.pth"
IMAGE_PATH = "captured/grape_0.jpg"
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
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
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Error: Image not found at", IMAGE_PATH)
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
img_tensor = transform(img_pil).unsqueeze(0).to(device)

# ---------------- PREDICT ----------------
with torch.no_grad():
    output = torch.sigmoid(model(img_tensor))
prediction = output.item()
result = "GOOD" if prediction > 0.5 else "BAD"

print(f"Result: {result} (Confidence: {prediction:.2f})")

# ---------------- SEND TO ARDUINO ----------------
if arduino:
    arduino.write(b'G' if result == "GOOD" else b'B')
    print("✅ Sent to Arduino:", "G" if result == "GOOD" else "B")
