import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ---------------- PARAMETERS -----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MajorProject/
MODEL_PATH = os.path.join(BASE_DIR, "models", "grape_mobilenetv2.pth")
IMG_SIZE = (224, 224)

# ---------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD MODEL -----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # Binary output
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# ---------------- IMAGE TRANSFORM -----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------- FUNCTION TO CLASSIFY FRAME -----------------
def classify_frame(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).item()  # scalar between 0 and 1
    label = "GOOD" if pred > 0.5 else "BAD"
    return label, pred

# ---------------- START CAMERA -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("📷 Press 'q' to quit real-time classification.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    label, confidence = classify_frame(frame)

    # Display prediction on frame
    color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Grape Classification", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
