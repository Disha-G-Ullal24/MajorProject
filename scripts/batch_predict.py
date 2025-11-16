import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- PATHS ----------------
MODEL_PATH = "../models/grape_mobilenetv2.pth"
TEST_DIR = "data/test_images"   # Folder with multiple images
IMG_SIZE = (224, 224)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------- PREDICT ALL IMAGES ----------------
for img_name in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_name)
    if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).item()

    label = "GOOD" if pred > 0.5 else "BAD"
    print(f"{img_name} → {label} ({pred:.2f})")
