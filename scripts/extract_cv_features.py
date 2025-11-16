import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

# ---------------- PARAMETERS ----------------
MODEL_PATH = "models/grape_mobilenetv2.pth"
LABELS_CSV = "data/labels.csv"
FEATURES_SAVE_PATH = "models/cv_features.npy"
IMG_SIZE = (224, 224)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD TRAINED MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # binary
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- CORRECT FEATURE EXTRACTOR ----------------
# MobilenetV2 features come from model.features (last conv output)
feature_extractor = model.features.to(device)
feature_extractor.eval()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------- LOAD LABEL CSV ----------------
labels = pd.read_csv(LABELS_CSV)

features = []

for idx, row in labels.iterrows():

    img_path = row['image_path']

    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- extract features from MobileNet ---
    with torch.no_grad():
        feat = feature_extractor(img_tensor)  # shape = [1, 1280, 7, 7]
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)  # final shape = [1, 1280]

    features.append(feat.cpu().numpy()[0])

# Save features
features_array = np.array(features)
np.save(FEATURES_SAVE_PATH, features_array)

print(f"✅ Features extracted and saved: {FEATURES_SAVE_PATH}")
