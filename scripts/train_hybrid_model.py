# train_hybrid_model.py
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import os

# -------------------- SETTINGS --------------------
data_dir = "data/images/train"
batch_size = 32
feature_path = "models"
os.makedirs(feature_path, exist_ok=True)

cnn_path = os.path.join(feature_path, "cnn_extractor.pth")
rf_path = os.path.join(feature_path, "rf_model.pkl")

# -------------------- TRANSFORMS --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- DATASET --------------------
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Loaded {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

# -------------------- CNN FEATURE EXTRACTOR --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
base_model.classifier = nn.Identity()  # remove final FC layer
base_model = base_model.to(device)
base_model.eval()

# -------------------- EXTRACT FEATURES --------------------
features = []
labels = []

with torch.no_grad():
    for imgs, lbls in tqdm(dataloader, desc="Extracting CNN features"):
        imgs = imgs.to(device)
        output = base_model(imgs)
        features.append(output.cpu().numpy())
        labels.append(lbls.numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)

print(f"Feature shape: {features.shape}, Labels shape: {labels.shape}")

# -------------------- TRAIN RANDOM FOREST --------------------
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(features, labels)
print("✅ Random Forest training complete.")

# -------------------- SAVE MODELS --------------------
torch.save(base_model.state_dict(), cnn_path)
joblib.dump(rf, rf_path)

print(f"✅ CNN feature extractor saved to: {cnn_path}")
print(f"✅ Random Forest model saved to: {rf_path}")
print("Hybrid training complete.")
