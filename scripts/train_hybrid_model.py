# train_hybrid_model_with_nir.py
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import os

# -------------------- SETTINGS --------------------
data_dir = r"C:\Users\Disha G Ullal\Desktop\Disha Engineering\Projects\major project\MajorProject\data\images"
nir_csv = r"C:\Users\Disha G Ullal\Desktop\Disha Engineering\Projects\major project\MajorProject\data\nir\nir_data.csv"
results_csv = r"C:\Users\Disha G Ullal\Desktop\Disha Engineering\Projects\major project\MajorProject\grape_results.csv"

batch_size = 32
feature_path = "models"
os.makedirs(feature_path, exist_ok=True)

cnn_path = os.path.join(feature_path, "cnn_extractor.pth")
rf_path = os.path.join(feature_path, "rf_model.pkl")
scaler_path = os.path.join(feature_path, "nir_scaler.pkl")

# -------------------- IMAGE TRANSFORMS --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- IMAGE DATASET --------------------
if not os.path.exists(data_dir):
    raise ValueError(f"Dataset directory does not exist: {data_dir}")

dataset = datasets.ImageFolder(data_dir, transform=transform)
if len(dataset.classes) == 0:
    raise ValueError(f"No class folders found in {data_dir}. Make sure you have subfolders for each class.")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print(f"✅ Loaded {len(dataset)} images across {len(dataset.classes)} classes: {dataset.classes}")

# -------------------- CNN FEATURE EXTRACTOR --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
base_model.classifier = nn.Identity()
base_model = base_model.to(device)
base_model.eval()

# -------------------- EXTRACT IMAGE FEATURES --------------------
features_img = []
labels_img = []

with torch.no_grad():
    for imgs, lbls in tqdm(dataloader, desc="Extracting CNN features"):
        imgs = imgs.to(device)
        output = base_model(imgs)
        features_img.append(output.cpu().numpy())
        labels_img.append(lbls.numpy())

features_img = np.concatenate(features_img)
labels_img = np.concatenate(labels_img)
print(f"Image features shape: {features_img.shape}, Labels shape: {labels_img.shape}")

# -------------------- LOAD AND SCALE NIR DATA --------------------
nir_df = pd.read_csv(nir_csv).dropna()
if 'Label' not in nir_df.columns:
    raise ValueError("NIR CSV must contain 'Label' column")

X_nir = nir_df.drop(columns=['Label'])
y_nir = nir_df['Label']

scaler = StandardScaler()
X_nir_scaled = scaler.fit_transform(X_nir)

# Save scaler
joblib.dump(scaler, scaler_path)
print(f"✅ NIR scaler saved to: {scaler_path}")

# -------------------- COMBINE IMAGE + NIR FEATURES --------------------
# Note: ensure same order of samples; here we assume alignment by index
# If you have different samples, you'll need proper matching logic
min_samples = min(features_img.shape[0], X_nir_scaled.shape[0])
X_combined = np.hstack((features_img[:min_samples], X_nir_scaled[:min_samples]))
y_combined = labels_img[:min_samples]  # using image labels as target

print(f"Combined feature shape: {X_combined.shape}, Combined labels shape: {y_combined.shape}")

# -------------------- TRAIN RANDOM FOREST --------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_combined, y_combined)
print("✅ Random Forest training complete")
print("Training accuracy:", rf.score(X_combined, y_combined))

# -------------------- SAVE MODELS --------------------
torch.save(base_model.state_dict(), cnn_path)
joblib.dump(rf, rf_path)
print(f"✅ CNN feature extractor saved to: {cnn_path}")
print(f"✅ Random Forest model saved to: {rf_path}")
print("🎉 Hybrid training with images + NIR CSV complete!")
