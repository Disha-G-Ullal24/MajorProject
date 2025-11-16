import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# ---------------- PARAMETERS ----------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "../models/grape_mobilenetv2.pth")

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------- DATASETS ----------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)  # ✅ updated

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # Binary classification
model = model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- TRAIN ----------------
print(f"Using device: {DEVICE}")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)  # BCEWithLogitsLoss expects (N,1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).long().squeeze()
            correct += (preds == labels).sum().item()

    val_acc = correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
