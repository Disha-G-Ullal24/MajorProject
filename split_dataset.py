import os
import shutil
import random

# Paths
source_dir = "dataset"  # your original dataset folder
target_dir = "data/images"

# Train/val split ratio
split_ratio = 0.8  

# Classes
classes = ["good", "bad"]

for cls in classes:
    source_folder = os.path.join(source_dir, cls)
    images = os.listdir(source_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create target folders
    train_folder = os.path.join(target_dir, "train", cls)
    val_folder = os.path.join(target_dir, "val", cls)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Move files
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

    for img in val_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print("✅ Dataset split completed!")
