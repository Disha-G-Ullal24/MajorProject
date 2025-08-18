import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

labels = pd.read_csv("data/labels.csv")
model = load_model("models/cv_model.h5")

features = []
for idx, row in labels.iterrows():
    img = image.load_img(row['image_path'], target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    feat = model.layers[-3].predict(img_array)  # penultimate layer features
    features.append(feat[0])

np.save("models/cv_features.npy", np.array(features))
