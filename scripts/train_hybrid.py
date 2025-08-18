import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

labels = pd.read_csv("data/labels.csv")
cv_features = np.load("models/cv_features.npy")

# Combine CV features + NIR spectroscopy values
nir_data = labels[['brix','polyphenol']].values
X = np.hstack((cv_features, nir_data))
y = (labels['label'] == 'good').astype(int)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/hybrid_model.pkl")
print("Hybrid model trained and saved.")
