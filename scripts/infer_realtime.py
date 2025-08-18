import cv2
import numpy as np
import serial
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load config
with open("config.json") as f:
    cfg = json.load(f)

# Serial connection
ser = serial.Serial(cfg["serial_port"], cfg["baud_rate"], timeout=1)

# Load models
cv_model = load_model("models/cv_model.h5")
hybrid_model = joblib.load("models/hybrid_model.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess for CV model
    img = cv2.resize(frame, (224,224))
    img_array = np.expand_dims(img/255.0, axis=0)
    pred = cv_model.predict(img_array)[0]
    label = np.argmax(pred)
    conf = np.max(pred)

    # If confidence high, skip NIR
    if conf >= cfg["skip_nir_confidence"]:
        if label == 0:  # good
            ser.write(b'A\n')
        else:
            ser.write(b'B\n')
    else:
        # Simulate NIR values (replace with actual sensor reading)
        brix, poly = 17, 155
        feat = cv_model.layers[-3].predict(img_array)
        X = np.hstack((feat, [[brix, poly]]))
        hybrid_pred = hybrid_model.predict(X)[0]

        if hybrid_pred == 1:  # good
            if brix >= cfg["nir_thresholds"]["brix_min"] and poly >= cfg["nir_thresholds"]["polyphenol_min"]:
                ser.write(b'A1\n')
            else:
                ser.write(b'B1\n')
        else:
            ser.write(b'B1\n')

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
