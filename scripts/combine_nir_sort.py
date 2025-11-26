import os
import pandas as pd
import serial
import time

# ---------------- PARAMETERS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MajorProject/
PRED_FILE = os.path.join(BASE_DIR, "grape_results.csv")        # Hybrid model predictions
NIR_FILE = os.path.join(BASE_DIR, "data", "nir", "nir_data.csv")  # NIR features
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
MERGED_FILE = os.path.join(BASE_DIR, "grape_final_results.csv")   # Output file

# ---------------- CONNECT TO ARDUINO ----------------
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("✅ Arduino connected.")
except:
    arduino = None
    print("⚠️ Arduino not connected! Sorting will be simulated.")

# ---------------- READ CSV FILES ----------------
if not os.path.exists(PRED_FILE):
    print(f"❌ Prediction file not found: {PRED_FILE}")
    exit()
if not os.path.exists(NIR_FILE):
    print(f"❌ NIR data file not found: {NIR_FILE}")
    exit()

pred_df = pd.read_csv(PRED_FILE)
nir_df = pd.read_csv(NIR_FILE)

# ---------------- CLEAN AND RENAME COLUMNS ----------------
# Ensure first column in each CSV is ImageName and second column is Prediction
pred_df.columns = ["ImageName", "Prediction"] + list(pred_df.columns[2:])
nir_df.columns = ["ImageName", "Brix", "pH"] + list(nir_df.columns[3:])

# Convert ImageName to string
pred_df["ImageName"] = pred_df["ImageName"].astype(str)
nir_df["ImageName"] = nir_df["ImageName"].astype(str)

# ---------------- MERGE ----------------
merged_df = pd.merge(pred_df, nir_df, on="ImageName", how="left")

# Check column names after merge
print("Merged columns:", merged_df.columns.tolist())

# ---------------- DECISION LOGIC ----------------
def final_decision(row):
    try:
        if row['Prediction'] == "GoodGrapes" and row['Brix'] >= 15 and row['pH'] <= 3.8:
            return "GOOD"
        else:
            return "BAD"
    except KeyError:
        # fallback if Brix/pH not present
        return "GOOD" if row.get('Prediction', '') == "GoodGrapes" else "BAD"

merged_df['FinalPrediction'] = merged_df.apply(final_decision, axis=1)

# ---------------- SEND COMMANDS TO ARDUINO ----------------
for idx, row in merged_df.iterrows():
    command = b'G' if row['FinalPrediction'] == "GOOD" else b'B'
    if arduino:
        arduino.write(command)
        print(f"{row['ImageName']} -> {row['FinalPrediction']} sent to Arduino: {command.decode()}")
        time.sleep(0.5)
    else:
        print(f"{row['ImageName']} -> {row['FinalPrediction']} (Arduino not connected)")

# ---------------- SAVE RESULTS ----------------
merged_df.to_csv(MERGED_FILE, index=False)
print(f"✅ Final merged results saved to {MERGED_FILE}")
