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

# ---------------- MERGE ON IMAGE NAME ----------------
merged_df = pd.merge(pred_df, nir_df, on="ImageName", how="left")

# ---------------- DECISION LOGIC ----------------
# Example logic: If predicted GOOD and Brix >= 15 and pH <= 3.8 -> GOOD, else BAD
def final_decision(row):
    if row['Prediction'] == "GoodGrapes" and row['Brix'] >= 15 and row['pH'] <= 3.8:
        return "GOOD"
    else:
        return "BAD"

merged_df['FinalPrediction'] = merged_df.apply(final_decision, axis=1)

# ---------------- SEND COMMANDS TO ARDUINO ----------------
for idx, row in merged_df.iterrows():
    command = b'G' if row['FinalPrediction'] == "GOOD" else b'B'
    if arduino:
        arduino.write(command)
        print(f"{row['ImageName']} -> {row['FinalPrediction']} sent to Arduino: {command.decode()}")
        time.sleep(0.5)  # small delay for servo to move
    else:
        print(f"{row['ImageName']} -> {row['FinalPrediction']} (Arduino not connected)")

# ---------------- SAVE FINAL MERGED RESULTS ----------------
merged_df.to_csv(MERGED_FILE, index=False)
print(f"✅ Final merged results saved to {MERGED_FILE}")
