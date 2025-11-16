import cv2
import os

# Create folder if it doesn't exist
if not os.path.exists("captured"):
    os.makedirs("captured")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("📷 Press SPACE to capture image, ESC to exit.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Capture Grape Image", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        filename = f"captured/grape_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
        count += 1

cap.release()
cv2.destroyAllWindows()
