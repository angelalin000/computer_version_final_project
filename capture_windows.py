import cv2
import os
import time

SHARED_DIR = r"C:\shared"
os.makedirs(SHARED_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)  # iVCam 常見是 1

if not cap.isOpened():
    print("Camera open failed")
    exit(1)

print("Press q to quit")

idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: failed to grab frame")
        time.sleep(0.1)
        continue

    path = os.path.join(SHARED_DIR, f"frame_{idx}.jpg")
    cv2.imwrite(path, frame)

    idx = 1 - idx  # 0 ↔ 1 切換

    cv2.imshow("Windows Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
