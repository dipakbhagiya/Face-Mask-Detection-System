"""
test_camera.py
Run this to find your working camera index.
"""
import cv2
import time

print("Testing cameras 0 to 4...\n")

for i in range(5):
    print(f"Testing index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)  # MSMF = Windows Media Foundation
    if not cap.isOpened():
        print(f"  ❌ Index {i}: Cannot open\n")
        continue

    time.sleep(1)  # wait for camera to warm up
    success_count = 0
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            success_count += 1

    cap.release()

    if success_count > 0:
        print(f"  ✅ Index {i}: WORKS! ({success_count}/10 frames OK)")
        print(f"     >>> USE THIS INDEX: {i} <<<\n")
    else:
        print(f"  ❌ Index {i}: Opens but no frames\n")

print("Done. Use the working index in detect.py")