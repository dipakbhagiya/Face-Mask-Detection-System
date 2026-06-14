"""
detect.py - Real-time Face Mask Detector
Label map: 0 = with_mask, 1 = without_mask
"""

import os
import cv2
import time
import pathlib
import urllib.request
import numpy as np

from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# Linux / Qt Fix
# ==============================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==============================
# SETTINGS
# ==============================
CAMERA_INDEX = 1
CONFIDENCE_MIN = 0.5
IMG_SIZE = 224

MODEL_PATH = "mask_detector.keras"

FACE_PROTO = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

# ==============================
# COLORS
# ==============================
WHITE = (255, 255, 255)
GREEN = (0, 200, 80)
RED = (0, 60, 220)
YELLOW = (0, 200, 255)
GRAY = (120, 120, 120)

# ==============================
# LABEL MAP
# ==============================
LABEL_MAP = {
    0: ("Mask", GREEN),
    1: ("No Mask", RED),
}


# ==============================
# DOWNLOAD FACE DETECTOR
# ==============================
def download_face_detector():

    pathlib.Path("face_detector").mkdir(exist_ok=True)

    files = {
        FACE_PROTO:
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",

        FACE_MODEL:
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }

    for path, url in files.items():

        if not os.path.exists(path):

            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)

    print("Face detector ready!")


# ==============================
# CAMERA OPEN
# ==============================
def open_camera():

    print("Searching for webcam...")

    for idx in [1, 2, 0]:

        print(f"Trying camera {idx}...")

        cap = cv2.VideoCapture(idx)

        if cap.isOpened():

            ret, frame = cap.read()

            if ret and frame is not None:

                print(f"Webcam connected on camera index {idx}")

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                time.sleep(2)

                return cap

            cap.release()

    return None


# ==============================
# UI FUNCTIONS
# ==============================
def draw_rounded_rect(img, x1, y1, x2, y2, color, r=10, t=2):

    cv2.line(img, (x1+r, y1), (x2-r, y1), color, t)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, t)

    cv2.line(img, (x1, y1+r), (x1, y2-r), color, t)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, t)

    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, t)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, t)

    cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, t)
    cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, t)


def draw_label(img, text, x, y, bg):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(text, font, 0.65, 2)

    cv2.rectangle(img, (x, y-th-12), (x+tw+12, y), bg, -1)

    cv2.putText(
        img,
        text,
        (x+6, y-6),
        font,
        0.65,
        WHITE,
        2
    )


def draw_hud(frame, faces, masks, fps):

    h = frame.shape[0]

    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (270, 115), (15, 15, 15), -1)

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "NOVA MASK DETECTOR",
                (10, 22), font, 0.5, YELLOW, 1)

    cv2.putText(frame, f"FPS        : {fps:.1f}",
                (10, 45), font, 0.5, WHITE, 1)

    cv2.putText(frame, f"Faces      : {faces}",
                (10, 65), font, 0.5, WHITE, 1)

    cv2.putText(frame, f"With Mask  : {masks}",
                (10, 85), font, 0.5, GREEN, 1)

    cv2.putText(frame, f"No Mask    : {faces-masks}",
                (10, 105), font, 0.5, RED, 1)

    cv2.putText(frame, "[Q] Quit  [S] Screenshot",
                (10, h-10), font, 0.4, GRAY, 1)


# ==============================
# MAIN
# ==============================
def main():

    print("=" * 50)
    print("Nova — Real-Time Face Mask Detector")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):

        print("Model not found!")
        print("Run: python train_model.py")

        return

    download_face_detector()

    print("\nLoading model...")

    model = load_model(MODEL_PATH)

    print("Model loaded!")

    print("\nLoading face detector...")

    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)

    cap = open_camera()

    if cap is None:

        print("\nNo webcam found!")
        return

    print("\nRunning...")
    print("Press Q to quit")
    print("Press S for screenshot\n")

    prev = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ==============================
        # FACE DETECTION
        # ==============================
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)

        detections = face_net.forward()

        faces = []
        locs = []

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < CONFIDENCE_MIN:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (x1, y1, x2, y2) = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

            face_rgb = img_to_array(face_rgb)

            face_rgb = preprocess_input(face_rgb)

            faces.append(face_rgb)

            locs.append((x1, y1, x2, y2))

        preds = []

        if len(faces) > 0:

            faces_np = np.array(faces, dtype="float32")

            preds = model.predict(faces_np, verbose=0)

        mask_count = 0

        for (box, pred) in zip(locs, preds):

            (x1, y1, x2, y2) = box

            idx = np.argmax(pred)

            conf = pred[idx] * 100

            label, color = LABEL_MAP[idx]

            if idx == 0:
                mask_count += 1

            text = f"{label} {conf:.0f}%"

            draw_rounded_rect(frame, x1, y1, x2, y2, color)

            draw_label(frame, text, x1, y1, color)

        now = time.time()

        fps = 1.0 / (now - prev + 1e-6)

        prev = now

        draw_hud(frame, len(locs), mask_count, fps)

        cv2.imshow("Nova - Face Mask Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Closing...")
            break

        elif key == ord("s"):

            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            cv2.imwrite(filename, frame)

            print(f"Saved: {filename}")

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()