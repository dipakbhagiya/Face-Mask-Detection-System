# 😷 Real-Time Face Mask Detector

> Real-time face mask detection using webcam, OpenCV, and MobileNetV2 deep learning — built from scratch.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## 📌 About

A real-time face mask detection system built as part of the **Uneeq Interns** internship program. The system uses your webcam to detect faces and classify each one as **with mask** or **without mask** — instantly, with a confidence percentage displayed on screen.

---

## 🎬 Demo

| With Mask ✅ | Without Mask ❌ |
|---|---|
| Green bounding box | Red bounding box |
| `Mask 94%` label | `No Mask 87%` label |

🎥 Watch the demo: [YouTube Link]

---

## 🔄 How It Works

```
Webcam Frame
    │
    ▼
OpenCV DNN (ResNet SSD)
    │  Detects all faces → bounding box coordinates
    ▼
MobileNetV2 Classifier
    │  For each face: with_mask or without_mask + confidence %
    ▼
Draw Results on Frame
    │  Green box = Mask ✅   Red box = No Mask ❌
    ▼
Display in Real Time
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Webcam capture | OpenCV | Read live video frames |
| Face detection | OpenCV DNN + ResNet SSD | Locate faces in each frame |
| Mask classification | TensorFlow / Keras MobileNetV2 | Predict mask / no mask |
| Training | Transfer learning + ImageDataGenerator | Fine-tune on mask dataset |
| Data split | scikit-learn | Train/test split |
| Environment | Python venv | Isolated dependencies |

---

## 📁 Project Structure

```
face_mask_detector/
├── train_model.py          ← Download dataset + train model (run once)
├── detect.py               ← Real-time webcam detection
├── test_camera.py          ← Find your correct camera index
├── requirements.txt        ← Python dependencies
├── mask_detector.keras     ← Saved model (created after training)
├── label_map.txt           ← Label mapping (created after training)
└── face_detector/          ← OpenCV face detector files (auto-downloaded)
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Webcam
- ~500MB disk space (TensorFlow + dataset)

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detector.git
cd face-mask-detector
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows CMD
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model (run ONCE)
```bash
python train_model.py
```

This will automatically:
- Download the face mask dataset (~1,376 images)
- Load images using OpenCV
- Train MobileNetV2 for 20 epochs
- Save `mask_detector.keras`
- Print a sanity check to confirm labels are correct

Training takes about **5–10 minutes** on CPU.

### Step 5 — Run real-time detection
```bash
python detect.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Base model | MobileNetV2 (ImageNet weights) |
| Input size | 224 × 224 × 3 |
| Output | 2 classes (with_mask / without_mask) |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 20 |
| Batch size | 32 |
| Augmentation | Rotation, zoom, flip, shift |
| Test accuracy | ~95% |

### Label Map
```
Index 0 = with_mask    → Green box ✅
Index 1 = without_mask → Red box ❌
```

---

## 🐛 Troubleshooting

**Wrong camera / black screen**
```bash
python test_camera.py
```
Then set `CAMERA_INDEX` at the top of `detect.py` to the working index.

**Model predicts wrong label (mask = no mask)**
Delete `mask_detector.keras` and retrain:
```bash
del mask_detector.keras
python train_model.py
```

**Images fail to load during training**
The project uses OpenCV (`cv2.imdecode`) instead of PIL for image loading, which is more robust on Windows.

---

## 📄 Requirements

```
opencv-python
numpy
tensorflow
scikit-learn
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- Built as part of the **Uneeq Interns** internship program
- Dataset: [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- Face detector: OpenCV ResNet SSD
- Classification: MobileNetV2 (Google)
