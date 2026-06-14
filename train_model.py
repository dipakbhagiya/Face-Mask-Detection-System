"""
train_model.py - Train face mask detector
Uses OpenCV to load images (more robust than Keras on Windows)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import urllib.request, zipfile, pathlib, shutil
import cv2

print("=" * 55)
print("  Face Mask Detector — Model Trainer")
print("=" * 55)

# ── Download dataset ──────────────────────────────────────────
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    print("\nDownloading dataset...")
    url = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/refs/heads/master.zip"
    urllib.request.urlretrieve(url, "dataset.zip")
    print("Extracting...")
    with zipfile.ZipFile("dataset.zip", "r") as z:
        z.extractall(".")
    shutil.copytree("Face-Mask-Detection-master/dataset", DATASET_DIR)
    shutil.rmtree("Face-Mask-Detection-master")
    os.remove("dataset.zip")
    print("Dataset ready!")

# ── Find image files ──────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder):
    p = pathlib.Path(folder)
    return [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS]

with_mask_files    = get_images(os.path.join(DATASET_DIR, "with_mask"))
without_mask_files = get_images(os.path.join(DATASET_DIR, "without_mask"))

print(f"\nwith_mask    : {len(with_mask_files)} images")
print(f"without_mask : {len(without_mask_files)} images")

# ── Load with OpenCV ──────────────────────────────────────────
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
data, labels = [], []

def load_image_cv2(fpath):
    """Load image using OpenCV and convert to RGB float array."""
    img = cv2.imdecode(
        np.fromfile(str(fpath), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    arr = img.astype("float32")
    arr = preprocess_input(arr)
    return arr

print("\nLoading with_mask images...")
ok, fail = 0, 0
for fpath in with_mask_files:
    arr = load_image_cv2(fpath)
    if arr is not None:
        data.append(arr)
        labels.append(0)   # 0 = with_mask
        ok += 1
    else:
        fail += 1
print(f"   Loaded: {ok}   Failed: {fail}")

print("Loading without_mask images...")
ok, fail = 0, 0
for fpath in without_mask_files:
    arr = load_image_cv2(fpath)
    if arr is not None:
        data.append(arr)
        labels.append(1)   # 1 = without_mask
        ok += 1
    else:
        fail += 1
print(f"   Loaded: {ok}   Failed: {fail}")

if len(data) == 0:
    print("\nNo images loaded. Try deleting the dataset folder and rerun.")
    exit(1)

data   = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"\nTotal: {len(data)} images")
print(f"   With mask    (0): {np.sum(labels==0)}")
print(f"   Without mask (1): {np.sum(labels==1)}")

with open("label_map.txt", "w") as f:
    f.write("0 = with_mask\n1 = without_mask\n")

# ── Split ─────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

labels_cat = to_categorical(labels, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_cat, test_size=0.2, stratify=labels_cat, random_state=42
)
print(f"   Train: {len(X_train)}  Test: {len(X_test)}")

# ── Augmentation ──────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True, fill_mode="nearest"
)

# ── Build model ───────────────────────────────────────────────
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("\nBuilding MobileNetV2...")
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
base.trainable = False

head = AveragePooling2D(pool_size=(7,7))(base.output)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=base.input, outputs=head)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])

# ── Train ─────────────────────────────────────────────────────
BS, EPOCHS = 32, 20
print(f"Training {EPOCHS} epochs...\n")
model.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train)//BS,
    validation_data=(X_test, y_test),
    epochs=EPOCHS, verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy: {acc*100:.2f}%")

# ── Sanity check ──────────────────────────────────────────────
print("\nSanity check on 10 samples:")
preds = model.predict(X_test[:10], verbose=0)
true  = np.argmax(y_test[:10], axis=1)
pred  = np.argmax(preds, axis=1)
names = {0:"with_mask", 1:"without_mask"}
for i in range(10):
    ok = "OK" if true[i]==pred[i] else "FAIL"
    print(f"  {ok} True: {names[true[i]]:15s}  Pred: {names[pred[i]]}")

model.save("mask_detector.keras")
print("\nSaved: mask_detector.keras")
print("Done! Run: python detect.py")