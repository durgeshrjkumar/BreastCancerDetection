import os, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import itertools

# -------- CONFIG --------
MODEL_PATH = "/kaggle/input/kerasmodel/keras/default/1/breast_idc_efficientnet.keras"   # change if needed
DATA_DIR   = "/kaggle/input/breast-histopathology-images"  # or local path
IMG_SIZE   = 128
BATCH_SIZE = 32
MAX_SAMPLES = 3000   # keeps it FAST
# ------------------------

print("Loading model...")
model = load_model(MODEL_PATH)
model.summary()

# -------- DATASET (LIGHTWEIGHT) --------
paths, labels = [], []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".png"):
            paths.append(os.path.join(root, f))
            labels.append(1 if os.path.basename(root) == "1" else 0)

paths, labels = paths[:MAX_SAMPLES], labels[:MAX_SAMPLES]
print(f"\nUsing {len(paths)} images for evaluation")

def load_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img, label

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((paths, labels))
    .map(load_img)
    .batch(BATCH_SIZE)
)

# -------- PREDICTIONS --------
y_true, y_pred, y_prob = [], [], []

for imgs, lbls in val_ds:
    preds = model.predict(imgs, verbose=0).flatten()
    y_prob.extend(preds)
    y_pred.extend((preds > 0.5).astype(int))
    y_true.extend(lbls.numpy())

# -------- METRICS --------
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)

print("\nCONFUSION MATRIX")
print(cm)
print("\nCLASSIFICATION REPORT")
print(report)

# -------- VISUALS (ONE PAGE) --------
plt.figure(figsize=(14,5))

# Confusion Matrix
plt.subplot(1,2,1)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Non-Cancer","Cancer"])
plt.yticks([0,1], ["Non-Cancer","Cancer"])

for i, j in itertools.product(range(2), range(2)):
    plt.text(j, i, cm[i,j], ha="center", va="center", color="black")

# Prediction Confidence Histogram
plt.subplot(1,2,2)
plt.hist(y_prob, bins=30)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

