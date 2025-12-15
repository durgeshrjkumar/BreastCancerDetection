import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_DIR = "/kaggle/input/breast-histopathology-images"
IMG_SIZE = 128  # upscale 50x50 to 128x128 for EfficientNet
BATCH_SIZE = 64
VAL_SPLIT = 0.15  # 15% for validation
all_image_paths = []
all_labels = []

# Walk through the directory and collect image paths
for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(root, fname)
            all_image_paths.append(fpath)
            # label is folder name: 0 or 1
            label = 1 if os.path.basename(root) == '1' else 0
            all_labels.append(label)

print("Total images found:", len(all_image_paths))
# Convert to numpy
all_image_paths = np.array(all_image_paths)
all_labels = np.array(all_labels)

# Shuffle
indices = np.arange(len(all_image_paths))
np.random.shuffle(indices)
all_image_paths = all_image_paths[indices]
all_labels = all_labels[indices]

# Train/validation split
val_size = int(VAL_SPLIT * len(all_image_paths))
x_val = all_image_paths[:val_size]
y_val = all_labels[:val_size]

x_train = all_image_paths[val_size:]
y_train = all_labels[val_size:]

print("Train samples:", len(x_train))
print("Val samples  :", len(x_val))


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0–1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, tf.cast(label, tf.float32)


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=5000, seed=SEED)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_val, y_val))
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_image_eff(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, tf.cast(label, tf.float32)


train_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=5000, seed=SEED)
    .map(load_image_eff, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_val, y_val))
    .map(load_image_eff, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


def build_model(img_size=IMG_SIZE):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3)
    )

    # Freeze base at first
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


model = build_model()
model.summary()
METRICS = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=METRICS,
)
EPOCHS_WARMUP = 5

history_warmup = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_WARMUP
)
# Unfreeze base model
model.get_layer(index=1).trainable = True  # index 0 is Input, 1 is base_model usually

# Optional: only fine-tune last N layers
fine_tune_at = 200  # tune last 200 layers
for layer in model.get_layer(index=1).layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=METRICS,
)

EPOCHS_FINE = 15

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE
)
val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Validation AUC     : {val_auc:.4f}")
print(f"Validation precision: {val_prec:.4f}")
print(f"Validation recall   : {val_rec:.4f}")
model.save("/kaggle/working/breast_idc_efficientnet.h5")
import matplotlib.pyplot as plt


def plot_history(history, history2=None, metric="accuracy"):
    plt.figure(figsize=(8, 5))

    plt.plot(history.history[metric], label=f"train_{metric}_warmup")
    plt.plot(history.history["val_" + metric], label=f"val_{metric}_warmup")

    if history2:
        offset = len(history.history[metric])
        plt.plot(
            np.arange(offset, offset + len(history2.history[metric])),
            history2.history[metric],
            label=f"train_{metric}_fine"
        )
        plt.plot(
            np.arange(offset, offset + len(history2.history["val_" + metric])),
            history2.history["val_" + metric],
            label=f"val_{metric}_fine"
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_history(history_warmup, history_fine, metric="accuracy")
plot_history(history_warmup, history_fine, metric="auc")
