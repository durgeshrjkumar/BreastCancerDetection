import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATASET_DIR = "IDC_regular_dataset"
IMG_SIZE = 50

X, y = [], []

for patient_id in os.listdir(DATASET_DIR):
    patient_path = os.path.join(DATASET_DIR, patient_id)

    if not os.path.isdir(patient_path):
        continue

    for label in ["0", "1"]:
        class_path = os.path.join(patient_path, label)

        if not os.path.exists(class_path):
            continue

        for img_name in os.listdir(class_path):
            # REDUCE DATASET SIZE (40%)
            # if np.random.rand() > 0.10:
            #     continue
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(int(label))

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

print("Loaded images:", X.shape)
print("Labels:", y.shape)
print("Cancer samples:", np.sum(y))
print("Non-cancer samples:", len(y) - np.sum(y))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(50,50,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# train and save the model for future usage.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save("breast_cancer_cnn.keras")


