import numpy as np
import glob
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import random

# ==============================
# CONFIG
# ==============================
BATCH_FILES = sorted(glob.glob("asl_dataset_batch_*.npz"))
LABELS_FILE = "asl_labels.json"
EPOCHS_STAGE1 = 3
EPOCHS_STAGE2 = 3
BATCH_SIZE = 32
STEPS_PER_EPOCH = 200

# ==============================
# Load Labels
# ==============================
with open(LABELS_FILE, 'r') as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
NUM_CLASSES = len(label_map)
print(f"Found {NUM_CLASSES} classes")

# ==============================
# Data Generator (Streaming)
# ==============================
def data_generator(batch_files, num_classes, batch_size=32):
    while True:
        for batch_file in batch_files:
            data = np.load(batch_file)
            X = data["X"]
            y = tf.keras.utils.to_categorical(data["y"], num_classes)

            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X, y = X[idx], y[idx]

            for i in range(0, len(X), batch_size):
                yield X[i:i+batch_size], y[i:i+batch_size]

# Split batch files 80/20 for train/val
split_idx = int(0.8 * len(BATCH_FILES))
train_files = BATCH_FILES[:split_idx]
val_files = BATCH_FILES[split_idx:]

VAL_STEPS_FULL = sum([len(np.load(f)["X"]) for f in val_files]) // BATCH_SIZE
print(f"Validation steps (full pass): {VAL_STEPS_FULL}")

# ==============================
# Build Model Function
# ==============================
def build_model(num_classes, trainable_base=False):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = trainable_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# ==============================
# Stage 1: Train with Frozen Base
# ==============================
model = build_model(NUM_CLASSES, trainable_base=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

print("Training stage 1 (frozen base)...")
model.fit(
    data_generator(train_files, NUM_CLASSES, BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=data_generator(val_files, NUM_CLASSES, BATCH_SIZE),
    validation_steps=max(1, VAL_STEPS_FULL // 5),
    epochs=EPOCHS_STAGE1
)

# ==============================
# Stage 2: Fine-tune with Partial Unfreezing
# ==============================
print("Training stage 2 (fine-tuning)...")
for layer in model.layers[:-20]:
    layer.trainable = False
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    data_generator(train_files, NUM_CLASSES, BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=data_generator(val_files, NUM_CLASSES, BATCH_SIZE),
    validation_steps=max(1, VAL_STEPS_FULL // 5),
    epochs=EPOCHS_STAGE2
)

# ==============================
# Final Evaluation (Subset of Validation Batches)
# ==============================
print("Evaluating on subset of validation batches...")
val_gen = data_generator(val_files, NUM_CLASSES, BATCH_SIZE)

# pick 5 random validation batches to test
sample_batches = random.sample(val_files, min(5, len(val_files)))
all_preds, all_labels = [], []
for batch_file in sample_batches:
    data = np.load(batch_file)
    X = data["X"]
    y = data["y"]

    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    all_preds.append(np.argmax(preds, axis=1))
    all_labels.append(y)

y_val_pred = np.concatenate(all_preds)
y_val_true = np.concatenate(all_labels)

print("Classification Report (subset):")
print(classification_report(y_val_true, y_val_pred, target_names=[label_map[i] for i in range(NUM_CLASSES)]))

# ==============================
# Export to SavedModel and TFLite
# ==============================
saved_model_dir = "asl_mobilenetv2_finetuned_streamval"
model.export(saved_model_dir)
print(f"Model exported to {saved_model_dir}")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open("asl_mobilenetv2_finetuned_streamval.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved to asl_mobilenetv2_finetuned_streamval.tflite")
