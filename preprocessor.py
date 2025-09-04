import cv2
import os
import numpy as np
import mediapipe as mp
import json

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "/home/mikege2/Documents/Senior Project/ASL - Kaggle/asl_alphabet_train/asl_alphabet_train"
OUTPUT_PREFIX = "asl_dataset_batch_"
LABELS_FILE = "asl_labels.json"
IMG_SIZE = 224
BATCH_SIZE = 2000

# ==============================
# MediaPipe Hand Detector
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ==============================
# Data Augmentation
# ==============================
def augment_image(img):
    if np.random.rand() < 0.3:
        factor = 0.5 + np.random.uniform()
        img = np.clip(img * factor, 0, 1)
    if np.random.rand() < 0.3:
        img = np.fliplr(img)
    if np.random.rand() < 0.3:
        angle = np.random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1)
        img = cv2.warpAffine((img * 255).astype(np.uint8), M, (IMG_SIZE, IMG_SIZE)) / 255.0
    if np.random.rand() < 0.2:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur((img * 255).astype(np.uint8), (k, k), 0) / 255.0
    return img

# ==============================
# Hand Extraction
# ==============================
def extract_hand(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        h, w, c = img.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)
        margin = 20
        x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
        x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
        hand_img = img[y_min:y_max, x_min:x_max]
    else:
        hand_img = img
    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    return hand_img / 255.0

# ==============================
# Dataset Builder with Batching
# ==============================
def build_dataset():
    X, y = [], []
    label_map = {}
    batch_idx = 0
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
    folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    if not folders:
        raise ValueError(f"No class folders found in {DATASET_PATH}")
    for label_idx, folder in enumerate(folders):
        folder_path = os.path.join(DATASET_PATH, folder)
        print(f"Processing class: {folder} (label {label_idx})")
        label_map[label_idx] = folder
        files = os.listdir(folder_path)
        for file in files:
            path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[-1].lower()
            if ext not in [".jpg", ".png", ".jpeg"]:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            processed = extract_hand(img)
            processed = augment_image(processed)
            X.append(processed)
            y.append(label_idx)
            if len(X) >= BATCH_SIZE:
                batch_file = f"{OUTPUT_PREFIX}{batch_idx}.npz"
                np.savez_compressed(batch_file, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int32))
                print(f"Saved batch {batch_idx} with {len(X)} samples -> {batch_file}")
                X, y = [], []
                batch_idx += 1
    if X:
        batch_file = f"{OUTPUT_PREFIX}{batch_idx}.npz"
        np.savez_compressed(batch_file, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int32))
        print(f"Saved batch {batch_idx} with {len(X)} samples -> {batch_file}")
    with open(LABELS_FILE, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Labels saved to {LABELS_FILE}")
    return label_map

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    try:
        label_map = build_dataset()
    except Exception as e:
        print(f"Error building dataset: {e}")
    finally:
        hands.close()
