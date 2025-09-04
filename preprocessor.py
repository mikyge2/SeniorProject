import cv2
import os
import numpy as np
import mediapipe as mp
import json
import logging
from collections import defaultdict
from typing import Tuple, Optional, Dict, List
import time

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "/home/mikege2/Documents/Senior Project/ASL - Kaggle/asl_alphabet_train/asl_alphabet_train"
OUTPUT_PREFIX = "asl_dataset_batch_"
LABELS_FILE = "asl_labels.json"
STATS_FILE = "preprocessing_stats.json"
IMG_SIZE = 224
BATCH_SIZE = 2000
MIN_HAND_CONFIDENCE = 0.7  # Increased for better hand detection
MIN_HAND_SIZE = 50  # Minimum hand region size in pixels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# MediaPipe Hand Detector
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=MIN_HAND_CONFIDENCE,
    min_tracking_confidence=0.5
)


# ==============================
# Statistics Tracking
# ==============================
class PreprocessingStats:
    def __init__(self):
        self.total_processed = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.skipped_files = 0
        self.class_distribution = defaultdict(int)
        self.processing_times = []
        self.start_time = time.time()

    def log_successful_detection(self, class_label: str):
        self.total_processed += 1
        self.successful_detections += 1
        self.class_distribution[class_label] += 1

    def log_failed_detection(self):
        self.total_processed += 1
        self.failed_detections += 1

    def log_skipped_file(self):
        self.skipped_files += 1

    def add_processing_time(self, time_taken: float):
        self.processing_times.append(time_taken)

    def get_detection_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return (self.successful_detections / self.total_processed) * 100

    def get_summary(self) -> Dict:
        total_time = time.time() - self.start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        return {
            "total_files_processed": self.total_processed,
            "successful_detections": self.successful_detections,
            "failed_detections": self.failed_detections,
            "skipped_files": self.skipped_files,
            "detection_rate_percent": round(self.get_detection_rate(), 2),
            "class_distribution": dict(self.class_distribution),
            "total_processing_time_seconds": round(total_time, 2),
            "average_processing_time_ms": round(avg_processing_time * 1000, 2),
            "images_per_second": round(self.total_processed / total_time, 2) if total_time > 0 else 0
        }


# ==============================
# Enhanced Data Augmentation
# ==============================
def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation with improved parameters
    """
    # Brightness adjustment (more conservative range)
    if np.random.rand() < 0.3:
        factor = 0.7 + np.random.uniform() * 0.6  # Range: 0.7-1.3
        img = np.clip(img * factor, 0, 1)

    # Horizontal flip (be cautious with directional signs)
    if np.random.rand() < 0.25:  # Reduced probability
        img = np.fliplr(img)

    # Rotation (maintain reasonable range)
    if np.random.rand() < 0.3:
        angle = np.random.randint(-12, 13)  # Slightly reduced range
        center = (IMG_SIZE // 2, IMG_SIZE // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_uint8 = (img * 255).astype(np.uint8)
        img = cv2.warpAffine(img_uint8, M, (IMG_SIZE, IMG_SIZE)) / 255.0

    # Gaussian blur (for slight noise robustness)
    if np.random.rand() < 0.15:  # Reduced probability
        kernel_size = np.random.choice([3, 5])
        img_uint8 = (img * 255).astype(np.uint8)
        img = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0) / 255.0

    # Small translation
    if np.random.rand() < 0.2:
        tx = np.random.randint(-10, 11)
        ty = np.random.randint(-10, 11)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img_uint8 = (img * 255).astype(np.uint8)
        img = cv2.warpAffine(img_uint8, M, (IMG_SIZE, IMG_SIZE)) / 255.0

    return img


# ==============================
# Enhanced Hand Extraction
# ==============================
def extract_hand(img: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract hand region from image with quality validation

    Returns:
        Tuple[Optional[np.ndarray], bool]: (processed_image, success_flag)
    """
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, c = img.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            # Get bounding box from all hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

            # Validate hand region size
            hand_width = x_max - x_min
            hand_height = y_max - y_min

            if hand_width < MIN_HAND_SIZE or hand_height < MIN_HAND_SIZE:
                return None, False

            # Add adaptive margin based on hand size
            margin = max(20, min(hand_width, hand_height) // 10)
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Extract hand region
            hand_img = img[y_min:y_max, x_min:x_max]

            # Resize and normalize
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img_normalized = hand_img / 255.0

            return hand_img_normalized, True
        else:
            return None, False

    except Exception as e:
        logger.error(f"Error in hand extraction: {e}")
        return None, False


# ==============================
# Balanced Batch Creation
# ==============================
def create_balanced_batch(samples: List[Tuple[np.ndarray, int]], labels_count: Dict[int, int]) -> bool:
    """
    Check if batch has reasonable class distribution
    """
    if len(samples) < BATCH_SIZE // 2:  # At least half full
        return False

    batch_labels = [label for _, label in samples]
    unique_labels = len(set(batch_labels))

    # Ensure we have samples from multiple classes
    return unique_labels > 1


# ==============================
# Enhanced Dataset Builder
# ==============================
def build_dataset() -> Dict[int, str]:
    """
    Build dataset with comprehensive error handling and statistics
    """
    stats = PreprocessingStats()
    X, y = [], []
    label_map = {}
    batch_idx = 0

    # Validate dataset path
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

    # Get class folders
    folders = [f for f in os.listdir(DATASET_PATH)
               if os.path.isdir(os.path.join(DATASET_PATH, f))]

    if not folders:
        raise ValueError(f"No class folders found in {DATASET_PATH}")

    folders.sort()  # Ensure consistent ordering
    logger.info(f"Found {len(folders)} classes: {folders}")

    # Process each class
    for label_idx, folder in enumerate(folders):
        folder_path = os.path.join(DATASET_PATH, folder)
        label_map[label_idx] = folder

        logger.info(f"Processing class: {folder} (label {label_idx})")

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        class_processed = 0
        class_failed = 0

        for file in files:
            file_start_time = time.time()
            path = os.path.join(folder_path, file)

            try:
                # Load image
                img = cv2.imread(path)
                if img is None:
                    logger.warning(f"Could not load image: {path}")
                    stats.log_skipped_file()
                    continue

                # Extract hand
                processed_img, success = extract_hand(img)

                if not success or processed_img is None:
                    stats.log_failed_detection()
                    class_failed += 1
                    continue

                # Apply augmentation
                augmented_img = augment_image(processed_img)

                # Add to batch
                X.append(augmented_img)
                y.append(label_idx)

                # Update statistics
                stats.log_successful_detection(folder)
                class_processed += 1

                # Record processing time
                processing_time = time.time() - file_start_time
                stats.add_processing_time(processing_time)

                # Save batch when full
                if len(X) >= BATCH_SIZE:
                    batch_file = f"{OUTPUT_PREFIX}{batch_idx}.npz"
                    np.savez_compressed(
                        batch_file,
                        X=np.array(X, dtype=np.float32),
                        y=np.array(y, dtype=np.int32)
                    )
                    logger.info(f"Saved batch {batch_idx} with {len(X)} samples -> {batch_file}")
                    X, y = [], []
                    batch_idx += 1

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                stats.log_skipped_file()
                continue

        logger.info(f"Class {folder}: {class_processed} processed, {class_failed} failed")

    # Save final batch if exists
    if X:
        batch_file = f"{OUTPUT_PREFIX}{batch_idx}.npz"
        np.savez_compressed(
            batch_file,
            X=np.array(X, dtype=np.float32),
            y=np.array(y, dtype=np.int32)
        )
        logger.info(f"Saved final batch {batch_idx} with {len(X)} samples -> {batch_file}")

    # Save label mapping
    with open(LABELS_FILE, 'w') as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Labels saved to {LABELS_FILE}")

    # Save statistics
    summary = stats.get_summary()
    with open(STATS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final statistics
    logger.info("=" * 50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total files processed: {summary['total_files_processed']}")
    logger.info(f"Successful detections: {summary['successful_detections']}")
    logger.info(f"Failed detections: {summary['failed_detections']}")
    logger.info(f"Skipped files: {summary['skipped_files']}")
    logger.info(f"Detection rate: {summary['detection_rate_percent']}%")
    logger.info(f"Processing speed: {summary['images_per_second']} images/second")
    logger.info(f"Total batches created: {batch_idx + 1 if X else batch_idx}")

    # Warn if detection rate is low
    if summary['detection_rate_percent'] < 80:
        logger.warning("Low hand detection rate! Consider reviewing dataset quality.")

    return label_map


# ==============================
# Validation Functions
# ==============================
def validate_preprocessing_output():
    """
    Validate the preprocessing output files
    """
    batch_files = [f for f in os.listdir('.') if f.startswith(OUTPUT_PREFIX)]

    if not batch_files:
        logger.error("No batch files found!")
        return False

    total_samples = 0
    for batch_file in sorted(batch_files):
        try:
            data = np.load(batch_file)
            X, y = data['X'], data['y']
            total_samples += len(X)
            logger.info(f"{batch_file}: {len(X)} samples, shape: {X.shape}")

            # Basic validation
            assert X.dtype == np.float32, f"Wrong X dtype: {X.dtype}"
            assert y.dtype == np.int32, f"Wrong y dtype: {y.dtype}"
            assert X.shape[1:] == (IMG_SIZE, IMG_SIZE, 3), f"Wrong X shape: {X.shape}"
            assert np.all(X >= 0) and np.all(X <= 1), "X values not in [0,1] range"

        except Exception as e:
            logger.error(f"Error validating {batch_file}: {e}")
            return False

    logger.info(f"Validation passed! Total samples: {total_samples}")
    return True


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    try:
        logger.info("Starting enhanced ASL preprocessing...")
        logger.info(f"Dataset path: {DATASET_PATH}")
        logger.info(f"Output prefix: {OUTPUT_PREFIX}")
        logger.info(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
        logger.info(f"Batch size: {BATCH_SIZE}")

        # Build dataset
        label_map = build_dataset()

        # Validate output
        validate_preprocessing_output()

        logger.info("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
    finally:
        # Clean up MediaPipe resources
        hands.close()
        logger.info("Resources cleaned up.")