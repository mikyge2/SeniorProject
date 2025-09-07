#!/usr/bin/env python3
"""
Perfect ASL Dataset Preprocessor
Advanced hand detection and image preprocessing for ASL recognition
"""

import cv2
import os
import numpy as np
import mediapipe as mp
import json
import logging
import argparse
from collections import defaultdict, Counter
from typing import Tuple, Optional, Dict, List, Set
import time
from pathlib import Path
import multiprocessing as mp_proc
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


# ==============================
# ENHANCED CONFIG
# ==============================
class PreprocessorConfig:
    def __init__(self):
        # Paths
        self.DATASET_PATH = "/home/mikege2/Documents/Senior Project/ASL - Kaggle/asl_alphabet_train/asl_alphabet_train"
        self.OUTPUT_PREFIX = "asl_dataset_batch_"
        self.LABELS_FILE = "asl_labels.json"
        self.STATS_FILE = "preprocessing_stats.json"
        self.ANALYSIS_FILE = "dataset_analysis.json"
        self.PLOTS_DIR = "preprocessing_plots"

        # Image processing
        self.IMG_SIZE = 224
        self.BATCH_SIZE = 2000
        self.MIN_HAND_CONFIDENCE = 0.7
        self.MIN_HAND_SIZE = 40  # Minimum hand region size
        self.MAX_HAND_SIZE = 0.9  # Maximum proportion of image
        self.MARGIN_RATIO = 0.15  # Margin around hand as ratio of hand size

        # Quality control
        self.MIN_SAMPLES_PER_CLASS = 50
        self.MAX_SAMPLES_PER_CLASS = 10000
        self.BLUR_THRESHOLD = 100  # Laplacian variance threshold for blur detection
        self.BRIGHTNESS_RANGE = (0.1, 0.9)  # Valid brightness range

        # Data splits
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.2
        self.TEST_SPLIT = 0.1

        # Processing
        self.USE_MULTIPROCESSING = True
        self.N_WORKERS = min(8, mp_proc.cpu_count())
        self.PREVIEW_SAMPLES = 5  # Number of samples to save for preview

        # Augmentation
        self.AUGMENT_DATA = True
        self.AUGMENT_MULTIPLIER = 2  # How many augmented versions per original

        # Advanced features
        self.SAVE_INTERMEDIATE = True  # Save hand detection visualizations
        self.VALIDATE_QUALITY = True
        self.BALANCE_CLASSES = True
        self.REMOVE_DUPLICATES = True


# ==============================
# ENHANCED LOGGING
# ==============================
def setup_logging(config: PreprocessorConfig):
    """Setup comprehensive logging"""
    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/preprocessing_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

    # Create specific loggers
    logger = logging.getLogger('ASL_Preprocessor')
    quality_logger = logging.getLogger('Quality_Control')
    stats_logger = logging.getLogger('Statistics')

    return logger, quality_logger, stats_logger


# ==============================
# ADVANCED STATISTICS TRACKING
# ==============================
class AdvancedPreprocessingStats:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.start_time = time.time()

        # Counters
        self.total_processed = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.skipped_files = 0
        self.quality_failures = 0
        self.duplicates_removed = 0

        # Quality metrics
        self.blur_failures = 0
        self.brightness_failures = 0
        self.size_failures = 0

        # Per-class tracking
        self.class_distribution = defaultdict(int)
        self.class_quality_stats = defaultdict(lambda: {
            'total': 0, 'passed': 0, 'blur_fail': 0,
            'brightness_fail': 0, 'size_fail': 0
        })

        # Processing times
        self.processing_times = []
        self.hand_detection_times = []
        self.quality_check_times = []

        # Image statistics
        self.image_sizes = []
        self.hand_sizes = []
        self.brightness_values = []
        self.blur_values = []

        # Batch information
        self.batches_created = 0
        self.samples_per_batch = []

    def log_successful_detection(self, class_label: str, processing_time: float,
                                 hand_detection_time: float, quality_check_time: float,
                                 hand_size: Tuple[int, int], brightness: float, blur_score: float):
        self.total_processed += 1
        self.successful_detections += 1
        self.class_distribution[class_label] += 1
        self.class_quality_stats[class_label]['total'] += 1
        self.class_quality_stats[class_label]['passed'] += 1

        self.processing_times.append(processing_time)
        self.hand_detection_times.append(hand_detection_time)
        self.quality_check_times.append(quality_check_time)

        self.hand_sizes.append(hand_size)
        self.brightness_values.append(brightness)
        self.blur_values.append(blur_score)

    def log_quality_failure(self, class_label: str, failure_type: str):
        self.total_processed += 1
        self.quality_failures += 1
        self.class_quality_stats[class_label]['total'] += 1

        if failure_type == 'blur':
            self.blur_failures += 1
            self.class_quality_stats[class_label]['blur_fail'] += 1
        elif failure_type == 'brightness':
            self.brightness_failures += 1
            self.class_quality_stats[class_label]['brightness_fail'] += 1
        elif failure_type == 'size':
            self.size_failures += 1
            self.class_quality_stats[class_label]['size_fail'] += 1

    def log_failed_detection(self, class_label: str = None):
        self.total_processed += 1
        self.failed_detections += 1
        if class_label:
            self.class_quality_stats[class_label]['total'] += 1

    def log_batch_created(self, batch_size: int):
        self.batches_created += 1
        self.samples_per_batch.append(batch_size)

    def get_comprehensive_summary(self) -> Dict:
        total_time = time.time() - self.start_time

        return {
            "processing_summary": {
                "total_files_processed": self.total_processed,
                "successful_detections": self.successful_detections,
                "failed_detections": self.failed_detections,
                "quality_failures": self.quality_failures,
                "skipped_files": self.skipped_files,
                "duplicates_removed": self.duplicates_removed,
                "detection_rate_percent": round((self.successful_detections / max(1, self.total_processed)) * 100, 2),
                "quality_pass_rate_percent": round(
                    (self.successful_detections / max(1, self.total_processed - self.skipped_files)) * 100, 2)
            },
            "quality_analysis": {
                "blur_failures": self.blur_failures,
                "brightness_failures": self.brightness_failures,
                "size_failures": self.size_failures,
                "average_brightness": round(np.mean(self.brightness_values) if self.brightness_values else 0, 3),
                "average_blur_score": round(np.mean(self.blur_values) if self.blur_values else 0, 2),
                "average_hand_area": round(np.mean([w * h for w, h in self.hand_sizes]) if self.hand_sizes else 0, 2)
            },
            "performance_metrics": {
                "total_processing_time_seconds": round(total_time, 2),
                "average_processing_time_ms": round(
                    np.mean(self.processing_times) * 1000 if self.processing_times else 0, 2),
                "average_hand_detection_time_ms": round(
                    np.mean(self.hand_detection_times) * 1000 if self.hand_detection_times else 0, 2),
                "images_per_second": round(self.total_processed / total_time if total_time > 0 else 0, 2)
            },
            "class_distribution": dict(self.class_distribution),
            "class_quality_details": dict(self.class_quality_stats),
            "batch_information": {
                "total_batches": self.batches_created,
                "samples_per_batch": self.samples_per_batch,
                "average_batch_size": round(np.mean(self.samples_per_batch) if self.samples_per_batch else 0, 2)
            }
        }


# ==============================
# ADVANCED HAND DETECTOR
# ==============================
class AdvancedHandDetector:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=config.MIN_HAND_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Hand landmark indices for different regions
        self.PALM_LANDMARKS = [0, 1, 2, 5, 9, 13, 17]
        self.FINGER_LANDMARKS = [4, 8, 12, 16, 20]

    def detect_hand_advanced(self, img: np.ndarray, save_visualization: bool = False,
                             filename: str = "") -> Tuple[Optional[np.ndarray], Dict, bool]:
        """
        Advanced hand detection with quality metrics
        """
        start_time = time.time()

        try:
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            detection_time = time.time() - start_time

            if not results.multi_hand_landmarks:
                return None, {'detection_time': detection_time, 'hand_found': False}, False

            h, w = img.shape[:2]
            hand_landmarks = results.multi_hand_landmarks[0]

            # Calculate bounding box with landmark analysis
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Calculate hand metrics
            hand_width = x_max - x_min
            hand_height = y_max - y_min
            hand_area = hand_width * hand_height
            image_area = h * w
            hand_coverage = hand_area / image_area

            # Quality checks
            if hand_width < self.config.MIN_HAND_SIZE or hand_height < self.config.MIN_HAND_SIZE:
                return None, {'detection_time': detection_time, 'hand_found': True,
                              'failure_reason': 'too_small'}, False

            if hand_coverage > self.config.MAX_HAND_SIZE:
                return None, {'detection_time': detection_time, 'hand_found': True,
                              'failure_reason': 'too_large'}, False

            # Calculate optimal margin
            margin = max(20, int(min(hand_width, hand_height) * self.config.MARGIN_RATIO))

            # Apply margin with bounds checking
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Extract hand region
            hand_region = img[y_min:y_max, x_min:x_max]

            # Save visualization if requested
            if save_visualization and filename:
                self._save_hand_visualization(img_rgb, hand_landmarks,
                                              (x_min, y_min, x_max, y_max), filename)

            # Resize to target size
            hand_resized = cv2.resize(hand_region, (self.config.IMG_SIZE, self.config.IMG_SIZE))
            hand_normalized = hand_resized.astype(np.float32) / 255.0

            # Return metrics
            metrics = {
                'detection_time': detection_time,
                'hand_found': True,
                'hand_size': (hand_width, hand_height),
                'hand_coverage': hand_coverage,
                'bounding_box': (x_min, y_min, x_max, y_max),
                'margin_applied': margin
            }

            return hand_normalized, metrics, True

        except Exception as e:
            return None, {'detection_time': time.time() - start_time, 'error': str(e)}, False

    def _save_hand_visualization(self, img_rgb: np.ndarray, hand_landmarks, bbox, filename: str):
        """Save hand detection visualization"""
        try:
            os.makedirs(f"{self.config.PLOTS_DIR}/hand_detections", exist_ok=True)

            # Create visualization
            img_vis = img_rgb.copy()
            self.mp_drawing.draw_landmarks(img_vis, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Draw bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Save
            vis_path = f"{self.config.PLOTS_DIR}/hand_detections/{filename}_detection.jpg"
            cv2.imwrite(vis_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logging.getLogger('ASL_Preprocessor').warning(f"Could not save visualization: {e}")

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


# ==============================
# ADVANCED QUALITY CONTROLLER
# ==============================
class QualityController:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def assess_image_quality(self, img: np.ndarray) -> Tuple[bool, Dict, str]:
        """
        Comprehensive image quality assessment
        """
        start_time = time.time()

        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # 1. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_sharp = laplacian_var > self.config.BLUR_THRESHOLD

            # 2. Brightness analysis
            brightness = np.mean(gray) / 255.0
            is_well_lit = self.config.BRIGHTNESS_RANGE[0] <= brightness <= self.config.BRIGHTNESS_RANGE[1]

            # 3. Contrast analysis
            contrast = gray.std()
            is_good_contrast = contrast > 20  # Minimum contrast threshold

            # 4. Noise analysis using median filter difference
            median_filtered = cv2.medianBlur(gray, 5)
            noise_level = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
            is_low_noise = noise_level < 10

            # Overall quality decision
            quality_checks = [is_sharp, is_well_lit, is_good_contrast, is_low_noise]
            passes_quality = sum(quality_checks) >= 3  # Pass if 3 out of 4 criteria met

            # Determine failure reason if applicable
            failure_reason = ""
            if not is_sharp:
                failure_reason = "blur"
            elif not is_well_lit:
                failure_reason = "brightness"
            elif not is_good_contrast:
                failure_reason = "contrast"
            elif not is_low_noise:
                failure_reason = "noise"

            metrics = {
                'assessment_time': time.time() - start_time,
                'blur_score': laplacian_var,
                'brightness': brightness,
                'contrast': contrast,
                'noise_level': noise_level,
                'is_sharp': is_sharp,
                'is_well_lit': is_well_lit,
                'is_good_contrast': is_good_contrast,
                'is_low_noise': is_low_noise,
                'quality_score': sum(quality_checks) / 4.0
            }

            return passes_quality, metrics, failure_reason

        except Exception as e:
            return False, {'error': str(e), 'assessment_time': time.time() - start_time}, "error"


# ==============================
# ADVANCED DATA AUGMENTATION
# ==============================
class AdvancedAugmentor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def augment_image_advanced(self, img: np.ndarray, augmentation_strength: float = 0.5) -> np.ndarray:
        """
        Apply advanced data augmentation optimized for ASL
        """
        augmented = img.copy()

        # 1. Brightness and contrast (conservative for sign language)
        if np.random.rand() < 0.4:
            alpha = 0.8 + np.random.rand() * 0.4  # Contrast: 0.8-1.2
            beta = -0.1 + np.random.rand() * 0.2  # Brightness: -0.1 to 0.1
            augmented = np.clip(augmented * alpha + beta, 0, 1)

        # 2. Slight rotation (preserve hand orientation)
        if np.random.rand() < 0.3:
            angle = np.random.uniform(-8, 8) * augmentation_strength
            center = (self.config.IMG_SIZE // 2, self.config.IMG_SIZE // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            augmented_uint8 = (augmented * 255).astype(np.uint8)
            augmented = cv2.warpAffine(augmented_uint8, M,
                                       (self.config.IMG_SIZE, self.config.IMG_SIZE)) / 255.0

        # 3. Small translation
        if np.random.rand() < 0.25:
            tx = int(np.random.uniform(-8, 8) * augmentation_strength)
            ty = int(np.random.uniform(-8, 8) * augmentation_strength)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            augmented_uint8 = (augmented * 255).astype(np.uint8)
            augmented = cv2.warpAffine(augmented_uint8, M,
                                       (self.config.IMG_SIZE, self.config.IMG_SIZE)) / 255.0

        # 4. Subtle noise addition
        if np.random.rand() < 0.15:
            noise = np.random.normal(0, 0.02 * augmentation_strength, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)

        # 5. Slight scale variation
        if np.random.rand() < 0.2:
            scale_factor = 0.95 + np.random.rand() * 0.1 * augmentation_strength  # 0.95-1.05
            new_size = int(self.config.IMG_SIZE * scale_factor)
            augmented_uint8 = (augmented * 255).astype(np.uint8)
            scaled = cv2.resize(augmented_uint8, (new_size, new_size))

            # Center crop or pad to maintain size
            if new_size > self.config.IMG_SIZE:
                start = (new_size - self.config.IMG_SIZE) // 2
                augmented = scaled[start:start + self.config.IMG_SIZE,
                start:start + self.config.IMG_SIZE] / 255.0
            else:
                pad = (self.config.IMG_SIZE - new_size) // 2
                augmented = np.pad(scaled, ((pad, pad), (pad, pad), (0, 0)),
                                   mode='constant', constant_values=0) / 255.0

        return augmented.astype(np.float32)


# ==============================
# ADVANCED DATASET ANALYZER
# ==============================
class DatasetAnalyzer:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

    def analyze_dataset_structure(self, dataset_path: str) -> Dict:
        """Comprehensive dataset structure analysis"""

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        folders = [f for f in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, f))]

        if not folders:
            raise ValueError(f"No class folders found in {dataset_path}")

        analysis = {
            'total_classes': len(folders),
            'class_names': sorted(folders),
            'class_file_counts': {},
            'total_files': 0,
            'file_extensions': Counter(),
            'problematic_classes': [],
            'recommendations': []
        }

        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            analysis['class_file_counts'][folder] = len(files)
            analysis['total_files'] += len(files)

            # Track extensions
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                analysis['file_extensions'][ext] += 1

            # Identify problematic classes
            if len(files) < self.config.MIN_SAMPLES_PER_CLASS:
                analysis['problematic_classes'].append({
                    'class': folder,
                    'count': len(files),
                    'issue': 'insufficient_samples'
                })

        # Generate recommendations
        self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict):
        """Generate preprocessing recommendations"""
        min_samples = min(analysis['class_file_counts'].values())
        max_samples = max(analysis['class_file_counts'].values())

        if max_samples / min_samples > 10:
            analysis['recommendations'].append("High class imbalance detected - consider balancing")

        if analysis['problematic_classes']:
            analysis['recommendations'].append("Some classes have insufficient samples")

        if analysis['total_files'] > 100000:
            analysis['recommendations'].append("Large dataset - consider using multiprocessing")

    def create_analysis_plots(self, analysis: Dict, stats: AdvancedPreprocessingStats):
        """Create comprehensive analysis plots"""

        # 1. Class distribution plot
        plt.figure(figsize=(15, 8))
        classes = list(analysis['class_file_counts'].keys())
        counts = list(analysis['class_file_counts'].values())

        plt.subplot(2, 3, 1)
        plt.bar(range(len(classes)), counts)
        plt.title('Original Class Distribution')
        plt.xlabel('Class Index')
        plt.ylabel('Sample Count')
        plt.xticks(range(len(classes)), [c[:3] for c in classes], rotation=45)

        # 2. Processed class distribution
        if stats.class_distribution:
            plt.subplot(2, 3, 2)
            processed_classes = list(stats.class_distribution.keys())
            processed_counts = list(stats.class_distribution.values())
            plt.bar(range(len(processed_classes)), processed_counts, color='green')
            plt.title('Processed Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Successful Samples')
            plt.xticks(range(len(processed_classes)), processed_classes, rotation=45)

        # 3. Quality metrics
        if stats.blur_values and stats.brightness_values:
            plt.subplot(2, 3, 3)
            plt.hist(stats.blur_values, bins=30, alpha=0.7, label='Blur Scores')
            plt.axvline(self.config.BLUR_THRESHOLD, color='red', linestyle='--', label='Threshold')
            plt.title('Image Blur Distribution')
            plt.xlabel('Laplacian Variance')
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(2, 3, 4)
            plt.hist(stats.brightness_values, bins=30, alpha=0.7, color='orange')
            plt.axvline(self.config.BRIGHTNESS_RANGE[0], color='red', linestyle='--')
            plt.axvline(self.config.BRIGHTNESS_RANGE[1], color='red', linestyle='--')
            plt.title('Image Brightness Distribution')
            plt.xlabel('Normalized Brightness')
            plt.ylabel('Frequency')

        # 4. Hand size distribution
        if stats.hand_sizes:
            plt.subplot(2, 3, 5)
            hand_areas = [w * h for w, h in stats.hand_sizes]
            plt.hist(hand_areas, bins=30, alpha=0.7, color='purple')
            plt.title('Hand Size Distribution')
            plt.xlabel('Hand Area (pixels²)')
            plt.ylabel('Frequency')

        # 5. Processing performance
        if stats.processing_times:
            plt.subplot(2, 3, 6)
            plt.hist(stats.processing_times, bins=30, alpha=0.7, color='cyan')
            plt.title('Processing Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'{self.config.PLOTS_DIR}/preprocessing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create quality analysis heatmap
        self._create_quality_heatmap(stats)

    def _create_quality_heatmap(self, stats: AdvancedPreprocessingStats):
        """Create quality analysis heatmap"""
        if not stats.class_quality_stats:
            return

        classes = list(stats.class_quality_stats.keys())
        metrics = ['passed', 'blur_fail', 'brightness_fail', 'size_fail']

        data = []
        for class_name in classes:
            class_stats = stats.class_quality_stats[class_name]
            total = max(1, class_stats['total'])
            row = [
                class_stats['passed'] / total,
                class_stats['blur_fail'] / total,
                class_stats['brightness_fail'] / total,
                class_stats['size_fail'] / total
            ]
            data.append(row)

        plt.figure(figsize=(10, max(6, len(classes) * 0.3)))
        sns.heatmap(data,
                    xticklabels=metrics,
                    yticklabels=classes,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    center=0.5)
        plt.title('Quality Analysis by Class')
        plt.ylabel('Classes')
        plt.xlabel('Quality Metrics (Proportion)')
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOTS_DIR}/quality_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


# ==============================
# PERFECT DATASET BUILDER
# ==============================
class PerfectDatasetBuilder:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.logger, self.quality_logger, self.stats_logger = setup_logging(config)

        # Initialize components
        self.stats = AdvancedPreprocessingStats(config)
        self.hand_detector = AdvancedHandDetector(config)
        self.quality_controller = QualityController(config)
        self.augmentor = AdvancedAugmentor(config)
        self.analyzer = DatasetAnalyzer(config)

        # Data storage
        self.processed_samples = []
        self.sample_hashes = set()  # For duplicate detection

    def _hash_image(self, img: np.ndarray) -> str:
        """Create hash for duplicate detection"""
        # Use perceptual hashing for better duplicate detection
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (8, 8))
        avg = resized.mean()
        binary = resized > avg
        return ''.join(['1' if pixel else '0' for pixel in binary.flatten()])

    def process_single_image(self, args: Tuple) -> Tuple[bool, Dict]:
        """Process a single image with comprehensive quality control"""
        file_path, class_label, save_preview = args

        overall_start = time.time()

        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                return False, {'error': 'Could not load image', 'file_path': file_path}

            # Initial quality assessment
            quality_start = time.time()
            passes_quality, quality_metrics, failure_reason = self.quality_controller.assess_image_quality(img)
            quality_time = time.time() - quality_start

            if not passes_quality and self.config.VALIDATE_QUALITY:
                return False, {
                    'quality_failure': True,
                    'failure_reason': failure_reason,
                    'quality_metrics': quality_metrics,
                    'file_path': file_path
                }

            # Hand detection
            hand_start = time.time()
            filename = os.path.splitext(os.path.basename(file_path))[0] if save_preview else ""
            processed_img, detection_metrics, success = self.hand_detector.detect_hand_advanced(
                img, save_visualization=save_preview, filename=filename
            )
            hand_time = time.time() - hand_start

            if not success or processed_img is None:
                return False, {
                    'hand_detection_failure': True,
                    'detection_metrics': detection_metrics,
                    'file_path': file_path
                }

            # Check for duplicates
            if self.config.REMOVE_DUPLICATES:
                img_hash = self._hash_image(processed_img)
                if img_hash in self.sample_hashes:
                    return False, {
                        'duplicate': True,
                        'hash': img_hash,
                        'file_path': file_path
                    }
                self.sample_hashes.add(img_hash)

            # Create augmented versions if enabled
            samples_to_add = [processed_img]
            if self.config.AUGMENT_DATA:
                for i in range(self.config.AUGMENT_MULTIPLIER):
                    augmented = self.augmentor.augment_image_advanced(processed_img)
                    samples_to_add.append(augmented)

            # Save preview if requested
            if save_preview:
                self._save_preview_samples(samples_to_add, class_label, filename)

            total_time = time.time() - overall_start

            return True, {
                'samples': samples_to_add,
                'class_label': class_label,
                'processing_time': total_time,
                'hand_detection_time': hand_time,
                'quality_check_time': quality_time,
                'quality_metrics': quality_metrics,
                'detection_metrics': detection_metrics,
                'file_path': file_path
            }

        except Exception as e:
            return False, {
                'error': str(e),
                'file_path': file_path,
                'processing_time': time.time() - overall_start
            }

    def _save_preview_samples(self, samples: List[np.ndarray], class_label: str, filename: str):
        """Save preview samples for quality inspection"""
        try:
            preview_dir = f"{self.config.PLOTS_DIR}/preview_samples/{class_label}"
            os.makedirs(preview_dir, exist_ok=True)

            for i, sample in enumerate(samples):
                sample_uint8 = (sample * 255).astype(np.uint8)
                suffix = "original" if i == 0 else f"aug_{i}"
                preview_path = f"{preview_dir}/{filename}_{suffix}.jpg"
                cv2.imwrite(preview_path, sample_uint8)

        except Exception as e:
            self.logger.warning(f"Could not save preview samples: {e}")

    def balance_dataset(self, samples_by_class: Dict[str, List]) -> Dict[str, List]:
        """Balance dataset by class"""
        if not self.config.BALANCE_CLASSES:
            return samples_by_class

        self.logger.info("Balancing dataset...")

        # Find target count (median or configurable)
        counts = [len(samples) for samples in samples_by_class.values()]
        target_count = min(max(counts), self.config.MAX_SAMPLES_PER_CLASS)
        target_count = max(target_count, self.config.MIN_SAMPLES_PER_CLASS)

        balanced_samples = {}
        for class_name, samples in samples_by_class.items():
            if len(samples) > target_count:
                # Randomly sample down
                indices = np.random.choice(len(samples), target_count, replace=False)
                balanced_samples[class_name] = [samples[i] for i in indices]
                self.logger.info(f"Downsampled {class_name}: {len(samples)} -> {target_count}")
            elif len(samples) < target_count and len(samples) >= self.config.MIN_SAMPLES_PER_CLASS:
                # Upsample with augmentation
                balanced_samples[class_name] = samples[:]
                needed = target_count - len(samples)
                for _ in range(needed):
                    # Pick random sample and augment
                    base_sample = np.random.choice(samples)
                    augmented = self.augmentor.augment_image_advanced(base_sample, augmentation_strength=0.8)
                    balanced_samples[class_name].append(augmented)
                self.logger.info(f"Upsampled {class_name}: {len(samples)} -> {target_count}")
            else:
                balanced_samples[class_name] = samples
                if len(samples) < self.config.MIN_SAMPLES_PER_CLASS:
                    self.logger.warning(f"Class {class_name} has insufficient samples: {len(samples)}")

        return balanced_samples

    def create_stratified_batches(self, samples_by_class: Dict[str, List]) -> Tuple[List, Dict]:
        """Create stratified batches ensuring class distribution"""
        self.logger.info("Creating stratified batches...")

        # Flatten samples with labels
        all_samples = []
        label_map = {}

        for label_idx, (class_name, samples) in enumerate(samples_by_class.items()):
            label_map[label_idx] = class_name
            for sample in samples:
                all_samples.append((sample, label_idx))

        # Shuffle samples
        np.random.shuffle(all_samples)

        # Create batches
        batches = []
        current_batch_X = []
        current_batch_y = []

        for sample, label in all_samples:
            current_batch_X.append(sample)
            current_batch_y.append(label)

            if len(current_batch_X) >= self.config.BATCH_SIZE:
                # Convert to numpy arrays
                batch_X = np.array(current_batch_X, dtype=np.float32)
                batch_y = np.array(current_batch_y, dtype=np.int32)

                batches.append((batch_X, batch_y))
                self.stats.log_batch_created(len(current_batch_X))

                current_batch_X = []
                current_batch_y = []

        # Handle remaining samples
        if current_batch_X:
            batch_X = np.array(current_batch_X, dtype=np.float32)
            batch_y = np.array(current_batch_y, dtype=np.int32)
            batches.append((batch_X, batch_y))
            self.stats.log_batch_created(len(current_batch_X))

        return batches, label_map

    def save_batches_and_splits(self, batches: List, label_map: Dict):
        """Save batches with train/val/test splits"""
        self.logger.info("Saving batches with data splits...")

        # Create splits
        n_batches = len(batches)
        n_test = max(1, int(n_batches * self.config.TEST_SPLIT))
        n_val = max(1, int(n_batches * self.config.VAL_SPLIT))
        n_train = n_batches - n_test - n_val

        # Shuffle batches for random splits
        batch_indices = np.random.permutation(n_batches)

        train_indices = batch_indices[:n_train]
        val_indices = batch_indices[n_train:n_train + n_val]
        test_indices = batch_indices[n_train + n_val:]

        # Save batches with split information
        split_info = {
            'train_batches': [],
            'val_batches': [],
            'test_batches': []
        }

        for batch_idx, (X, y) in enumerate(batches):
            batch_file = f"{self.config.OUTPUT_PREFIX}{batch_idx:04d}.npz"
            np.savez_compressed(batch_file, X=X, y=y)

            # Track which split this batch belongs to
            if batch_idx in train_indices:
                split_info['train_batches'].append(batch_file)
            elif batch_idx in val_indices:
                split_info['val_batches'].append(batch_file)
            else:
                split_info['test_batches'].append(batch_file)

            self.logger.info(f"Saved batch {batch_idx}: {batch_file} with {len(X)} samples")

        # Save label mapping
        with open(self.config.LABELS_FILE, 'w') as f:
            json.dump(label_map, f, indent=2)

        # Save split information
        with open('data_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        self.logger.info(
            f"Created {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test batches")

    def build_perfect_dataset(self) -> Dict:
        """Build the perfect ASL dataset"""

        self.logger.info("🚀 Starting Perfect ASL Dataset Preprocessing")
        self.logger.info(f"Configuration: {self.config.__dict__}")

        # 1. Analyze dataset structure
        self.logger.info("📊 Analyzing dataset structure...")
        analysis = self.analyzer.analyze_dataset_structure(self.config.DATASET_PATH)
        self.logger.info(f"Found {analysis['total_classes']} classes with {analysis['total_files']} total files")

        # Save analysis
        with open(self.config.ANALYSIS_FILE, 'w') as f:
            json.dump(analysis, f, indent=2)

        # 2. Prepare file list for processing
        file_list = []
        preview_count = 0

        for class_name in analysis['class_names']:
            class_path = os.path.join(self.config.DATASET_PATH, class_name)
            files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            for file in files:
                file_path = os.path.join(class_path, file)
                save_preview = (self.config.SAVE_INTERMEDIATE and
                                preview_count < self.config.PREVIEW_SAMPLES)
                file_list.append((file_path, class_name, save_preview))
                if save_preview:
                    preview_count += 1

        self.logger.info(f"Prepared {len(file_list)} files for processing")

        # 3. Process images
        self.logger.info("🔄 Processing images...")
        samples_by_class = defaultdict(list)

        if self.config.USE_MULTIPROCESSING and len(file_list) > 100:
            # Multiprocessing for large datasets
            with mp_proc.Pool(processes=self.config.N_WORKERS) as pool:
                results = pool.map(self.process_single_image, file_list)
        else:
            # Sequential processing
            results = [self.process_single_image(args) for args in file_list]

        # 4. Collect results and update statistics
        for i, (success, result_data) in enumerate(results):
            file_path, class_label, save_preview = file_list[i]  # Get original args

            if success:
                samples = result_data['samples']
                quality_metrics = result_data['quality_metrics']
                detection_metrics = result_data['detection_metrics']

                # Add samples to class collection
                for sample in samples:
                    samples_by_class[class_label].append(sample)

                # Update statistics
                self.stats.log_successful_detection(
                    class_label,
                    result_data['processing_time'],
                    result_data['hand_detection_time'],
                    result_data['quality_check_time'],
                    detection_metrics.get('hand_size', (0, 0)),
                    quality_metrics.get('brightness', 0),
                    quality_metrics.get('blur_score', 0)
                )

            else:
                # Handle failures
                if result_data.get('quality_failure'):
                    self.stats.log_quality_failure(class_label, result_data['failure_reason'])
                elif result_data.get('duplicate'):
                    self.stats.duplicates_removed += 1
                else:
                    self.stats.log_failed_detection(class_label)

            # Progress logging
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / len(file_list) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(file_list)})")

        # 5. Balance dataset if enabled
        if self.config.BALANCE_CLASSES:
            samples_by_class = self.balance_dataset(dict(samples_by_class))

        # 6. Create stratified batches
        batches, label_map = self.create_stratified_batches(dict(samples_by_class))

        # 7. Save batches and create splits
        self.save_batches_and_splits(batches, label_map)

        # 8. Generate comprehensive statistics and plots
        final_stats = self.stats.get_comprehensive_summary()

        with open(self.config.STATS_FILE, 'w') as f:
            json.dump(final_stats, f, indent=2)

        # 9. Create analysis plots
        self.analyzer.create_analysis_plots(analysis, self.stats)

        # 10. Print final report
        self._print_final_report(final_stats, analysis, samples_by_class)

        return final_stats

    def _print_final_report(self, stats: Dict, analysis: Dict, samples_by_class: Dict):
        """Print comprehensive final report"""

        print("\n" + "=" * 80)
        print("🎉 PERFECT ASL DATASET PREPROCESSING COMPLETE!")
        print("=" * 80)

        processing = stats['processing_summary']
        quality = stats['quality_analysis']
        performance = stats['performance_metrics']

        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   Total files processed: {processing['total_files_processed']:,}")
        print(f"   Successful detections: {processing['successful_detections']:,}")
        print(f"   Failed detections: {processing['failed_detections']:,}")
        print(f"   Quality failures: {processing['quality_failures']:,}")
        print(f"   Duplicates removed: {processing['duplicates_removed']:,}")
        print(f"   Detection rate: {processing['detection_rate_percent']:.1f}%")
        print(f"   Quality pass rate: {processing['quality_pass_rate_percent']:.1f}%")

        print(f"\n🎯 QUALITY ANALYSIS:")
        print(f"   Average brightness: {quality['average_brightness']:.3f}")
        print(f"   Average blur score: {quality['average_blur_score']:.1f}")
        print(f"   Average hand area: {quality['average_hand_area']:.0f} pixels²")
        print(f"   Blur failures: {quality['blur_failures']:,}")
        print(f"   Brightness failures: {quality['brightness_failures']:,}")
        print(f"   Size failures: {quality['size_failures']:,}")

        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Total processing time: {performance['total_processing_time_seconds']:.1f} seconds")
        print(f"   Average processing time: {performance['average_processing_time_ms']:.1f} ms/image")
        print(f"   Hand detection time: {performance['average_hand_detection_time_ms']:.1f} ms/image")
        print(f"   Processing speed: {performance['images_per_second']:.1f} images/second")

        print(f"\n📁 DATASET STRUCTURE:")
        print(f"   Total classes: {len(samples_by_class)}")
        print(f"   Total batches created: {stats['batch_information']['total_batches']}")
        print(f"   Average batch size: {stats['batch_information']['average_batch_size']:.0f}")

        print(f"\n📈 CLASS DISTRIBUTION:")
        for class_name, samples in samples_by_class.items():
            print(f"   {class_name}: {len(samples):,} samples")

        if stats.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis.get('recommendations', []):
                print(f"   • {rec}")

        print(f"\n📂 OUTPUT FILES:")
        print(f"   • Batch files: {self.config.OUTPUT_PREFIX}*.npz")
        print(f"   • Labels: {self.config.LABELS_FILE}")
        print(f"   • Statistics: {self.config.STATS_FILE}")
        print(f"   • Analysis: {self.config.ANALYSIS_FILE}")
        print(f"   • Plots: {self.config.PLOTS_DIR}/")
        print(f"   • Data splits: data_splits.json")

        print("\n" + "=" * 80)
        print("✅ Dataset is ready for training!")
        print("=" * 80)

    def cleanup(self):
        """Clean up resources"""
        try:
            self.hand_detector.cleanup()
        except:
            pass


# ==============================
# VALIDATION FUNCTIONS
# ==============================
def validate_preprocessing_output(config: PreprocessorConfig) -> bool:
    """Comprehensive validation of preprocessing output"""
    logger = logging.getLogger('Validator')
    logger.info("🔍 Validating preprocessing output...")

    try:
        # Check for required files
        required_files = [config.LABELS_FILE, config.STATS_FILE, 'data_splits.json']
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Missing required file: {file}")
                return False

        # Check batch files
        batch_files = sorted([f for f in os.listdir('.') if f.startswith(config.OUTPUT_PREFIX)])
        if not batch_files:
            logger.error("No batch files found!")
            return False

        # Load and validate data splits
        with open('data_splits.json', 'r') as f:
            splits = json.load(f)

        total_samples = 0
        for batch_file in batch_files:
            try:
                data = np.load(batch_file)
                X, y = data['X'], data['y']

                # Validate shapes and types
                assert X.shape[0] == y.shape[0], f"Mismatched shapes in {batch_file}"
                assert X.shape[1:] == (config.IMG_SIZE, config.IMG_SIZE, 3), f"Wrong image shape in {batch_file}"
                assert X.dtype == np.float32, f"Wrong X dtype in {batch_file}: {X.dtype}"
                assert y.dtype == np.int32, f"Wrong y dtype in {batch_file}: {y.dtype}"

                # Validate value ranges
                assert np.all(X >= 0) and np.all(X <= 1), f"X values out of range in {batch_file}"
                assert np.all(y >= 0), f"Negative labels in {batch_file}"

                total_samples += len(X)
                logger.info(f"✅ {batch_file}: {len(X)} samples, shapes OK, types OK, ranges OK")

            except Exception as e:
                logger.error(f"❌ Error validating {batch_file}: {e}")
                return False

        # Validate splits coverage
        all_split_files = (splits['train_batches'] + splits['val_batches'] + splits['test_batches'])
        if set(all_split_files) != set(batch_files):
            logger.warning("Mismatch between batch files and split assignments")

        logger.info(f"✅ Validation passed! Total samples: {total_samples:,}")
        logger.info(f"   Train batches: {len(splits['train_batches'])}")
        logger.info(f"   Val batches: {len(splits['val_batches'])}")
        logger.info(f"   Test batches: {len(splits['test_batches'])}")

        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


# ==============================
# COMMAND LINE INTERFACE
# ==============================
def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Perfect ASL Dataset Preprocessor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='Path to the ASL dataset directory'
    )

    # Optional arguments
    parser.add_argument('--output-prefix', default='asl_dataset_batch_',
                        help='Prefix for output batch files')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Samples per batch file')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                        help='Minimum hand detection confidence')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes (0 = auto)')

    # Feature flags
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable class balancing')
    parser.add_argument('--no-quality-check', action='store_true',
                        help='Disable quality validation')
    parser.add_argument('--no-multiprocess', action='store_true',
                        help='Disable multiprocessing')
    parser.add_argument('--save-previews', action='store_true',
                        help='Save preview images for inspection')

    return parser


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    """Main execution function"""
    parser = create_parser()
    args = parser.parse_args()

    # Create configuration
    config = PreprocessorConfig()

    # Apply command line arguments
    config.DATASET_PATH = args.dataset_path
    config.OUTPUT_PREFIX = args.output_prefix
    config.IMG_SIZE = args.img_size
    config.BATCH_SIZE = args.batch_size
    config.MIN_HAND_CONFIDENCE = args.min_confidence

    if args.workers > 0:
        config.N_WORKERS = args.workers

    # Apply feature flags
    config.AUGMENT_DATA = not args.no_augment
    config.BALANCE_CLASSES = not args.no_balance
    config.VALIDATE_QUALITY = not args.no_quality_check
    config.USE_MULTIPROCESSING = not args.no_multiprocess
    config.SAVE_INTERMEDIATE = args.save_previews

    # Initialize and run preprocessor
    builder = None
    try:
        builder = PerfectDatasetBuilder(config)
        final_stats = builder.build_perfect_dataset()

        # Validate output
        if validate_preprocessing_output(config):
            print("\n🎉 Perfect ASL dataset preprocessing completed successfully!")
            return 0
        else:
            print("\n❌ Preprocessing completed but validation failed!")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if builder:
            builder.cleanup()


if __name__ == "__main__":
    exit(main())