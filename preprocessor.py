#!/usr/bin/env python3
"""
Sign Language Dataset Preprocessor

A preprocessor for both alphabet (static images) and word (video) datasets.
Outputs TFRecords with hand landmarks and preprocessed images.

"""

import os
import json
import argparse
import logging
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Union
from collections import defaultdict
import time
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """Configuration class for preprocessing parameters.

    Attributes:
        dataset_path: Path to the root of the input dataset.
        output_path: Path where the processed data (TFRecords) will be saved.
        img_size: The target square size for images after resizing.
        frame_rate: The desired frames per second to extract from video files.
        batch_size: The number of items to process in a single batch.
        train_split: The proportion of the dataset to allocate for training.
        val_split: The proportion of the dataset to allocate for validation.
        test_split: The proportion of the dataset to allocate for testing.
        augmentation: A flag to enable or disable data augmentation.
        rotation_range: The range (in degrees) for random rotations during augmentation.
        brightness_range: The range for random brightness adjustments.
        crop_zoom_range: The range for random cropping and zooming.
    """
    dataset_path: str
    output_path: str
    img_size: int = 224
    frame_rate: float = 10.0  # FPS for video extraction
    batch_size: int = 100
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augmentation: bool = True
    rotation_range: float = 15.0
    brightness_range: float = 0.2
    crop_zoom_range: float = 0.1


class MediaPipeHandDetector:
    """A wrapper for MediaPipe Hands to detect and process hand landmarks.

    This class initializes the MediaPipe Hands solution and provides a simple
    interface to detect hands in an image, crop the hand region, and extract
    normalized 2D landmarks.

    Attributes:
        mp_hands: The MediaPipe hands solution.
        hands: The Hands processing object from MediaPipe.
    """

    def __init__(self, confidence: float = 0.5, max_hands: int = 2):
        """Initializes the MediaPipeHandDetector.

        Args:
            confidence: The minimum detection confidence for a hand to be detected.
            max_hands: The maximum number of hands to detect.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detects hands in an image and extracts landmarks and a cropped image.

        Processes a BGR image, detects the most prominent hand, calculates its
        bounding box, crops the hand region with padding, and extracts the
        normalized 2D landmarks.

        Args:
            image: The input image in BGR format.

        Returns:
            A tuple containing:
            - cropped_hand: A numpy array of the cropped hand region.
            - landmarks_2d: A numpy array of normalized 2D hand landmarks.
            Returns None if no hands are detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return None

        # Get the first hand (primary hand)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract 2D landmark coordinates
        h, w, _ = image.shape
        landmarks_2d = []

        x_coords, y_coords = [], []
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks_2d.extend([landmark.x, landmark.y])  # Normalized coordinates
            x_coords.append(x)
            y_coords.append(y)

        # Calculate bounding box with padding
        min_x, max_x = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
        min_y, max_y = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

        # Crop hand region
        cropped_hand = image[min_y:max_y, min_x:max_x]

        if cropped_hand.size == 0:
            return None

        return cropped_hand, np.array(landmarks_2d, dtype=np.float32)

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


class ImageProcessor:
    """Handles image preprocessing, including resizing, augmentation, and normalization.

    This class encapsulates the logic for preparing images for model training.
    It can apply a series of augmentations like rotation, brightness adjustment,
    and zooming.

    Attributes:
        config: A PreprocessConfig object containing preprocessing parameters.
    """

    def __init__(self, config: PreprocessConfig):
        """Initializes the ImageProcessor.

        Args:
            config: A PreprocessConfig object with settings for image processing.
        """
        self.config = config

    def preprocess(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """Resizes, optionally augments, and normalizes an image.

        Args:
            image: The input image as a numpy array.
            augment: If True, applies data augmentation based on the config.

        Returns:
            The preprocessed image, resized and normalized to a [0, 1] float range.
        """
        # Resize to target size
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))

        if augment and self.config.augmentation:
            image = self._augment(image)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Applies a series of random augmentations to an image.

        The augmentations include random rotation, brightness adjustment, and
        cropping/zooming, based on the ranges specified in the configuration.

        Args:
            image: The input image to augment.

        Returns:
            The augmented image.
        """
        h, w = image.shape[:2]

        # Random rotation (±rotation_range degrees)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random brightness adjustment
        if np.random.random() < 0.5:
            brightness_factor = 1.0 + np.random.uniform(-self.config.brightness_range,
                                                        self.config.brightness_range)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # Random crop and zoom
        if np.random.random() < 0.5:
            crop_factor = np.random.uniform(1.0 - self.config.crop_zoom_range, 1.0)
            new_h, new_w = int(h * crop_factor), int(w * crop_factor)

            if new_h < h and new_w < w:
                start_h = np.random.randint(0, h - new_h)
                start_w = np.random.randint(0, w - new_w)
                image = image[start_h:start_h + new_h, start_w:start_w + new_w]
                image = cv2.resize(image, (w, h))

        return image


class VideoProcessor:
    """Extracts frames from video files at a specified frame rate.

    Attributes:
        config: A PreprocessConfig object with settings for frame extraction.
    """

to extract frames from.
    """

    def __init__(self, config: PreprocessConfig):
        """Initializes the VideoProcessor.

        Args:
            config: A PreprocessConfig object containing the target frame rate.
        """
        self.config = config

    def extract_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Yields frames from a video file at a specified rate.

        Opens a video file and reads it frame by frame, yielding frames at an
        interval calculated to match the desired frame rate.

        Args:
            video_path: The path to the video file.

        Yields:
            Frames from the video as numpy arrays (in BGR format).
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.warning(f"Could not open video: {video_path}")
            return

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback

            frame_interval = max(1, int(fps / self.config.frame_rate))
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    yield frame

                frame_count += 1

        finally:
            cap.release()


class TFRecordWriter:
    """A utility class for creating and writing TFRecord examples."""

    @staticmethod
    def _bytes_feature(value: Union[bytes, tf.Tensor]) -> tf.train.Feature:
        """Creates a tf.train.Feature from a byte string.

        Args:
            value: The byte string or TensorFlow tensor to encode.

        Returns:
            A tf.train.Feature containing the bytes list.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value: Union[float, List[float], np.ndarray]) -> tf.train.Feature:
        """Creates a tf.train.Feature from a float or list of floats.

        Args:
            value: The float or list/array of floats to encode.

        Returns:
            A tf.train.Feature containing the float list.
        """
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _int64_feature(value: Union[int, List[int], np.ndarray]) -> tf.train.Feature:
        """Creates a tf.train.Feature from an integer or list of integers.

        Args:
            value: The integer or list/array of integers to encode.

        Returns:
            A tf.train.Feature containing the int64 list.
        """
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def create_example(image: np.ndarray, landmarks: np.ndarray,
                       label: int, label_text: str) -> tf.train.Example:
        """Creates a tf.train.Example proto from image, landmarks, and label.

        Args:
            image: The preprocessed image data (as a float array in [0, 1]).
            landmarks: The hand landmarks associated with the image.
            label: The integer label for the class.
            label_text: The string representation of the label.

        Returns:
            A tf.train.Example containing the encoded features.
        """
        # Encode image as PNG
        _, encoded_image = cv2.imencode('.png', (image * 255).astype(np.uint8))
        encoded_image = encoded_image.tobytes()

        feature = {
            'image/encoded': TFRecordWriter._bytes_feature(encoded_image),
            'image/landmarks': TFRecordWriter._float_feature(landmarks),
            'label': TFRecordWriter._int64_feature(label),
            'label_text': TFRecordWriter._bytes_feature(label_text.encode('utf-8'))
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


class DatasetDetector:
    """A utility to automatically detect the type of a dataset (images or videos)."""

    @staticmethod
    def detect_dataset_type(dataset_path: str) -> str:
        """Determines if a dataset consists of images or videos.

        Scans the files in the given directory and its subdirectories, counting
        common image and video file extensions to infer the dataset type.

        Args:
            dataset_path: The path to the dataset directory.

        Returns:
            'videos' if video files are more numerous, 'images' if image
            files are more numerous.

        Raises:
            ValueError: If no supported image or video files are found.
        """
        dataset_path = Path(dataset_path)

        # Common image extensions
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        # Common video extensions
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}

        image_count = 0
        video_count = 0

        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in image_exts:
                    image_count += 1
                elif ext in video_exts:
                    video_count += 1

        if video_count > image_count:
            return 'videos'
        elif image_count > 0:
            return 'images'
        else:
            raise ValueError(f"No supported media files found in {dataset_path}")


class SignLanguagePreprocessor:
    """Orchestrates the end-to-end sign language dataset preprocessing pipeline.

    This class manages the entire workflow:
    1. Detects the dataset type (images or videos).
    2. Gathers file paths and creates a class-to-integer mapping.
    3. Splits the data into training, validation, and test sets.
    4. Processes each data split in batches, handling both images and videos.
    5. For each item, it detects hands, extracts landmarks, preprocesses the image,
       and applies augmentation to the training set.
    6. Writes the processed data into TFRecord files for efficient loading.
    7. Saves a `metadata.json` file with details about the preprocessing run.

    Attributes:
        config: A PreprocessConfig object with all preprocessing settings.
        hand_detector: An instance of MediaPipeHandDetector.
        image_processor: An instance of ImageProcessor.
        video_processor: An instance of VideoProcessor.
        stats: A dictionary to keep track of processing statistics.
        logger: A logger for logging progress and warnings.
    """

    def __init__(self, config: PreprocessConfig):
        """Initializes the SignLanguagePreprocessor.

        Args:
            config: A PreprocessConfig object containing all settings.
        """
        self.config = config
        self.hand_detector = MediaPipeHandDetector()
        self.image_processor = ImageProcessor(config)
        self.video_processor = VideoProcessor(config)
        self.stats = defaultdict(int)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_class_mapping(self, dataset_path: str) -> Dict[str, int]:
        """Creates a mapping from class names (directory names) to integer labels.

        Args:
            dataset_path: The path to the dataset directory.

        Returns:
            A dictionary mapping class names to unique integer labels.
        """
        dataset_path = Path(dataset_path)
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        return {class_name: idx for idx, class_name in enumerate(classes)}

    def _split_data_paths(self, all_paths: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Shuffles and splits a list of file paths into train, validation, and test sets.

        Args:
            all_paths: A list of tuples, where each tuple contains a file path and its class name.

        Returns:
            A dictionary with keys 'train', 'val', and 'test', each containing a
            list of file path tuples.
        """
        np.random.shuffle(all_paths)

        n_total = len(all_paths)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)

        splits = {
            'train': all_paths[:n_train],
            'val': all_paths[n_train:n_train + n_val],
            'test': all_paths[n_train + n_val:]
        }

        return splits

    def _process_batch(self, batch_data: List[Tuple[np.ndarray, int, str]],
                       writer: tf.io.TFRecordWriter, split_name: str):
        """Processes a batch of images and writes the results to a TFRecord file.

        For each image in the batch, it detects hands, preprocesses the cropped
        hand image (with augmentation for the training set), and writes the
        resulting features to the TFRecord file.

        Args:
            batch_data: A list of tuples, each containing an image, its integer
                        label, and its string label.
            writer: The TFRecordWriter to write the examples to.
            split_name: The name of the data split (e.g., 'train') being processed.
        """
        for image, label, label_text in batch_data:
            try:
                # Detect hands and get landmarks
                result = self.hand_detector.detect(image)
                if result is None:
                    self.stats[f'{split_name}_no_hands'] += 1
                    continue

                cropped_hand, landmarks = result

                # Preprocess image
                augment = split_name == 'train'  # Only augment training data
                processed_image = self.image_processor.preprocess(cropped_hand, augment=augment)

                # Create TFRecord example
                example = TFRecordWriter.create_example(
                    processed_image, landmarks, label, label_text
                )

                # Write to TFRecord
                writer.write(example.SerializeToString())
                self.stats[f'{split_name}_processed'] += 1

            except Exception as e:
                self.logger.warning(f"Error processing sample: {e}")
                self.stats[f'{split_name}_errors'] += 1

    def _process_images(self, file_paths: List[Tuple[str, str]],
                        class_mapping: Dict[str, int], split_name: str):
        """Processes a list of image files and saves them to a TFRecord file.

        It reads images, groups them into batches, and uses `_process_batch`
        to handle the processing and writing.

        Args:
            file_paths: A list of tuples, each containing an image file path
                        and its class name.
            class_mapping: A dictionary mapping class names to integer labels.
            split_name: The name of the data split.
        """
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tfrecord_path = output_path / f"{split_name}.tfrecord"

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            batch_data = []

            for i, (file_path, class_name) in enumerate(file_paths):
                try:
                    # Load image
                    image = cv2.imread(file_path)
                    if image is None:
                        self.stats[f'{split_name}_load_errors'] += 1
                        continue

                    label = class_mapping[class_name]
                    batch_data.append((image, label, class_name))

                    # Process batch when full
                    if len(batch_data) >= self.config.batch_size:
                        self._process_batch(batch_data, writer, split_name)
                        batch_data = []

                    # Log progress
                    if (i + 1) % 100 == 0:
                        self.logger.info(f"Processed {i + 1}/{len(file_paths)} {split_name} samples")

                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
                    self.stats[f'{split_name}_load_errors'] += 1

            # Process remaining batch
            if batch_data:
                self._process_batch(batch_data, writer, split_name)

    def _process_videos(self, file_paths: List[Tuple[str, str]],
                        class_mapping: Dict[str, int], split_name: str):
        """Processes a list of video files and saves the frames to a TFRecord file.

        It extracts frames from each video, groups them into batches, and uses
        `_process_batch` to handle the processing and writing.

        Args:
            file_paths: A list of tuples, each containing a video file path
                        and its class name.
            class_mapping: A dictionary mapping class names to integer labels.
            split_name: The name of the data split.
        """
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        tfrecord_path = output_path / f"{split_name}.tfrecord"

        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            batch_data = []
            total_videos = len(file_paths)

            for video_idx, (video_path, class_name) in enumerate(file_paths):
                try:
                    label = class_mapping[class_name]

                    # Extract frames from video
                    for frame in self.video_processor.extract_frames(video_path):
                        batch_data.append((frame, label, class_name))

                        # Process batch when full
                        if len(batch_data) >= self.config.batch_size:
                            self._process_batch(batch_data, writer, split_name)
                            batch_data = []

                    # Log progress
                    if (video_idx + 1) % 10 == 0:
                        self.logger.info(f"Processed {video_idx + 1}/{total_videos} {split_name} videos")

                except Exception as e:
                    self.logger.warning(f"Error processing video {video_path}: {e}")
                    self.stats[f'{split_name}_video_errors'] += 1

            # Process remaining batch
            if batch_data:
                self._process_batch(batch_data, writer, split_name)

    def preprocess_dataset(self):
        """Executes the main preprocessing pipeline.

        This method coordinates the entire process from detecting the dataset type
        to processing all data splits and saving the final artifacts.
        """
        start_time = time.time()

        self.logger.info("Starting sign language dataset preprocessing...")
        self.logger.info(f"Dataset path: {self.config.dataset_path}")
        self.logger.info(f"Output path: {self.config.output_path}")

        # Detect dataset type
        dataset_type = DatasetDetector.detect_dataset_type(self.config.dataset_path)
        self.logger.info(f"Detected dataset type: {dataset_type}")

        # Get class mapping
        class_mapping = self._get_class_mapping(self.config.dataset_path)
        self.logger.info(f"Found {len(class_mapping)} classes: {list(class_mapping.keys())}")

        # Collect all file paths
        dataset_path = Path(self.config.dataset_path)
        all_paths = []

        if dataset_type == 'images':
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in image_exts:
                            all_paths.append((str(img_file), class_name))
        else:  # videos
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for video_file in class_dir.iterdir():
                        if video_file.suffix.lower() in video_exts:
                            all_paths.append((str(video_file), class_name))

        self.logger.info(f"Found {len(all_paths)} total files")

        # Split data
        data_splits = self._split_data_paths(all_paths)

        # Process each split
        for split_name, file_paths in data_splits.items():
            self.logger.info(f"Processing {split_name} split ({len(file_paths)} files)...")

            if dataset_type == 'images':
                self._process_images(file_paths, class_mapping, split_name)
            else:
                self._process_videos(file_paths, class_mapping, split_name)

        # Save metadata
        self._save_metadata(class_mapping, data_splits)

        # Log final statistics
        self._log_final_stats(start_time)

    def _save_metadata(self, class_mapping: Dict[str, int], data_splits: Dict[str, List]):
        """Saves metadata about the preprocessing run to a JSON file.

        The metadata includes dataset info, class mappings, data split counts,
        and processing statistics.

        Args:
            class_mapping: The mapping from class names to integer labels.
            data_splits: A dictionary containing the file paths for each data split.
        """
        # Count samples per class per split
        class_counts = defaultdict(lambda: defaultdict(int))

        for split_name, file_paths in data_splits.items():
            for _, class_name in file_paths:
                class_counts[class_name][split_name] += 1

        metadata = {
            'dataset_info': {
                'total_classes': len(class_mapping),
                'image_size': self.config.img_size,
                'frame_rate': self.config.frame_rate,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'classes': dict(class_mapping),
            'class_counts': dict(class_counts),
            'splits': {
                split: len(paths) for split, paths in data_splits.items()
            },
            'processing_stats': dict(self.stats)
        }

        metadata_path = Path(self.config.output_path) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to: {metadata_path}")

    def _log_final_stats(self, start_time: float):
        """Logs a summary of the preprocessing statistics to the console.

        Args:
            start_time: The time at which the preprocessing started, used to
                        calculate the total elapsed time.
        """
        elapsed = time.time() - start_time

        self.logger.info("=" * 50)
        self.logger.info("PREPROCESSING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {elapsed:.2f} seconds")

        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the preprocessor script.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sign Language Dataset Preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process ASL alphabet dataset (static images)
  python preprocessor.py --dataset-path ./asl_alphabet --output-path ./processed_asl

  # Process Ethiopian sign language words (videos)
  python preprocessor.py --dataset-path ./esl_words --output-path ./processed_esl --frame-rate 15.0
        """
    )

    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to input dataset folder')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to output TFRecord folder')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Target image size (default: 224)')
    parser.add_argument('--frame-rate', type=float, default=10.0,
                        help='Frame extraction rate for videos (default: 10.0)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--rotation-range', type=float, default=15.0,
                        help='Rotation range for augmentation (default: 15.0)')
    parser.add_argument('--brightness-range', type=float, default=0.2,
                        help='Brightness adjustment range (default: 0.2)')

    return parser.parse_args()


def main():
    """The main entry point for the preprocessor script.

    Parses command-line arguments, sets up the configuration, and runs the
    SignLanguagePreprocessor.
    """
    args = parse_args()

    # Validate splits sum to 1.0
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")

    # Create configuration
    config = PreprocessConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        img_size=args.img_size,
        frame_rate=args.frame_rate,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        augmentation=not args.no_augmentation,
        rotation_range=args.rotation_range,
        brightness_range=args.brightness_range
    )

    # Run preprocessor
    preprocessor = SignLanguagePreprocessor(config)

    try:
        preprocessor.preprocess_dataset()
        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Output saved to: {args.output_path}")
        print("📊 Check metadata.json for detailed statistics")

    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        raise

    finally:
        # Cleanup MediaPipe resources
        del preprocessor.hand_detector


if __name__ == "__main__":
    main()