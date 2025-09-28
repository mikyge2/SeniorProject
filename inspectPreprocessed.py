"""
TFRecord Preprocessing Inspector.

This script provides a diagnostic tool to inspect and visualize data from a
TFRecord file created by `preprocessor.py`. It is designed to help identify
potential issues in the preprocessed data, such as corrupted images or invalid
landmark data, before starting a lengthy training process.

The script checks for two common problems:
1.  **Low Landmark Variance**: Detects samples where all hand landmarks are
    clustered together, which usually indicates a failure in hand detection.
2.  **Out-of-Range Landmarks**: Checks if landmark coordinates fall outside the
    expected normalized range of [0, 1].

Any samples flagged as problematic are then visualized in a grid, showing the
cropped hand image with the detected landmarks overlaid, allowing for quick
manual verification of the issue.

Usage:
    python inspectPreprocessed.py --tfrecord_path <path_to_your.tfrecord>
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Dict


def parse_tfrecord_fn(example_proto: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parses a single serialized TFRecord example.

    Args:
        example_proto: A scalar string tensor from a TFRecord file.

    Returns:
        A dictionary mapping feature names to their corresponding tensors.
    """
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/landmarks': tf.io.FixedLenFeature([42], tf.float32),  # 21 points x 2
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_text': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def inspect_records(tfrecord_path: Path):
    """
    Inspects a TFRecord file for problematic data and visualizes any issues found.

    Args:
        tfrecord_path: The path to the TFRecord file to inspect.
    """
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord file not found at: {tfrecord_path}")

    print(f"Inspecting TFRecord file: {tfrecord_path}")

    # Load the dataset
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    # Thresholds for detecting potential problems
    MIN_LANDMARK_VARIANCE = 0.001  # Detects if landmarks are all at the same point
    MAX_LANDMARK_VALUE = 1.0       # Landmarks should be normalized between 0 and 1

    problem_samples = []

    # Iterate through each record to check for issues
    print("Analyzing records for potential issues...")
    for i, record in enumerate(parsed_dataset):
        landmarks = record['image/landmarks'].numpy()

        # Condition 1: Check for very low variance in landmarks
        if np.var(landmarks) < MIN_LANDMARK_VARIANCE:
            print(f"Record {i}: Found low landmark variance.")
            problem_samples.append(record)
            continue

        # Condition 2: Check if landmarks are outside the normalized [0, 1] range
        if np.any(landmarks < 0.0) or np.any(landmarks > MAX_LANDMARK_VALUE):
            print(f"Record {i}: Found landmarks out of range.")
            problem_samples.append(record)
            continue

    # Visualize the problematic samples
    if problem_samples:
        print(f"\nFound {len(problem_samples)} potential problem samples. Visualizing up to 25...")
        n_rows = 5
        n_cols = 5
        n_samples_to_show = min(len(problem_samples), n_rows * n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        axes = axes.flatten()

        for idx, record in enumerate(problem_samples[:n_samples_to_show]):
            # Decode the image from byte string
            image_raw = record['image/encoded'].numpy()
            image = tf.io.decode_jpeg(image_raw).numpy()

            # Reshape landmarks for plotting
            landmarks = record['image/landmarks'].numpy().reshape(-1, 2)
            h, w, _ = image.shape

            # Plot the image
            axes[idx].imshow(image)
            axes[idx].axis('off')

            # Overlay landmarks onto the image
            # The coordinates are normalized, so we scale them by image dimensions
            for x, y in landmarks:
                axes[idx].scatter(x * w, y * h, color='red', s=10, edgecolors='white', linewidths=0.5)

            # Display the label as the title
            label_text = record['label_text'].numpy().decode('utf-8')
            axes[idx].set_title(label_text, fontsize=10)

        # Hide any unused subplots
        for i in range(n_samples_to_show, len(axes)):
            axes[i].axis('off')

        plt.suptitle("Problematic Samples Visualization", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("\n✅ No obvious landmark issues detected in the dataset.")


def main():
    """Parses command-line arguments and runs the inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect a TFRecord file for data quality issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tfrecord_path",
        type=str,
        default="./processed_asl/train.tfrecord",
        help="Path to the TFRecord file to inspect."
    )
    args = parser.parse_args()
    inspect_records(Path(args.tfrecord_path))


if __name__ == "__main__":
    main()