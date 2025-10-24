import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Path to TFRecord
tfrecord_path = "./processed_asl/train.tfrecord"

# Feature description to parse TFRecord
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/landmarks': tf.io.FixedLenFeature([42], tf.float32),  # 21 points x 2
    'label': tf.io.FixedLenFeature([], tf.int64),
    'label_text': tf.io.FixedLenFeature([], tf.string)
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# Load TFRecord dataset
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
parsed_dataset = raw_dataset.map(_parse_function)

# Thresholds for detecting potential problems
MIN_LANDMARK_VARIANCE = 0.001  # Landmarks not all same point
MAX_LANDMARK_VALUE = 1.0  # Should be normalized between 0 and 1

problem_samples = []

# Check each record for issues
for record in parsed_dataset:
    landmarks = record['image/landmarks'].numpy()

    # Condition 1: All landmarks are the same
    if np.var(landmarks) < MIN_LANDMARK_VARIANCE:
        problem_samples.append(record)
        continue

    # Condition 2: Landmarks out of normalized range
    if np.any(landmarks < 0.0) or np.any(landmarks > MAX_LANDMARK_VALUE):
        problem_samples.append(record)
        continue

# Visualize problem samples
if problem_samples:
    n = min(len(problem_samples), 25)  # Show up to 25 samples
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()

    for idx, record in enumerate(problem_samples[:n]):
        # Decode image
        image_raw = record['image/encoded'].numpy()
        image = tf.io.decode_jpeg(image_raw).numpy()

        # Landmarks
        landmarks = record['image/landmarks'].numpy().reshape(-1, 2)
        h, w, _ = image.shape

        # Plot image
        axes[idx].imshow(image)
        axes[idx].axis('off')

        # Overlay landmarks
        for x, y in landmarks:
            axes[idx].scatter(x * w, y * h, color='red', s=10)

        # Label
        label_text = record['label_text'].numpy().decode('utf-8')
        axes[idx].set_title(label_text, fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"Found {len(problem_samples)} potential problem samples in the dataset.")
else:
    print("No obvious landmark issues detected in the dataset.")
