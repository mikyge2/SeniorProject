"""
Model Evaluation and Confusion Matrix Visualization.

This script evaluates a trained sign language recognition model on a test dataset
provided in TFRecord format. It performs the following steps:
1.  Loads the trained Keras model.
2.  Loads the test dataset and class names from metadata.
3.  Makes predictions on the entire test set.
4.  Generates and prints a detailed classification report (precision, recall, F1-score).
5.  Computes and displays a confusion matrix using seaborn and matplotlib to
    visualize the model's performance across different classes.

Usage:
    python evaluate_confusion.py --model <path_to_model.keras> \
                                 --tfrecord <path_to_test.tfrecord> \
                                 --metadata <path_to_metadata.json>
"""
import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_tfrecord_fn(example: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Parses a single TFRecord example into features and a label.

    Args:
        example: A scalar string tensor representing a serialized `tf.train.Example`.

    Returns:
        A tuple containing:
        - A tuple of features: (image_tensor, landmarks_tensor).
        - The integer label tensor.
    """
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/landmarks": tf.io.FixedLenFeature([42], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "label_text": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_jpeg(ex["image/encoded"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    landmarks = ex["image/landmarks"]
    label = ex["label"]

    return (image, landmarks), label


def load_dataset(tfrecord_path: Path, img_size: int, batch_size: int) -> tf.data.Dataset:
    """Loads and preprocesses the dataset from a TFRecord file.

    Args:
        tfrecord_path: The path to the `.tfrecord` file.
        img_size: The target square size for the images.
        batch_size: The batch size for processing.

    Returns:
        A `tf.data.Dataset` object ready for evaluation.
    """
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y: ((tf.image.resize(x[0], [img_size, img_size]), x[1]), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main(args: argparse.Namespace):
    """Main function to run the model evaluation and plot the confusion matrix.

    Args:
        args: An `argparse.Namespace` object containing the script's arguments.
    """
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    print(f"Loading class names from {args.metadata}...")
    with open(args.metadata, "r") as f:
        meta = json.load(f)
    class_names = list(meta["classes"].values())

    print(f"Loading test dataset from {args.tfrecord}...")
    ds = load_dataset(Path(args.tfrecord), args.img_size, args.batch_size)

    print("Making predictions on the test set...")
    y_true, y_pred = [], []
    for (img_batch, lm_batch), labels in ds:
        # The model expects a list of inputs
        preds = model.predict([img_batch, lm_batch], verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()

    # Save the plot
    output_path = Path("confusion_matrix.png")
    plt.savefig(output_path)
    print(f"\nConfusion matrix plot saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model and plot confusion matrix")
    parser.add_argument("--model", required=True, help="Path to .keras or .h5 model file")
    parser.add_argument("--tfrecord", required=True, help="Path to test TFRecord")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (default 224)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64)")
    args = parser.parse_args()
    main(args)
