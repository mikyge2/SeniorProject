import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- TFRecord Parsing ---------- #
def parse_tfrecord_fn(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/landmarks": tf.io.FixedLenFeature([42], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "label_text": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feature_description)

    # Decode and scale image
    image = tf.image.decode_jpeg(ex["image/encoded"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0–1
    landmarks = ex["image/landmarks"]

    return (image, landmarks), ex["label"]

def load_dataset(tfrecord_path, img_size, batch_size):
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Resize image, keep landmarks unchanged
    dataset = dataset.map(
        lambda x, y: ((tf.image.resize(x[0], [img_size, img_size]), x[1]), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------- Main ---------- #
def main(args):
    # Load class names from metadata
    with open(args.metadata, "r") as f:
        meta = json.load(f)
    class_names = meta["classes"]

    # Prepare dataset
    ds = load_dataset(args.tfrecord, args.img_size, args.batch_size)

    # Load trained model
    model = tf.keras.models.load_model(args.model)

    # Predict
    y_true, y_pred = [], []
    for (img_batch, lm_batch), labels in ds:
        preds = model.predict([img_batch, lm_batch], verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
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
