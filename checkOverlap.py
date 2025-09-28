"""
Dataset Overlap Checker for TFRecords.

This script verifies the integrity of dataset splits (training, validation, test)
by checking for data leakage between them. It computes the SHA1 hash of each
raw image within the TFRecord files and reports the number of overlapping images
between any two splits.

This is a crucial step to ensure that the model is evaluated on data it has
never seen during training, providing a true measure of its generalization
performance.

Usage:
    python check_overlap.py <path_to_train.tfrecord> <path_to_val.tfrecord> <path_to_test.tfrecord>
"""

import tensorflow as tf
import hashlib
import sys
from pathlib import Path
from typing import Set


def tfrecord_hashes(tfrecord_path: Path) -> Set[str]:
    """Computes a set of unique SHA1 hashes for all images in a TFRecord file.

    Args:
        tfrecord_path: The path to the `.tfrecord` file.

    Returns:
        A set of unique SHA1 hash strings, where each hash corresponds to an
        image in the TFRecord file.
    """
    hashes = set()
    for record in tf.data.TFRecordDataset(str(tfrecord_path)):
        ex = tf.train.Example()
        ex.ParseFromString(record.numpy())
        img_bytes = ex.features.feature["image/encoded"].bytes_list.value[0]
        hashes.add(hashlib.sha1(img_bytes).hexdigest())
    return hashes


def main(train_path: Path, val_path: Path, test_path: Path):
    """Main function to check and report overlaps between dataset splits.

    Args:
        train_path: Path to the training TFRecord file.
        val_path: Path to the validation TFRecord file.
        test_path: Path to the test TFRecord file.
    """
    print("Computing hashes for training set...")
    train_hashes = tfrecord_hashes(train_path)
    print("Computing hashes for validation set...")
    val_hashes   = tfrecord_hashes(val_path)
    print("Computing hashes for test set...")
    test_hashes  = tfrecord_hashes(test_path)

    dup_train_val  = train_hashes & val_hashes
    dup_train_test = train_hashes & test_hashes
    dup_val_test   = val_hashes  & test_hashes

    print(f"Train-Val overlaps:  {len(dup_train_val)}")
    print(f"Train-Test overlaps: {len(dup_train_test)}")
    print(f"Val-Test overlaps:   {len(dup_val_test)}")

    if dup_train_val or dup_train_test or dup_val_test:
        print("\nSample of duplicate hashes:")
        print(list(dup_train_val | dup_train_test | dup_val_test)[:10])
    else:
        print("\nNo overlaps detected ✅")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python check_overlap.py train.tfrecord val.tfrecord test.tfrecord")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
