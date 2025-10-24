import tensorflow as tf
import hashlib
import sys
from pathlib import Path

def tfrecord_hashes(tfrecord_path):
    """Return a set of SHA1 hashes of the raw image bytes in a TFRecord."""
    hashes = set()
    for record in tf.data.TFRecordDataset(str(tfrecord_path)):
        ex = tf.train.Example()
        ex.ParseFromString(record.numpy())
        img_bytes = ex.features.feature["image/encoded"].bytes_list.value[0]
        hashes.add(hashlib.sha1(img_bytes).hexdigest())
    return hashes

def main(train_path, val_path, test_path):
    train_hashes = tfrecord_hashes(train_path)
    val_hashes   = tfrecord_hashes(val_path)
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
