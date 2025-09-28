"""
Keras Model to TensorFlow Lite Converter.

This script facilitates the conversion of a trained Keras model (in `.keras` or
`.h5` format) into a TensorFlow Lite (`.tflite`) model, which is optimized for
deployment on mobile and edge devices.

The conversion process involves two main steps:
1.  Exporting the Keras model to TensorFlow's SavedModel format. This is a
    necessary intermediate step that creates a self-contained, serialized
    version of the model including its architecture and weights.
2.  Using the TensorFlow Lite converter to transform the SavedModel into the
    highly optimized `.tflite` flat-buffer format.

Usage:
    python exportModel.py --model_path <path_to_keras_model> --output_path <directory_for_tflite_model>
"""
import argparse
import os
from pathlib import Path
import tensorflow as tf


def export_model(model_path: Path, output_path: Path):
    """Loads a Keras model and converts it to TensorFlow Lite format.

    Args:
        model_path: The path to the input Keras model file (.keras or .h5).
        output_path: The directory where the exported TFLite model will be saved.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Export as TensorFlow SavedModel (required for TFLite)
    # We use a temporary directory for the intermediate SavedModel.
    saved_model_dir = output_path / "temp_saved_model"
    print(f"Exporting to intermediate SavedModel format at: {saved_model_dir}")
    model.export(str(saved_model_dir))
    print("SavedModel export complete.")

    # 2. Convert the SavedModel to TensorFlow Lite
    print("Converting SavedModel to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    print("TFLite conversion complete.")

    # 3. Save the TFLite model to a file
    tflite_model_path = output_path / "asl_model.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"\n✅ TensorFlow Lite model successfully exported to: {tflite_model_path}")

    # 4. Clean up the temporary SavedModel directory
    try:
        import shutil
        shutil.rmtree(saved_model_dir)
        print(f"Cleaned up temporary directory: {saved_model_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {saved_model_dir}. Error: {e}")


def main():
    """Parses command-line arguments and runs the model export process."""
    parser = argparse.ArgumentParser(
        description="Convert a Keras model to TensorFlow Lite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/best_model.keras",
        help="Path to the trained Keras model (.keras or .h5 file)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./export",
        help="Directory to save the exported .tflite model."
    )
    args = parser.parse_args()

    export_model(Path(args.model_path), Path(args.output_path))


if __name__ == "__main__":
    main()