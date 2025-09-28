#!/usr/bin/env python3
"""
Single Image Inference for Hand Gesture Recognition.

This script performs hand gesture recognition on a single static image using a
pre-trained TensorFlow Lite model. It demonstrates the complete inference
pipeline for one image:
1.  Loads the image from the specified path.
2.  Uses MediaPipe to detect a hand and extract its 2D landmarks.
3.  Preprocesses both the image and the landmarks to match the model's input
    requirements.
4.  Loads the TFLite model.
5.  Runs inference using both the image and landmark data.
6.  Prints the predicted class label and the model's confidence score.

Usage:
    python predictImage.py --image <path_to_image> --model <path_to_model.tflite> --metadata <path_to_metadata.json>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


def load_and_preprocess_image(image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Loads an image, resizes it, and preprocesses it for model inference.

    Args:
        image_path: The path to the input image file.
        target_size: A tuple (height, width) for resizing the image.

    Returns:
        A preprocessed image as a numpy array with shape (1, height, width, 3)
        and values normalized to [0, 1], or None if an error occurs.
    """
    try:
        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None


def extract_hand_landmarks(image_path: Path) -> np.ndarray:
    """Extracts 2D hand landmarks from an image using MediaPipe Hands.

    Args:
        image_path: The path to the input image file.

    Returns:
        A numpy array of shape (42,) containing the flattened x, y coordinates
        of the 21 hand landmarks. Returns a zero array if no hand is detected.
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise IOError(f"Could not read image from {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [lm for landmark in hand_landmarks.landmark for lm in (landmark.x, landmark.y)]
                return np.array(landmarks, dtype=np.float32)
            else:
                print("Warning: No hand detected in the image. Using zero landmarks.")
                return np.zeros(42, dtype=np.float32)
        except Exception as e:
            print(f"Error extracting hand landmarks: {e}")
            return np.zeros(42, dtype=np.float32)


def load_tflite_model(model_path: Path) -> Optional[tf.lite.Interpreter]:
    """Loads a TensorFlow Lite model from a file.

    Args:
        model_path: The path to the `.tflite` model file.

    Returns:
        An initialized TensorFlow Lite interpreter, or None if loading fails.
    """
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None


def run_inference(
    interpreter: tf.lite.Interpreter,
    image_input: np.ndarray,
    landmarks_input: np.ndarray,
) -> Optional[np.ndarray]:
    """Runs inference on a TFLite model that requires image and landmark inputs.

    This function dynamically determines the correct input tensor indices for
    the image and landmarks based on their expected shapes.

    Args:
        interpreter: The initialized TFLite interpreter.
        image_input: The preprocessed image tensor (shape: 1, H, W, 3).
        landmarks_input: The landmark data tensor (shape: 42).

    Returns:
        The raw prediction output from the model, or None if an error occurs.
    """
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 2:
            raise ValueError(f"Model expects 2 inputs, but found {len(input_details)}")

        # Identify input tensor indices by shape
        image_idx = next(i for i, d in enumerate(input_details) if tuple(d['shape']) == (1, 224, 224, 3))
        landmarks_idx = next(i for i, d in enumerate(input_details) if tuple(d['shape']) == (1, 42))

        # Set tensors
        interpreter.set_tensor(input_details[image_idx]['index'], image_input)
        interpreter.set_tensor(input_details[landmarks_idx]['index'], np.expand_dims(landmarks_input, axis=0))

        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except (StopIteration, ValueError) as e:
        print(f"Error: Model input signature mismatch. Could not find expected input shapes. Details: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None


def main():
    """Orchestrates the single-image prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Run inference on a single image for hand gesture recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    parser.add_argument("--model", type=Path, default=Path("export/asl_model.tflite"), help="Path to the TFLite model.")
    parser.add_argument("--metadata", type=Path, default=Path("processed_asl/metadata.json"), help="Path to the metadata JSON file.")
    args = parser.parse_args()

    print(f"Loading metadata from: {args.metadata}")
    with open(args.metadata) as f:
        metadata = json.load(f)
    # Create a mapping from integer index to class name string
    class_labels = {int(k): v for k, v in metadata["classes"].items()}

    print("-" * 50)
    print(f"Processing image: {args.image}")

    image_input = load_and_preprocess_image(args.image)
    if image_input is None:
        sys.exit(1)
    print(f"Image preprocessed successfully. Shape: {image_input.shape}")

    landmarks_input = extract_hand_landmarks(args.image)
    print(f"Hand landmarks extracted. Shape: {landmarks_input.shape}")

    print(f"Loading model: {args.model}")
    interpreter = load_tflite_model(args.model)
    if interpreter is None:
        sys.exit(1)
    print("TFLite model loaded successfully.")

    print("Running inference...")
    predictions = run_inference(interpreter, image_input, landmarks_input)
    if predictions is None:
        sys.exit(1)
    print("Inference complete.")

    # Process and display results
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = class_labels.get(predicted_class_idx, f"Unknown({predicted_class_idx})")
    confidence = predictions[0][predicted_class_idx]

    print("-" * 50)
    print("INFERENCE RESULTS")
    print("-" * 50)
    print(f"Predicted ASL Letter: '{predicted_label}'")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()