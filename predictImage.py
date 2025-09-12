#!/usr/bin/env python3
"""
TensorFlow Lite inference script for hand gesture recognition.
Takes an image and extracts hand landmarks using MediaPipe, then runs inference
on a TFLite model that expects both image and landmark inputs.
"""

import argparse
import sys
import os
from typing import Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for model inference.

    Args:
        image_path: Path to the input image
        target_size: Target size for resizing (width, height)

    Returns:
        Preprocessed image array of shape (1, height, width, 3) or None if error
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target size
        image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and add batch dimension
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def extract_hand_landmarks(image_path: str) -> np.ndarray:
    """
    Extract hand landmarks from an image using MediaPipe Hands.

    Args:
        image_path: Path to the input image

    Returns:
        Array of shape (42,) containing x,y coordinates of 21 landmarks
        or zeros if no hand detected
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return np.zeros(42, dtype=np.float32)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image to find hands
        results = hands.process(image_rgb)

        # Extract landmarks if hand detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            # Get first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extract x, y coordinates for all 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])

            return np.array(landmarks, dtype=np.float32)
        else:
            print("Warning: No hand detected in image, using zero landmarks")
            return np.zeros(42, dtype=np.float32)

    except Exception as e:
        print(f"Error extracting hand landmarks: {e}")
        return np.zeros(42, dtype=np.float32)

    finally:
        hands.close()


def load_tflite_model(model_path: str) -> Optional[tf.lite.Interpreter]:
    """
    Load a TensorFlow Lite model.

    Args:
        model_path: Path to the .tflite model file

    Returns:
        TensorFlow Lite interpreter or None if error
    """
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None

        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        return interpreter

    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None


def run_inference(interpreter: tf.lite.Interpreter, image_input: np.ndarray,
                  landmarks_input: np.ndarray) -> Optional[np.ndarray]:
    """
    Run inference on the TFLite model with image and landmarks inputs.

    Args:
        interpreter: TensorFlow Lite interpreter
        image_input: Preprocessed image array of shape (1, 224, 224, 3)
        landmarks_input: Hand landmarks array of shape (42,)

    Returns:
        Output predictions array or None if error
    """
    try:
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details) != 2:
            print(f"Error: Model expects 2 inputs, but found {len(input_details)}")
            return None

        # Sort input details by index to ensure correct order
        input_details = sorted(input_details, key=lambda x: x['index'])

        print(f"Input 0 shape: {input_details[0]['shape']}")
        print(f"Input 1 shape: {input_details[1]['shape']}")

        # Determine which input is landmarks vs image based on shape
        landmarks_input_idx = None
        image_input_idx = None

        for i, detail in enumerate(input_details):
            shape = detail['shape']
            if len(shape) == 2 and shape[1] == 42:  # landmarks input (batch_size, 42)
                landmarks_input_idx = i
            elif len(shape) == 4 and shape[1] == 224 and shape[2] == 224 and shape[3] == 3:  # image input
                image_input_idx = i

        if landmarks_input_idx is None or image_input_idx is None:
            print("Error: Could not identify landmarks and image inputs from model")
            return None

        # Prepare landmarks input with batch dimension
        landmarks_batch = np.expand_dims(landmarks_input, axis=0)

        # Set input tensors in correct order
        interpreter.set_tensor(input_details[landmarks_input_idx]['index'], landmarks_batch)
        interpreter.set_tensor(input_details[image_input_idx]['index'], image_input)

        print(f"Setting landmarks to input {landmarks_input_idx}: shape {landmarks_batch.shape}")
        print(f"Setting image to input {image_input_idx}: shape {image_input.shape}")

        # Run inference
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        return output_data

    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def main():
    class_labels = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z",
        26: "del",
        27: "nothing",
        28: "space"
    }
    parser = argparse.ArgumentParser(description='Run inference on hand gesture recognition model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='export/asl_model.tflite',
                        help='Path to TFLite model (default: export/asl_model.tflite)')

    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    print(f"Loading model: {args.model}")
    print("-" * 50)

    # Load and preprocess image
    image_input = load_and_preprocess_image(args.image)
    if image_input is None:
        sys.exit(1)

    print(f"Image preprocessed to shape: {image_input.shape}")

    # Extract hand landmarks
    landmarks_input = extract_hand_landmarks(args.image)
    print(f"Hand landmarks extracted: shape {landmarks_input.shape}")

    # Load TFLite model
    interpreter = load_tflite_model(args.model)
    if interpreter is None:
        sys.exit(1)

    print("TFLite model loaded successfully")

    # Run inference
    predictions = run_inference(interpreter, image_input, landmarks_input)
    if predictions is None:
        sys.exit(1)

    # Process results
    print("-" * 50)
    print("INFERENCE RESULTS:")
    print(f"Raw output shape: {predictions.shape}")
    print(f"Raw probabilities: {predictions[0]}")

    # Get predicted class (highest probability)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels.get(predicted_class, f"Unknown({predicted_class})")
    predicted_confidence = predictions[0][predicted_class]

    print(f"Predicted class index: {predicted_class}")
    print(f"Predicted ASL letter: '{predicted_label}'")
    print(f"Confidence: {predicted_confidence:.4f}")
    print(f"Result: Class {predicted_class} = '{predicted_label}' ({predicted_confidence:.4f})")
    print("-" * 50)


if __name__ == "__main__":
    main()