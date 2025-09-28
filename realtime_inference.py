#!/usr/bin/env python3
"""
Real-time American Sign Language (ASL) Inference.

This script captures video from a webcam, uses MediaPipe to detect hand
landmarks, and feeds the landmarks along with the cropped hand image into a
TensorFlow Lite model to predict the signed letter in real-time.

It provides a basic, standalone demonstration of the core computer vision and
inference pipeline.

Usage:
    python realtime_inference.py --model <path_to_model.tflite> \
                                 --metadata <path_to_metadata.json> \
                                 --camera <camera_index>
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASLRealTimeInference:
    """Manages real-time ASL inference using a webcam, MediaPipe, and a TFLite model."""

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0):
        """Initializes the ASL inference system.

        Args:
            model_path: The path to the TensorFlow Lite model file.
            metadata_path: The path to the metadata JSON file containing class mappings.
            camera_index: The index of the camera device to use.
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.camera_index = camera_index

        self.interpreter: Optional[tf.lite.Interpreter] = None
        self.class_mapping: dict[int, str] = {}
        self.mp_hands = mp.solutions.hands
        self.hands: Optional[mp.solutions.hands.Hands] = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap: Optional[cv2.VideoCapture] = None

        self.input_details = None
        self.output_details = None

        self.IMAGE_SIZE = (224, 224)
        self.LANDMARK_FEATURES = 42

    def load_model(self) -> None:
        """Loads the TFLite model and allocates its tensors.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model does not have the expected number of inputs.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading TensorFlow Lite model from {self.model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info("Model loaded successfully.")

        if len(self.input_details) != 2:
            raise ValueError(f"Expected 2 model inputs, but found {len(self.input_details)}")

    def load_metadata(self) -> None:
        """Loads the class mapping from the specified metadata JSON file.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If the metadata file does not contain a valid class mapping.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Handle different possible metadata structures
        if 'classes' in metadata:
            self.class_mapping = {int(k): v for k, v in metadata['classes'].items()}
        else:
            raise ValueError("Metadata file must contain a 'classes' key with class mappings.")

        if not self.class_mapping:
            raise ValueError("No valid class mapping found in metadata.")
        logger.info(f"Loaded {len(self.class_mapping)} classes.")

    def initialize_mediapipe(self) -> None:
        """Initializes the MediaPipe Hands solution for hand tracking."""
        logger.info("Initializing MediaPipe Hands...")
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe Hands initialized.")

    def initialize_camera(self) -> None:
        """Initializes the camera capture device.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        logger.info(f"Initializing camera (index: {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {self.camera_index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("Camera initialized.")

    def extract_hand_landmarks(self, results: Any) -> Optional[np.ndarray]:
        """Extracts and flattens hand landmarks from MediaPipe results.

        Args:
            results: The output from `mediapipe.solutions.hands.process`.

        Returns:
            A flattened numpy array of shape (42,) containing 21 x,y landmark
            coordinates, or None if no hand was detected.
        """
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [lm for landmark in hand_landmarks.landmark for lm in (landmark.x, landmark.y)]
        return np.array(landmarks, dtype=np.float32)

    def crop_hand_region(self, frame: np.ndarray, results: Any) -> Optional[np.ndarray]:
        """Crops the hand region from the frame based on detected landmarks.

        Args:
            frame: The input video frame.
            results: The output from `mediapipe.solutions.hands.process`.

        Returns:
            A cropped image of the hand resized to the model's input size,
            or None if no hand is detected.
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

        padding = 20
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

        if x_max <= x_min or y_max <= y_min:
            return None

        hand_crop = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(hand_crop, self.IMAGE_SIZE)

    def preprocess_image(self, image: Optional[np.ndarray]) -> np.ndarray:
        """Prepares an image for model input.

        Args:
            image: A cropped hand image, or None if no hand was detected.

        Returns:
            A preprocessed image tensor of shape (1, 224, 224, 3), normalized to [0, 1].
        """
        if image is None:
            processed_image = np.zeros((*self.IMAGE_SIZE, 3), dtype=np.float32)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.astype(np.float32) / 255.0
        return np.expand_dims(processed_image, axis=0)

    def preprocess_landmarks(self, landmarks: Optional[np.ndarray]) -> np.ndarray:
        """Prepares landmark data for model input.

        Args:
            landmarks: A flattened array of landmark data, or None.

        Returns:
            A preprocessed landmark tensor of shape (1, 42).
        """
        if landmarks is None:
            processed_landmarks = np.zeros(self.LANDMARK_FEATURES, dtype=np.float32)
        else:
            processed_landmarks = landmarks
        return np.expand_dims(processed_landmarks, axis=0)

    def predict(self, image_input: np.ndarray, landmarks_input: np.ndarray) -> Tuple[str, float]:
        """Runs inference on the model using the provided image and landmark data.

        Args:
            image_input: The preprocessed image tensor.
            landmarks_input: The preprocessed landmarks tensor.

        Returns:
            A tuple containing the predicted class name and the confidence score.
        """
        try:
            image_idx = next(i for i, d in enumerate(self.input_details) if 'image' in d['name'])
            landmarks_idx = next(i for i, d in enumerate(self.input_details) if 'landmarks' in d['name'])

            self.interpreter.set_tensor(self.input_details[image_idx]['index'], image_input)
            self.interpreter.set_tensor(self.input_details[landmarks_idx]['index'], landmarks_input)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_idx])
            predicted_class = self.class_mapping.get(predicted_idx, "Unknown")
            return predicted_class, confidence
        except (StopIteration, ValueError):
            logger.error("Could not find expected 'image' and 'landmarks' inputs in model.")
            return "Error", 0.0
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def draw_prediction(self, frame: np.ndarray, prediction: str, confidence: float, results: Any) -> np.ndarray:
        """Draws the prediction and hand landmarks onto the video frame.

        Args:
            frame: The input video frame.
            prediction: The predicted class name.
            confidence: The prediction confidence score.
            results: The raw results from MediaPipe for drawing landmarks.

        Returns:
            The frame with the prediction and landmarks overlaid.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {prediction}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def run(self) -> None:
        """Runs the main real-time inference loop."""
        logger.info("Starting real-time ASL inference... (Press 'q' to quit)")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame from camera.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                landmarks = self.extract_hand_landmarks(results)
                hand_crop = self.crop_hand_region(frame, results)

                image_input = self.preprocess_image(hand_crop)
                landmarks_input = self.preprocess_landmarks(landmarks)

                prediction, confidence = self.predict(image_input, landmarks_input)
                frame = self.draw_prediction(frame, prediction, confidence, results)

                cv2.imshow('ASL Real-time Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Inference stopped by user.")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Releases all resources."""
        logger.info("Cleaning up resources...")
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete.")


def main():
    """Parses arguments and runs the real-time inference system."""
    parser = argparse.ArgumentParser(
        description="Real-time ASL inference using a TensorFlow Lite model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default='export/asl_model.tflite', help="Path to the TFLite model file.")
    parser.add_argument("--metadata", type=str, default='processed_asl/metadata.json', help="Path to the metadata JSON file.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    args = parser.parse_args()

    try:
        inference_system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera,
        )
        inference_system.load_model()
        inference_system.load_metadata()
        inference_system.initialize_mediapipe()
        inference_system.initialize_camera()
        inference_system.run()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()