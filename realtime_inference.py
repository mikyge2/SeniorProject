#!/usr/bin/env python3
"""
Real-time ASL inference script using TensorFlow Lite model with dual inputs:
- Hand landmarks (42 float values from 21 x,y coordinates)
- RGB image (224x224x3 normalized to [0,1])

Compatible with Python 3.12, TensorFlow Lite, OpenCV, and MediaPipe.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ASLRealTimeInference:
    """Real-time ASL inference using TensorFlow Lite model with MediaPipe hand detection."""

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0):
        """
        Initialize the ASL inference system.

        Args:
            model_path: Path to the TensorFlow Lite model file
            metadata_path: Path to the metadata JSON file containing class mappings
            camera_index: Camera device index (default: 0)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.camera_index = camera_index

        # Initialize components
        self.interpreter = None
        self.class_mapping = {}
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        self.cap = None

        # Model input/output details
        self.input_details = None
        self.output_details = None

        # Constants
        self.IMAGE_SIZE = (224, 224)
        self.NUM_LANDMARKS = 21
        self.LANDMARK_FEATURES = 42  # 21 landmarks × 2 coordinates (x, y)

    def load_model(self) -> None:
        """Load and initialize the TensorFlow Lite model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading TensorFlow Lite model from {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            logger.info("Model loaded successfully")
            logger.info(f"Input details: {self.input_details}")
            logger.info(f"Output details: {self.output_details}")

            # Verify we have the expected inputs
            if len(self.input_details) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(self.input_details)}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def load_metadata(self) -> None:
        """Load class mapping from metadata JSON file."""
        try:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Extract class mapping (assuming it's stored as index -> class_name)
            if 'class_mapping' in metadata:
                self.class_mapping = {int(k): v for k, v in metadata['class_mapping'].items()}
            elif 'classes' in metadata:
                self.class_mapping = {i: cls for i, cls in enumerate(metadata['classes'])}
            else:
                # Try to infer from the structure
                self.class_mapping = {int(k): v for k, v in metadata.items() if k.isdigit()}

            if not self.class_mapping:
                raise ValueError("No valid class mapping found in metadata")

            logger.info(f"Loaded {len(self.class_mapping)} classes: {list(self.class_mapping.values())}")

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            sys.exit(1)

    def initialize_mediapipe(self) -> None:
        """Initialize MediaPipe Hands solution."""
        try:
            logger.info("Initializing MediaPipe Hands")
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils

            # Initialize hands detection with live stream mode
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,  # Live stream mode
                max_num_hands=1,  # Maximum 1 hand
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

            logger.info("MediaPipe Hands initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing MediaPipe: {e}")
            sys.exit(1)

    def initialize_camera(self) -> None:
        """Initialize camera capture."""
        try:
            logger.info(f"Initializing camera (index: {self.camera_index})")
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera with index {self.camera_index}")

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Camera initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            sys.exit(1)

    def extract_hand_landmarks(self, results) -> Optional[np.ndarray]:
        """
        Extract hand landmarks and convert to flat array.

        Args:
            results: MediaPipe detection results

        Returns:
            Flattened array of shape (42,) containing x,y coordinates of 21 landmarks,
            or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None

        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract x, y coordinates (normalized to [0, 1])
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])

        return np.array(landmarks, dtype=np.float32)

    def crop_hand_region(self, frame: np.ndarray, results) -> Optional[np.ndarray]:
        """
        Crop the hand region from the frame based on hand landmarks.

        Args:
            frame: Input frame
            results: MediaPipe detection results

        Returns:
            Cropped hand region resized to (224, 224, 3) or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]

        # Get bounding box of hand landmarks
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding around hand region
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            return None

        # Resize to model input size
        hand_resized = cv2.resize(hand_crop, self.IMAGE_SIZE)

        return hand_resized

    def preprocess_image(self, image: Optional[np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image or None

        Returns:
            Preprocessed image array of shape (1, 224, 224, 3) normalized to [0, 1]
        """
        if image is None:
            # Return black image if no hand detected
            processed_image = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            # Convert BGR to RGB
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            processed_image = processed_image.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(processed_image, axis=0)

    def preprocess_landmarks(self, landmarks: Optional[np.ndarray]) -> np.ndarray:
        """
        Preprocess landmarks for model input.

        Args:
            landmarks: Flattened landmarks array or None

        Returns:
            Preprocessed landmarks array of shape (1, 42)
        """
        if landmarks is None:
            # Return zeros if no hand detected
            processed_landmarks = np.zeros(self.LANDMARK_FEATURES, dtype=np.float32)
        else:
            processed_landmarks = landmarks

        # Add batch dimension
        return np.expand_dims(processed_landmarks, axis=0)

    def predict(self, image_input: np.ndarray, landmarks_input: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on the model.

        Args:
            image_input: Preprocessed image input
            landmarks_input: Preprocessed landmarks input

        Returns:
            Tuple of (predicted_class, confidence)
        """
        try:
            # Set input tensors
            # Find the correct input indices by name
            landmarks_idx = None
            image_idx = None

            for i, input_detail in enumerate(self.input_details):
                if 'landmarks' in input_detail['name']:
                    landmarks_idx = input_detail['index']
                elif 'image' in input_detail['name']:
                    image_idx = input_detail['index']

            if landmarks_idx is None or image_idx is None:
                # Fall back to index-based assignment
                landmarks_idx = self.input_details[0]['index']
                image_idx = self.input_details[1]['index']

            self.interpreter.set_tensor(landmarks_idx, landmarks_input)
            self.interpreter.set_tensor(image_idx, image_input)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Get prediction
            predicted_class_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_class_idx])

            # Map to class name
            predicted_class = self.class_mapping.get(predicted_class_idx, f"Class_{predicted_class_idx}")

            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def draw_prediction(self, frame: np.ndarray, prediction: str, confidence: float,
                        results) -> np.ndarray:
        """
        Draw prediction and hand landmarks on frame.

        Args:
            frame: Input frame
            prediction: Predicted class
            confidence: Prediction confidence
            results: MediaPipe detection results

        Returns:
            Frame with overlaid information
        """
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Draw prediction text
        text = f"Prediction: {prediction}"
        conf_text = f"Confidence: {confidence:.2f}"

        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

    def run(self) -> None:
        """Run the real-time inference loop."""
        logger.info("Starting real-time ASL inference...")
        logger.info("Press 'q' to quit")

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                results = self.hands.process(rgb_frame)

                # Extract landmarks and crop hand region
                landmarks = self.extract_hand_landmarks(results)
                hand_crop = self.crop_hand_region(frame, results)

                # Preprocess inputs
                image_input = self.preprocess_image(hand_crop)
                landmarks_input = self.preprocess_landmarks(landmarks)

                # Run prediction
                prediction, confidence = self.predict(image_input, landmarks_input)

                # Draw results on frame
                frame = self.draw_prediction(frame, prediction, confidence, results)

                # Display frame
                cv2.imshow('ASL Real-time Inference', frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        if self.cap is not None:
            self.cap.release()

        if self.hands is not None:
            self.hands.close()

        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Real-time ASL inference using TensorFlow Lite model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='export/asl_model.tflite',
        help='Path to the TensorFlow Lite model file'
    )

    parser.add_argument(
        '--metadata',
        type=str,
        default='processed_asl/metadata.json',
        help='Path to the metadata JSON file containing class mappings'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index'
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Create and run inference system
    try:
        inference_system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera
        )

        # Initialize all components
        inference_system.load_model()
        inference_system.load_metadata()
        inference_system.initialize_mediapipe()
        inference_system.initialize_camera()

        # Run inference
        inference_system.run()

    except Exception as e:
        logger.error(f"Failed to initialize inference system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()