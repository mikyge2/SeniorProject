"""
Enhanced Real-time ASL inference script with finger-spelling detection and speech output.
Adds word assembly, misclassification handling, and text-to-speech capabilities.

Compatible with Python 3.12, TensorFlow Lite, OpenCV, MediaPipe, and pyttsx3.
"""

import argparse
import json
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, Counter
import threading
import queue

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WordTracker:
    """Handles word assembly from letter predictions with misclassification filtering and word prediction."""

    def __init__(self,
                 window_size: int = 15,
                 confidence_threshold: float = 0.7,
                 pause_threshold: float = 3.0,
                 min_letter_duration: float = 2.0):
        """
        Initialize word tracker with slower, more deliberate letter detection.

        Args:
            window_size: Size of sliding window for majority vote
            confidence_threshold: Minimum confidence to consider a prediction
            pause_threshold: Seconds of pause to finalize word
            min_letter_duration: Minimum duration to hold a letter before accepting (2 seconds)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.pause_threshold = pause_threshold
        self.min_letter_duration = min_letter_duration

        # Tracking state
        self.prediction_buffer = deque(maxlen=window_size)
        self.current_word = ""
        self.current_letter = ""
        self.current_letter_start = time.time()
        self.current_letter_confidence = 0.0
        self.last_prediction_time = time.time()
        self.word_finalized = False

        # Letter stability tracking
        self.letter_hold_progress = 0.0  # Progress from 0.0 to 1.0

        # Word prediction
        self.word_suggestions = []
        self.load_word_dictionary()

        # Statistics
        self.total_words = 0
        self.recognized_words = []

    def load_word_dictionary(self):
        """Load a simple English word dictionary for predictions."""
        # Basic English words that can be finger-spelled
        self.common_words = [
            # Common short words
            "hello", "world", "yes", "no", "good", "bad", "help", "stop", "go", "come",
            "please", "thank", "you", "me", "my", "your", "his", "her", "and", "or",
            "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should", "may",

            # Common longer words
            "water", "food", "home", "work", "school", "family", "friend", "love",
            "happy", "sad", "angry", "tired", "hungry", "thirsty", "cold", "hot",
            "big", "small", "fast", "slow", "new", "old", "easy", "hard", "right", "wrong",
            "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",

            # Numbers as words
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "hundred",

            # Colors
            "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink",

            # Actions
            "eat", "drink", "sleep", "walk", "run", "sit", "stand", "look", "see", "hear",
            "talk", "speak", "read", "write", "learn", "teach", "play", "work", "rest",

            # Body parts
            "head", "face", "eye", "ear", "nose", "mouth", "hand", "finger", "arm", "leg", "foot",

            # Places
            "house", "room", "kitchen", "bathroom", "bedroom", "office", "store", "hospital",
            "school", "church", "park", "street", "city", "country"
        ]

        # Sort by length for better matching
        self.common_words.sort(key=len)

    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """Get word suggestions based on partial input."""
        if not partial_word:
            return []

        partial_lower = partial_word.lower()
        suggestions = []

        # Find words that start with the partial word
        for word in self.common_words:
            if word.startswith(partial_lower) and word != partial_lower:
                suggestions.append(word)
                if len(suggestions) >= max_suggestions:
                    break

        return suggestions

    def add_prediction(self, letter: str, confidence: float) -> Tuple[str, bool, float]:
        """
        Add a new letter prediction to the tracking system.

        Args:
            letter: Predicted letter
            confidence: Prediction confidence

        Returns:
            Tuple of (current_word, word_was_finalized, letter_hold_progress)
        """
        current_time = time.time()
        word_finalized = False

        # Check for pause (no confident predictions)
        if confidence < self.confidence_threshold:
            if current_time - self.last_prediction_time > self.pause_threshold:
                if self.current_word and not self.word_finalized:
                    word_finalized = True
                    self.word_finalized = True
                    self.recognized_words.append(self.current_word)
                    self.total_words += 1

            # Reset letter tracking if confidence is too low
            self.letter_hold_progress = 0.0
            return self.current_word, word_finalized, self.letter_hold_progress

        # Update last prediction time for confident predictions
        self.last_prediction_time = current_time
        self.word_finalized = False

        # Add to buffer
        self.prediction_buffer.append((letter, confidence, current_time))

        # Get most stable letter in window
        stable_letter = self._get_stable_letter()

        if stable_letter == self.current_letter and stable_letter:
            # Same letter - check how long we've held it
            hold_duration = current_time - self.current_letter_start
            self.letter_hold_progress = min(1.0, hold_duration / self.min_letter_duration)

            # Accept letter if held long enough
            if hold_duration >= self.min_letter_duration and self.letter_hold_progress >= 1.0:
                if stable_letter != "SPACE":  # Handle space gesture if exists
                    self.current_word += stable_letter.lower()
                    logger.info(f"Letter '{stable_letter}' added to word: '{self.current_word}'")
                else:
                    # Space gesture - finalize current word
                    if self.current_word:
                        word_finalized = True
                        self.word_finalized = True
                        self.recognized_words.append(self.current_word)
                        self.total_words += 1

                # Reset for next letter
                self.current_letter = ""
                self.current_letter_start = current_time
                self.letter_hold_progress = 0.0

        elif stable_letter != self.current_letter:
            # Letter changed - reset tracking
            self.current_letter = stable_letter
            self.current_letter_start = current_time
            self.letter_hold_progress = 0.0
            logger.info(f"Letter changed to: '{stable_letter}'")

        # Update word suggestions
        self.word_suggestions = self.get_word_suggestions(self.current_word)

        return self.current_word, word_finalized, self.letter_hold_progress

    def _get_stable_letter(self) -> str:
        """Get the most stable letter from the prediction buffer."""
        if not self.prediction_buffer:
            return ""

        # Weight predictions by confidence and recency
        letter_scores = Counter()
        current_time = time.time()

        for letter, confidence, timestamp in self.prediction_buffer:
            # Recent predictions get higher weight
            time_weight = max(0.1, 1.0 - (current_time - timestamp) / 3.0)
            score = confidence * time_weight
            letter_scores[letter] += score

        if letter_scores:
            return letter_scores.most_common(1)[0][0]
        return ""

    def select_word_suggestion(self, suggestion_index: int) -> Optional[str]:
        """Select a word suggestion and finalize it."""
        if 0 <= suggestion_index < len(self.word_suggestions):
            selected_word = self.word_suggestions[suggestion_index]
            self.current_word = selected_word
            self.recognized_words.append(selected_word)
            self.total_words += 1
            logger.info(f"Word suggestion selected: '{selected_word}'")
            return selected_word
        return None

    def reset_word(self):
        """Reset current word (for manual reset)."""
        self.current_word = ""
        self.word_finalized = False
        self.current_letter = ""
        self.current_letter_start = time.time()
        self.letter_hold_progress = 0.0
        self.word_suggestions = []

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            "total_words": self.total_words,
            "current_word": self.current_word,
            "current_letter": self.current_letter,
            "letter_progress": self.letter_hold_progress,
            "word_suggestions": self.word_suggestions,
            "recent_words": self.recognized_words[-10:],  # Last 10 words
            "buffer_size": len(self.prediction_buffer)
        }


class SimpleTextToSpeechEngine:
    """Simple TTS engine using system espeak command with better error handling."""

    def __init__(self, rate: int = 150):
        """
        Initialize simple TTS engine.

        Args:
            rate: Speech rate (words per minute)
        """
        self.rate = max(80, min(300, rate))  # Clamp rate for espeak
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.tts_working = False

        # Test if espeak is available
        self._test_espeak()

        # Initialize TTS worker thread
        if self.tts_working:
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def _test_espeak(self):
        """Test if espeak command is available and working."""
        try:
            import subprocess
            import shutil

            # Check if espeak command exists
            if not shutil.which('espeak'):
                logger.error("espeak command not found in PATH")
                self.tts_working = False
                return

            # Test espeak with actual speech
            logger.info("Testing espeak TTS...")
            result = subprocess.run(['espeak', '-s150', '-a50', 'test'],
                                  capture_output=True, timeout=5, text=True)

            if result.returncode == 0:
                self.tts_working = True
                logger.info("✓ espeak TTS engine is working and tested successfully")
            else:
                logger.error(f"✗ espeak test failed with return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"espeak stderr: {result.stderr}")
                self.tts_working = False

        except subprocess.TimeoutExpired:
            logger.error("✗ espeak test timed out")
            self.tts_working = False
        except Exception as e:
            logger.error(f"✗ espeak test failed with exception: {e}")
            self.tts_working = False

    def _tts_worker(self):
        """TTS worker thread using espeak system command."""
        logger.info("✓ Simple TTS worker thread started with espeak")

        while self.is_running:
            try:
                word = self.speech_queue.get(timeout=1.0)
                if word and word.strip():
                    logger.info(f"🔊 Speaking with espeak: '{word}'")
                    try:
                        import subprocess
                        # Use espeak system command directly with better parameters
                        cmd = ['espeak', f'-s{self.rate}', '-a80', '-g5', word.strip()]
                        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)

                        if result.returncode == 0:
                            logger.info(f"✓ Successfully spoke: '{word}'")
                        else:
                            logger.error(f"✗ espeak failed with return code {result.returncode}")
                            if result.stderr:
                                logger.error(f"espeak stderr: {result.stderr}")

                    except subprocess.TimeoutExpired:
                        logger.error(f"✗ espeak timed out speaking '{word}'")
                    except Exception as e:
                        logger.error(f"✗ Error speaking '{word}' with espeak: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    def speak(self, text: str):
        """Queue text for speech."""
        if text and text.strip() and self.is_running and self.tts_working:
            try:
                clean_text = text.strip()
                self.speech_queue.put_nowait(clean_text)
                logger.info(f"📝 Queued for espeak speech: '{clean_text}'")
                return True
            except queue.Full:
                logger.warning("Speech queue full, dropping speech request")
                return False
        else:
            if not self.tts_working:
                logger.error("Cannot speak - TTS not working")
            return False

    def is_working(self) -> bool:
        """Check if TTS is working properly."""
        return self.tts_working

    def stop(self):
        """Stop the TTS engine."""
        logger.info("Stopping TTS engine...")
        self.is_running = False
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)


class ASLRealTimeInference:
    """Enhanced Real-time ASL inference with word detection and speech output."""

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0,
                 enable_speech: bool = True):
        """
        Initialize the enhanced ASL inference system.

        Args:
            model_path: Path to the TensorFlow Lite model file
            metadata_path: Path to the metadata JSON file containing class mappings
            camera_index: Camera device index (default: 0)
            enable_speech: Enable text-to-speech output
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.camera_index = camera_index
        self.enable_speech = enable_speech

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

        # Enhanced components
        self.word_tracker = WordTracker(
            window_size=15,
            confidence_threshold=0.65,
            pause_threshold=3.0,
            min_letter_duration=2.0  # 2 seconds to hold each letter
        )
        self.tts_engine = None
        if enable_speech:
            self.tts_engine = SimpleTextToSpeechEngine()
            # Give TTS time to initialize
            import time
            time.sleep(1)

        # Constants
        self.IMAGE_SIZE = (224, 224)
        self.NUM_LANDMARKS = 21
        self.LANDMARK_FEATURES = 42  # 21 landmarks × 2 coordinates (x, y)

        # Display state
        self.last_spoken_word = ""
        self.word_flash_time = 0
        self.flash_duration = 3.0  # Show spoken word for 3 seconds

        # Selection feedback
        self.selected_suggestion_text = ""
        self.selection_flash_time = 0
        self.selection_flash_duration = 2.0

        # Window size for better UI
        self.window_width = 1200
        self.window_height = 800

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
        """Initialize camera capture with larger window."""
        try:
            logger.info(f"Initializing camera (index: {self.camera_index})")
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera with index {self.camera_index}")

            # Set camera properties for better performance and larger display
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

    def draw_enhanced_ui(self, frame: np.ndarray, prediction: str, confidence: float,
                        results, current_word: str, word_finalized: bool,
                        letter_progress: float, word_suggestions: List[str]) -> np.ndarray:
        """
        Draw enhanced UI with prediction, word assembly, letter progress, and word suggestions.

        Args:
            frame: Input frame
            prediction: Predicted class
            confidence: Prediction confidence
            results: MediaPipe detection results
            current_word: Currently assembled word
            word_finalized: Whether a word was just finalized
            letter_progress: Progress of holding current letter (0.0 to 1.0)
            word_suggestions: List of suggested words

        Returns:
            Frame with enhanced UI overlays
        """
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        h, w = frame.shape[:2]

        # Main info panel (top-left)
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, panel_height), (50, 50, 50), 2)

        # Current prediction
        pred_text = f"Letter: {prediction}"
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, pred_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Current word assembly
        word_color = (255, 255, 0)  # Yellow
        if confidence < self.word_tracker.confidence_threshold:
            word_color = (100, 100, 100)  # Gray when not confident

        word_text = f"Word: {current_word.upper() if current_word else '...'}"
        cv2.putText(frame, word_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, word_color, 2)

        # Word suggestions panel (right side)
        if word_suggestions and current_word:
            suggestions_panel_w = 250
            suggestions_panel_h = 120 + len(word_suggestions) * 30
            suggestions_x = w - suggestions_panel_w - 10
            suggestions_y = 10

            cv2.rectangle(frame, (suggestions_x, suggestions_y),
                         (suggestions_x + suggestions_panel_w, suggestions_y + suggestions_panel_h),
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (suggestions_x, suggestions_y),
                         (suggestions_x + suggestions_panel_w, suggestions_y + suggestions_panel_h),
                         (0, 255, 255), 2)  # Cyan border

            # Title
            cv2.putText(frame, "Word Suggestions:", (suggestions_x + 10, suggestions_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, "Click to select:", (suggestions_x + 10, suggestions_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw suggestions as clickable buttons
            for i, suggestion in enumerate(word_suggestions):
                button_y = suggestions_y + 65 + i * 30
                button_h = 25

                # Button background
                cv2.rectangle(frame, (suggestions_x + 10, button_y),
                             (suggestions_x + suggestions_panel_w - 10, button_y + button_h),
                             (40, 40, 40), -1)
                cv2.rectangle(frame, (suggestions_x + 10, button_y),
                             (suggestions_x + suggestions_panel_w - 10, button_y + button_h),
                             (100, 200, 100), 1)

                # Button text
                button_text = f"{i+1}. {suggestion.upper()}"
                cv2.putText(frame, button_text, (suggestions_x + 15, button_y + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 100), 1)

                # Store button coordinates for click detection
                setattr(self, f'suggestion_{i}_coords',
                       (suggestions_x + 10, button_y, suggestions_x + suggestions_panel_w - 10, button_y + button_h))

        # Word finalization flash
        current_time = time.time()
        if word_finalized:
            self.last_spoken_word = current_word
            self.word_flash_time = current_time

        # Show spoken word with flash effect
        if (self.last_spoken_word and
            current_time - self.word_flash_time < self.flash_duration):

            # Flash effect - alternate between bright colors
            flash_phase = int((current_time - self.word_flash_time) * 3) % 2
            flash_color = (0, 255, 255) if flash_phase else (255, 0, 255)  # Cyan/Magenta

            # Spoken word panel (center)
            spoken_panel_y = h // 3
            cv2.rectangle(frame, (w//4, spoken_panel_y),
                         (3*w//4, spoken_panel_y + 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (w//4, spoken_panel_y),
                         (3*w//4, spoken_panel_y + 80), flash_color, 3)

            spoken_text = f"SPOKEN: {self.last_spoken_word.upper()}"
            text_size = cv2.getTextSize(spoken_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = w//2 - text_size[0]//2
            text_y = spoken_panel_y + 50

            cv2.putText(frame, spoken_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, flash_color, 3)

        # Stats panel (bottom-right)
        stats = self.word_tracker.get_stats()
        stats_panel_w, stats_panel_h = 300, 140
        stats_x = w - stats_panel_w - 10
        stats_y = h - stats_panel_h - 10

        cv2.rectangle(frame, (stats_x, stats_y),
                     (stats_x + stats_panel_w, stats_y + stats_panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (stats_x, stats_y),
                     (stats_x + stats_panel_w, stats_y + stats_panel_h), (50, 50, 50), 2)

        # Stats text
        total_words_text = f"Total Words: {stats['total_words']}"
        buffer_text = f"Buffer: {stats['buffer_size']}/{self.word_tracker.window_size}"

        # TTS status
        tts_status = "TTS: Not Available"
        tts_color = (0, 0, 255)

        if self.tts_engine and self.tts_engine.is_working():
            tts_status = "TTS: espeak working"
            tts_color = (0, 255, 0)
        elif self.tts_engine:
            tts_status = "TTS: Failed"

        cv2.putText(frame, total_words_text, (stats_x + 10, stats_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, buffer_text, (stats_x + 10, stats_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, tts_status, (stats_x + 10, stats_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 1)

        # Controls (bottom-left)
        controls_text = "Press 'r' to reset word, '1'/'2'/'3' for suggestions, 'q' to quit"
        cv2.putText(frame, controls_text, (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        controls_text2 = "Hold each letter for 2 seconds to add it to the word"
        cv2.putText(frame, controls_text2, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return frame

    def run(self) -> None:
        """Run the enhanced real-time inference loop."""
        logger.info("Starting enhanced ASL inference with word detection...")
        logger.info("Press 'r' to reset current word, 'q' to quit")

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

                # Update word tracker with slower, more deliberate detection
                current_word, word_finalized, letter_progress = self.word_tracker.add_prediction(
                    prediction, confidence)

                # Get word suggestions
                stats = self.word_tracker.get_stats()
                word_suggestions = stats.get('word_suggestions', [])

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.word_tracker.reset_word()
                    logger.info("Word reset by user")
                elif key == ord('1'):
                    # Select first suggestion
                    stats = self.word_tracker.get_stats()
                    suggestions = stats.get('word_suggestions', [])
                    if len(suggestions) >= 1:
                        selected_word = self.word_tracker.select_word_suggestion(0)
                        if selected_word:
                            # Visual feedback
                            self.selected_suggestion_text = selected_word
                            self.selection_flash_time = time.time()

                            # Speak the word
                            if self.tts_engine and self.tts_engine.speak(selected_word):
                                logger.info(f"✓ Suggestion 1 selected and queued for speech: '{selected_word}'")
                            else:
                                logger.error(f"✗ Failed to speak suggestion 1: '{selected_word}'")

                            self.word_tracker.reset_word()
                elif key == ord('2'):
                    # Select second suggestion
                    stats = self.word_tracker.get_stats()
                    suggestions = stats.get('word_suggestions', [])
                    if len(suggestions) >= 2:
                        selected_word = self.word_tracker.select_word_suggestion(1)
                        if selected_word:
                            # Visual feedback
                            self.selected_suggestion_text = selected_word
                            self.selection_flash_time = time.time()

                            # Speak the word
                            if self.tts_engine and self.tts_engine.speak(selected_word):
                                logger.info(f"✓ Suggestion 2 selected and queued for speech: '{selected_word}'")
                            else:
                                logger.error(f"✗ Failed to speak suggestion 2: '{selected_word}'")

                            self.word_tracker.reset_word()
                elif key == ord('3'):
                    # Select third suggestion
                    stats = self.word_tracker.get_stats()
                    suggestions = stats.get('word_suggestions', [])
                    if len(suggestions) >= 3:
                        selected_word = self.word_tracker.select_word_suggestion(2)
                        if selected_word:
                            # Visual feedback
                            self.selected_suggestion_text = selected_word
                            self.selection_flash_time = time.time()

                            # Speak the word
                            if self.tts_engine and self.tts_engine.speak(selected_word):
                                logger.info(f"✓ Suggestion 3 selected and queued for speech: '{selected_word}'")
                            else:
                                logger.error(f"✗ Failed to speak suggestion 3: '{selected_word}'")

                            self.word_tracker.reset_word()

                # Display frame
                cv2.imshow('Enhanced ASL Real-time Inference', frame)

                # Handle word finalization
                if word_finalized and current_word and self.tts_engine:
                    if self.tts_engine.speak(current_word):
                        logger.info(f"✓ Word completed and queued for speech: '{current_word}'")
                    else:
                        logger.error(f"✗ Failed to speak completed word: '{current_word}'")
                    self.word_tracker.reset_word()  # Reset for next word

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up resources...")

        if self.tts_engine:
            self.tts_engine.stop()

        if self.cap is not None:
            self.cap.release()

        if self.hands is not None:
            self.hands.close()

        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced real-time ASL inference with word detection and speech output",
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

    parser.add_argument(
        '--no-speech',
        action='store_true',
        help='Disable text-to-speech output'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=15,
        help='Size of sliding window for letter stability (default: 15)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.65,
        help='Minimum confidence threshold for letter acceptance (default: 0.65)'
    )

    parser.add_argument(
        '--pause-threshold',
        type=float,
        default=3.0,
        help='Pause duration (seconds) to finalize word (default: 3.0)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Create and run enhanced inference system
    try:
        inference_system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera,
            enable_speech=not args.no_speech
        )

        # Configure word tracker with slower parameters
        inference_system.word_tracker = WordTracker(
            window_size=args.window_size,
            confidence_threshold=args.confidence_threshold,
            pause_threshold=args.pause_threshold,
            min_letter_duration=2.0  # Always 2 seconds for letter holding
        )

        # Initialize all components
        inference_system.load_model()
        inference_system.load_metadata()
        inference_system.initialize_mediapipe()
        inference_system.initialize_camera()

        # Run enhanced inference
        inference_system.run()

    except Exception as e:
        logger.error(f"Failed to initialize enhanced inference system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()