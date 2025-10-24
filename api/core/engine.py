import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import cv2
import numpy as np
from .inference import ASLRealTimeInference, WordTracker, SimpleAmharicTranslator
from ..services.connection_manager import PerformanceMonitor

logger = logging.getLogger(__name__)

class ASLInferenceEngine:
    """Enhanced headless ASL inference engine with advanced features."""

    def __init__(self, model_path: str, metadata_path: str, enable_amharic: bool = True):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_amharic = enable_amharic
        self.start_time = time.time()

        self.inference_system = ASLRealTimeInference(
            model_path=model_path,
            metadata_path=metadata_path,
            camera_index=0,
            enable_speech=True,
            use_google_tts=True,
            show_landmarks=False,
            enable_amharic=enable_amharic
        )

        self._initialize_system()

        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = SimpleAmharicTranslator(use_translation=True)

    def _initialize_system(self):
        """Initialize the inference system components."""
        try:
            self.inference_system.load_model()
            self.inference_system.load_metadata()
            self.inference_system.initialize_mediapipe()
        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            raise

    def process_frame(self, frame_data: bytes, word_tracker: WordTracker,
                     performance_monitor: PerformanceMonitor) -> Dict[str, Any]:
        """Process a single frame with enhanced features."""
        start_time = time.time()

        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image data")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.inference_system.hands.process(rgb_frame)
            landmarks = self.inference_system.extract_hand_landmarks(results)
            hand_crop = self.inference_system.crop_hand_region(frame, results)
            image_input, landmarks_input = self.inference_system.preprocess_inputs(hand_crop, landmarks)
            prediction, confidence = self.inference_system.predict(image_input, landmarks_input)

            is_sign_detected = True
            display_letter = prediction

            if prediction.lower() in ["del", "space"]:
                is_sign_detected = False
                display_letter = ""
                current_word = word_tracker.current_word
                word_finalized = False
                letter_progress = 0.0
                word_suggestions = []
            else:
                current_word, word_finalized, letter_progress = word_tracker.add_prediction(
                    prediction, confidence
                )
                word_suggestions = word_tracker.get_word_suggestions(current_word, max_suggestions=5)

            word_completed = None
            amharic_translation = None

            if word_finalized and current_word:
                word_completed = current_word
                if self.enable_amharic and self.amharic_translator and self.amharic_translator.translation_available:
                    amharic_translation = self.amharic_translator.translate(current_word)
                self._handle_word_completion_with_tts(current_word, amharic_translation)
                word_tracker.reset_word()

            processing_time = time.time() - start_time
            performance_monitor.record_frame(processing_time, confidence if is_sign_detected else 0.0)

            return {
                "letter": display_letter,
                "confidence": round(confidence, 3) if is_sign_detected else 0.0,
                "current_word": current_word if is_sign_detected else word_tracker.current_word,
                "word_suggestions": word_suggestions,
                "word_finalized": word_finalized,
                "word_completed": word_completed,
                "amharic_translation": amharic_translation,
                "letter_progress": round(letter_progress, 3),
                "timestamp": datetime.now().isoformat(),
                "is_sign_detected": is_sign_detected,
                "fps": round(performance_monitor.get_fps(), 1)
            }

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                "letter": "",
                "confidence": 0.0,
                "current_word": "",
                "word_suggestions": [],
                "word_finalized": False,
                "word_completed": None,
                "amharic_translation": None,
                "letter_progress": 0.0,
                "timestamp": datetime.now().isoformat(),
                "is_sign_detected": False,
                "error": str(e)
            }

    def _handle_word_completion_with_tts(self, word: str, amharic_translation: Optional[str]):
        """Handle word completion with TTS for both English and Amharic."""
        try:
            if hasattr(self.inference_system, 'tts_engine') and self.inference_system.tts_engine:
                self.inference_system.tts_engine.speak(word, 'en')
                if self.enable_amharic and amharic_translation:
                    import threading
                    threading.Timer(1.2, lambda: self.inference_system.tts_engine.speak(amharic_translation, 'am')).start()
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return time.time() - self.start_time
