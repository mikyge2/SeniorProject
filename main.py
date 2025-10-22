"""
FastAPI Backend for Real-time ASL Recognition Service
Version 2.2 - Enhanced with letter cooldown, better suggestions, and W-bias mitigation
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import deque
import io

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import classes from the original inference script
from enhanced_asl_inference import (
    ASLRealTimeInference,
    WordTracker,
    SimpleAmharicTranslator,
    OnlineDictionaryAPI
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Response models
class PredictionResponse(BaseModel):
    letter: str
    confidence: float
    current_word: str
    word_suggestions: List[str]
    word_finalized: bool
    word_completed: Optional[str] = None
    amharic_translation: Optional[str] = None
    letter_progress: float
    timestamp: str
    is_sign_detected: bool = True
    fps: Optional[float] = None
    session_stats: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    connections: int
    uptime_seconds: Optional[float] = None

class ResetResponse(BaseModel):
    status: str
    message: str

class SessionStats(BaseModel):
    total_frames: int
    letters_recognized: int
    words_completed: int
    average_confidence: float
    session_duration_seconds: float


class PerformanceMonitor:
    """Monitor and optimize performance metrics"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)

    def record_frame(self, processing_time: float, confidence: float):
        """Record frame processing metrics"""
        self.frame_times.append(processing_time)
        self.confidences.append(confidence)

    def get_fps(self) -> float:
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_average_confidence(self) -> float:
        """Get average confidence over window"""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class SmartWordPredictor:
    """Advanced word prediction with extensive dictionary"""

    def __init__(self):
        self.word_history = deque(maxlen=10)
        self.common_words = self._load_common_words()

    def _load_common_words(self) -> List[str]:
        """Load extensive list of common ASL words"""
        return [
            # Basic communication
            "HELLO", "HI", "THANKS", "THANK", "YOU", "PLEASE", "SORRY", "YES", "NO",
            "HELP", "LOVE", "LIKE", "WANT", "NEED", "HAVE", "GET",

            # Family & People
            "FAMILY", "FRIEND", "MOTHER", "FATHER", "SISTER", "BROTHER", "CHILD",
            "BABY", "PARENT", "GRANDMOTHER", "GRANDFATHER", "AUNT", "UNCLE",

            # Feelings & Emotions
            "HAPPY", "SAD", "ANGRY", "EXCITED", "TIRED", "SICK", "HURT", "PAIN",
            "GOOD", "BAD", "NICE", "MEAN", "KIND", "WORRIED", "AFRAID", "FEAR",
            "CALM", "STRESS", "RELAX", "COMFORTABLE", "UNCOMFORTABLE",

            # Food & Drink
            "WATER", "FOOD", "EAT", "DRINK", "HUNGRY", "THIRSTY", "BREAKFAST",
            "LUNCH", "DINNER", "SNACK", "MILK", "COFFEE", "TEA", "JUICE",

            # Time
            "TIME", "DAY", "NIGHT", "MORNING", "AFTERNOON", "EVENING", "TODAY",
            "TOMORROW", "YESTERDAY", "WEEK", "MONTH", "YEAR", "HOUR", "MINUTE",
            "NOW", "LATER", "SOON", "EARLY", "LATE", "ALWAYS", "NEVER", "SOMETIMES",

            # Places
            "HOME", "HOUSE", "WORK", "SCHOOL", "HOSPITAL", "STORE", "CHURCH",
            "RESTAURANT", "BATHROOM", "BEDROOM", "KITCHEN", "OFFICE",

            # Actions
            "GO", "COME", "STOP", "START", "FINISH", "CONTINUE", "WAIT", "RUN",
            "WALK", "SIT", "STAND", "SLEEP", "WAKE", "TEACH", "LEARN", "STUDY",
            "READ", "WRITE", "PLAY", "WORK", "CALL", "TEXT", "EMAIL",

            # Questions
            "WHERE", "WHAT", "WHEN", "HOW", "WHY", "WHO", "WHICH",

            # School & Learning
            "TEACHER", "STUDENT", "CLASS", "LESSON", "BOOK", "PAPER", "PEN",
            "PENCIL", "TEST", "HOMEWORK", "GRADE", "LEARN", "STUDY",

            # Technology
            "COMPUTER", "PHONE", "INTERNET", "EMAIL", "MESSAGE", "SEARCH",
            "WEBSITE", "VIDEO", "PICTURE", "CAMERA",

            # Body & Health
            "DOCTOR", "NURSE", "MEDICINE", "HOSPITAL", "HEALTHY", "EXERCISE",
            "BODY", "HEAD", "HAND", "FOOT", "EYE", "EAR", "MOUTH", "NOSE",

            # Communication
            "TALK", "SPEAK", "SAY", "TELL", "ASK", "ANSWER", "LISTEN", "HEAR",
            "SEE", "LOOK", "WATCH", "SHOW", "UNDERSTAND", "KNOW", "THINK",
            "REMEMBER", "FORGET", "BELIEVE", "DOUBT",

            # Quantities & Numbers
            "MORE", "LESS", "MANY", "FEW", "ALL", "SOME", "NONE", "ENOUGH",
            "TOO", "VERY", "LITTLE", "BIG", "SMALL",

            # Adjectives
            "BEAUTIFUL", "UGLY", "CLEAN", "DIRTY", "NEW", "OLD", "YOUNG",
            "FAST", "SLOW", "EASY", "HARD", "DIFFICULT", "SIMPLE", "IMPORTANT",
            "STRONG", "WEAK", "HOT", "COLD", "WARM", "COOL",

            # Common verbs
            "MAKE", "DO", "GIVE", "TAKE", "PUT", "FIND", "KEEP", "TRY",
            "HELP", "CHANGE", "USE", "MOVE", "OPEN", "CLOSE", "BEGIN", "END"
        ]

    def get_smart_suggestions(self, partial_word: str, base_suggestions: List[str]) -> List[str]:
        """Get enhanced word suggestions (up to 10)"""
        if not partial_word:
            return base_suggestions[:5]

        # Combine base suggestions with common words
        all_suggestions = list(set(base_suggestions + self.common_words))

        # Filter matches
        matches = [w for w in all_suggestions if w.startswith(partial_word.upper())]

        # Sort by length (prefer shorter completions)
        matches.sort(key=lambda w: len(w))

        # Return top 10 suggestions
        return matches[:10]

    def record_completed_word(self, word: str):
        """Record completed word for context"""
        self.word_history.append(word)


class ASLInferenceEngine:
    """Enhanced ASL inference engine with W-bias mitigation"""

    def __init__(self, model_path: str, metadata_path: str, enable_amharic: bool = True):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_amharic = enable_amharic
        self.start_time = time.time()

        # W-bias mitigation
        self.w_count = 0
        self.total_predictions = 0
        self.last_prediction = None
        self.prediction_count = 0

        # Initialize core inference system
        self.inference_system = ASLRealTimeInference(
            model_path=model_path,
            metadata_path=metadata_path,
            camera_index=0,
            enable_speech=False,  # Disable backend TTS (frontend handles it)
            use_google_tts=False,
            show_landmarks=False,
            enable_amharic=enable_amharic
        )

        self._initialize_system()

        # Initialize Amharic translator if enabled
        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = SimpleAmharicTranslator(use_translation=True)
            logger.info(f"Amharic translation: {'Available' if self.amharic_translator.translation_available else 'Not available'}")

    def _initialize_system(self):
        """Initialize the inference system components."""
        try:
            self.inference_system.load_model()
            self.inference_system.load_metadata()
            self.inference_system.initialize_mediapipe()

            logger.info("ASL inference engine initialized successfully")
            logger.info(f"Model: {self.model_path}")
            logger.info(f"Classes loaded: {len(self.inference_system.class_mapping)}")

        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            raise

    def _mitigate_w_bias(self, prediction: str, confidence: float) -> tuple[bool, str]:
        """
        Mitigate W over-recognition bias

        Returns: (is_valid, filtered_prediction)
        """
        # Track statistics
        self.total_predictions += 1
        if prediction.upper() == 'W':
            self.w_count += 1

        # Log W-bias ratio every 100 predictions
        if self.total_predictions % 100 == 0:
            w_ratio = (self.w_count / self.total_predictions) * 100
            logger.info(f"W-bias ratio: {w_ratio:.1f}% ({self.w_count}/{self.total_predictions})")

        # Apply stricter confidence threshold for W
        if prediction.upper() == 'W':
            if confidence < 0.80:  # Higher threshold for W
                logger.debug(f"Filtered W prediction (confidence: {confidence:.2f} < 0.80)")
                return False, ""

        # Require consecutive predictions for stability
        if prediction == self.last_prediction:
            self.prediction_count += 1
        else:
            self.last_prediction = prediction
            self.prediction_count = 1

        # Require at least 2 consecutive same predictions
        if self.prediction_count < 2:
            logger.debug(f"Waiting for consecutive prediction: {prediction} (count: {self.prediction_count})")
            return False, ""

        return True, prediction

    def process_frame(self, frame_data: bytes, word_tracker: WordTracker,
                     performance_monitor: PerformanceMonitor) -> Dict[str, Any]:
        """Process frame with enhanced features and W-bias mitigation"""
        start_time = time.time()

        try:
            # Decode image from bytes
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image data")

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.inference_system.hands.process(rgb_frame)

            # Extract features
            landmarks = self.inference_system.extract_hand_landmarks(results)
            hand_crop = self.inference_system.crop_hand_region(frame, results)

            # Preprocess inputs
            image_input, landmarks_input = self.inference_system.preprocess_inputs(hand_crop, landmarks)

            # Run inference
            prediction, confidence = self.inference_system.predict(image_input, landmarks_input)

            # Apply W-bias mitigation
            is_valid, filtered_prediction = self._mitigate_w_bias(prediction, confidence)

            if not is_valid:
                # Filtered out by W-bias mitigation
                is_sign_detected = False
                display_letter = ""
                current_word = word_tracker.current_word
                word_finalized = False
                letter_progress = 0.0
                word_suggestions = []
            elif filtered_prediction.lower() in ["del", "space"]:
                # Handle del/space as no sign detected
                is_sign_detected = False
                display_letter = ""
                current_word = word_tracker.current_word
                word_finalized = False
                letter_progress = 0.0
                word_suggestions = []
            else:
                # Valid sign detected
                is_sign_detected = True
                display_letter = filtered_prediction

                # Normal processing
                current_word, word_finalized, letter_progress = word_tracker.add_prediction(
                    filtered_prediction, confidence
                )
                word_suggestions = word_tracker.get_word_suggestions(current_word, max_suggestions=10)

            # Handle word completion
            word_completed = None
            amharic_translation = None

            if word_finalized and current_word:
                word_completed = current_word

                # Get Amharic translation if enabled
                if self.enable_amharic and self.amharic_translator and self.amharic_translator.translation_available:
                    amharic_translation = self.amharic_translator.translate(current_word)
                    logger.info(f"Translation: '{current_word}' -> '{amharic_translation}'")

                # Reset word tracker for next word
                word_tracker.reset_word()

                # Reset prediction tracking for new word
                self.last_prediction = None
                self.prediction_count = 0

                logger.info(f"Word completed: '{word_completed}'" +
                          (f" -> Amharic: '{amharic_translation}'" if amharic_translation else ""))

            # Record performance metrics
            processing_time = time.time() - start_time
            performance_monitor.record_frame(processing_time, confidence if is_sign_detected else 0.0)

            return {
                "letter": display_letter,
                "confidence": round(confidence, 3) if is_sign_detected else 0.0,
                "current_word": current_word,
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

    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return time.time() - self.start_time

    def get_w_bias_stats(self) -> Dict[str, Any]:
        """Get W-bias statistics"""
        if self.total_predictions == 0:
            return {"w_ratio": 0.0, "total_predictions": 0}

        w_ratio = (self.w_count / self.total_predictions) * 100
        return {
            "w_ratio": round(w_ratio, 2),
            "w_count": self.w_count,
            "total_predictions": self.total_predictions
        }


class ConnectionManager:
    """Enhanced connection manager with session tracking"""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept new WebSocket connection with enhanced tracking"""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        word_tracker = WordTracker(
            window_size=6,
            confidence_threshold=0.65,  # Slightly increased for better accuracy
            pause_threshold=2.0,
            min_letter_duration=0.8
        )

        performance_monitor = PerformanceMonitor()
        smart_predictor = SmartWordPredictor()

        self.active_connections[connection_id] = {
            "websocket": websocket,
            "word_tracker": word_tracker,
            "performance_monitor": performance_monitor,
            "smart_predictor": smart_predictor,
            "connected_at": datetime.now(),
            "frame_count": 0,
            "letters_recognized": 0,
            "words_completed": 0,
            "total_confidence": 0.0
        }

        logger.info(f"New WebSocket connection: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        """Remove connection with session summary"""
        if connection_id in self.active_connections:
            conn = self.active_connections[connection_id]
            duration = (datetime.now() - conn["connected_at"]).total_seconds()

            logger.info(f"WebSocket disconnected: {connection_id}")
            logger.info(f"  Session duration: {duration:.1f}s")
            logger.info(f"  Frames processed: {conn['frame_count']}")
            logger.info(f"  Words completed: {conn['words_completed']}")

            del self.active_connections[connection_id]

    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection data by ID"""
        return self.active_connections.get(connection_id)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_session_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get session statistics for a connection"""
        conn = self.get_connection(connection_id)
        if not conn:
            return None

        duration = (datetime.now() - conn["connected_at"]).total_seconds()
        avg_confidence = (conn["total_confidence"] / conn["letters_recognized"]
                         if conn["letters_recognized"] > 0 else 0.0)

        return {
            "total_frames": conn["frame_count"],
            "letters_recognized": conn["letters_recognized"],
            "words_completed": conn["words_completed"],
            "average_confidence": round(avg_confidence, 3),
            "session_duration_seconds": round(duration, 1),
            "fps": round(conn["performance_monitor"].get_fps(), 1)
        }

    def reset_word_tracker(self, connection_id: str) -> bool:
        """Reset word tracker for a connection"""
        if connection_id in self.active_connections:
            self.active_connections[connection_id]["word_tracker"].reset_word()
            logger.info(f"Word tracker reset for connection: {connection_id}")
            return True
        return False


# Initialize FastAPI app
app = FastAPI(
    title="ASL Recognition API",
    description="Real-time ASL recognition with Amharic translation - v2.2",
    version="2.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_engine: Optional[ASLInferenceEngine] = None
connection_manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the ASL inference engine on startup"""
    global inference_engine

    model_path = "export/asl_model.tflite"
    metadata_path = "processed_asl/metadata.json"

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not Path(metadata_path).exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        inference_engine = ASLInferenceEngine(
            model_path=model_path,
            metadata_path=metadata_path,
            enable_amharic=True
        )

        if inference_engine.amharic_translator:
            if inference_engine.amharic_translator.translation_available:
                test_translation = inference_engine.amharic_translator.translate("hello")
                logger.info(f"Translation test: 'hello' -> '{test_translation}'")

        logger.info("ASL Recognition API v2.2 started successfully")
        logger.info("Features: W-bias mitigation, Enhanced suggestions, Frontend TTS")

    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint"""
    uptime = inference_engine.get_uptime() if inference_engine else None

    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        timestamp=datetime.now().isoformat(),
        connections=connection_manager.get_connection_count(),
        uptime_seconds=round(uptime, 1) if uptime else None
    )


@app.get("/stats/{connection_id}")
async def get_session_stats(connection_id: str):
    """Get session statistics for a connection"""
    stats = connection_manager.get_session_stats(connection_id)
    if stats:
        return stats
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Connection {connection_id} not found"
        )


@app.get("/w-bias-stats")
async def get_w_bias_stats():
    """Get W-bias mitigation statistics"""
    if inference_engine:
        return inference_engine.get_w_bias_stats()
    else:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")


@app.post("/reset/{connection_id}", response_model=ResetResponse)
async def reset_connection(connection_id: str):
    """Reset word tracker for a specific connection"""
    if connection_manager.reset_word_tracker(connection_id):
        return ResetResponse(
            status="success",
            message=f"Word tracker reset for connection {connection_id}"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Connection {connection_id} not found"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_rest(
    file: UploadFile = File(...),
    connection_id: Optional[str] = None
):
    """REST endpoint for frame prediction"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        image_data = await file.read()

        if connection_id:
            connection = connection_manager.get_connection(connection_id)
            if connection:
                word_tracker = connection["word_tracker"]
                performance_monitor = connection["performance_monitor"]
            else:
                word_tracker = WordTracker()
                performance_monitor = PerformanceMonitor()
        else:
            word_tracker = WordTracker()
            performance_monitor = PerformanceMonitor()

        result = inference_engine.process_frame(image_data, word_tracker, performance_monitor)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"REST prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Enhanced WebSocket endpoint with all v2.2 features"""
    if inference_engine is None:
        await websocket.close(code=1013, reason="Inference engine not initialized")
        return

    connection_id = await connection_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive_text()

            try:
                data = json.loads(message)

                if data.get("type") == "frame":
                    image_base64 = data.get("data", "")
                    if not image_base64:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "No image data provided"
                        }))
                        continue

                    try:
                        image_data = base64.b64decode(image_base64)
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Failed to decode base64 data: {str(e)}"
                        }))
                        continue

                    connection = connection_manager.get_connection(connection_id)
                    if not connection:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Connection not found"
                        }))
                        continue

                    word_tracker = connection["word_tracker"]
                    performance_monitor = connection["performance_monitor"]
                    smart_predictor = connection["smart_predictor"]
                    connection["frame_count"] += 1

                    # Process frame
                    start_time = time.time()
                    result = inference_engine.process_frame(
                        image_data, word_tracker, performance_monitor
                    )
                    processing_time = (time.time() - start_time) * 1000

                    # Update connection stats
                    if result.get("is_sign_detected"):
                        connection["letters_recognized"] += 1
                        connection["total_confidence"] += result.get("confidence", 0.0)

                    if result.get("word_completed"):
                        connection["words_completed"] += 1
                        smart_predictor.record_completed_word(result["word_completed"])

                    # Enhance word suggestions with smart predictor (up to 10)
                    if result.get("current_word"):
                        result["word_suggestions"] = smart_predictor.get_smart_suggestions(
                            result["current_word"],
                            result.get("word_suggestions", [])
                        )

                    # Add processing stats
                    result["processing_time_ms"] = round(processing_time, 2)
                    result["frame_count"] = connection["frame_count"]

                    # Add session stats every 30 frames
                    if connection["frame_count"] % 30 == 0:
                        result["session_stats"] = connection_manager.get_session_stats(connection_id)

                    response = {
                        "type": "prediction",
                        "data": result
                    }

                    await websocket.send_text(json.dumps(response))

                    if processing_time > 200:
                        logger.warning(f"Slow processing: {processing_time:.1f}ms for frame {connection['frame_count']}")

                elif data.get("type") == "reset":
                    connection = connection_manager.get_connection(connection_id)
                    if connection:
                        connection["word_tracker"].reset_word()
                        await websocket.send_text(json.dumps({
                            "type": "reset_confirmed",
                            "message": "Word tracker reset"
                        }))

                elif data.get("type") == "get_stats":
                    stats = connection_manager.get_session_stats(connection_id)
                    if stats:
                        await websocket.send_text(json.dumps({
                            "type": "stats",
                            "data": stats
                        }))

                elif data.get("type") == "config":
                    config_data = data.get("data", {})
                    connection = connection_manager.get_connection(connection_id)

                    if connection and "hold_time" in config_data:
                        hold_time = float(config_data["hold_time"])
                        if 0.1 <= hold_time <= 5.0:
                            connection["word_tracker"].min_letter_duration = hold_time
                            await websocket.send_text(json.dumps({
                                "type": "config_updated",
                                "message": f"Hold time updated to {hold_time:.1f}s"
                            }))
                            logger.info(f"Hold time updated to {hold_time:.1f}s for connection {connection_id}")

                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {data.get('type')}"
                    }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message"
                }))

            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(connection_id)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ASL Recognition API - Enhanced Edition v2.2",
        "version": "2.2.0",
        "new_features": [
            "W-bias mitigation with consecutive prediction filtering",
            "Higher confidence threshold for W (0.80 vs 0.65)",
            "Enhanced word suggestions (10 words, 180+ dictionary)",
            "Frontend TTS handling (backend TTS disabled)",
            "Session statistics tracking",
            "Real-time FPS monitoring",
            "Smart 'No Sign Detected' handling"
        ],
        "endpoints": {
            "health": "GET /health",
            "websocket": "WS /ws/predict",
            "rest_predict": "POST /predict",
            "reset": "POST /reset/{connection_id}",
            "stats": "GET /stats/{connection_id}",
            "w_bias_stats": "GET /w-bias-stats"
        },
        "websocket_message_types": {
            "frame": "Send frame for prediction",
            "reset": "Reset word tracker",
            "config": "Update configuration (hold_time)",
            "get_stats": "Get session statistics"
        },
        "w_bias_mitigation": {
            "description": "Backend filters W predictions with stricter rules",
            "confidence_threshold_w": 0.80,
            "confidence_threshold_others": 0.65,
            "consecutive_predictions_required": 2,
            "monitoring": "Check /w-bias-stats for W ratio"
        },
        "frontend_responsibilities": {
            "letter_cooldown": "Frontend handles 0.5-3.0s cooldown between letters",
            "tts": "Frontend handles all Text-to-Speech (English + Amharic)",
            "google_translate": "Frontend calls Google Translate API directly",
            "enhanced_suggestions": "Frontend merges backend + local dictionary (180+ words)"
        },
        "backend_responsibilities": {
            "model_inference": "TensorFlow Lite model prediction",
            "w_bias_mitigation": "Filter W over-recognition",
            "word_tracking": "Track current word being spelled",
            "amharic_translation": "Fallback translation via SimpleAmharicTranslator",
            "suggestions": "Provide base word suggestions from dictionary"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )