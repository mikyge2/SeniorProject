"""
FastAPI Backend for Real-time ASL Recognition Service
Converts enhanced_asl_inference.py into a mobile-ready API
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
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
    """Defines the structure of a single prediction response."""
    letter: str
    confidence: float
    current_word: str
    word_suggestions: List[str]
    word_finalized: bool
    word_completed: Optional[str] = None
    amharic_translation: Optional[str] = None
    letter_progress: float
    timestamp: str

class HealthResponse(BaseModel):
    """Defines the structure of the health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    connections: int

class ResetResponse(BaseModel):
    """Defines the structure of the word tracker reset response."""
    status: str
    message: str


class ASLInferenceEngine:
    """A wrapper for the ASL inference system, adapted for a headless API environment.

    This class loads the ASL model and all related components (MediaPipe,
    translator, TTS) and provides a single method to process raw image frames.
    It is designed to be instantiated once at server startup.

    Attributes:
        model_path (str): The file path to the TFLite model.
        metadata_path (str): The file path to the class metadata JSON.
        enable_amharic (bool): Flag to enable Amharic translation.
        inference_system (ASLRealTimeInference): The core inference logic instance.
        amharic_translator (Optional[SimpleAmharicTranslator]): The translator instance.
    """

    def __init__(self, model_path: str, metadata_path: str, enable_amharic: bool = True):
        """Initializes and loads the entire ASL inference pipeline.

        Args:
            model_path: The path to the TFLite model file.
            metadata_path: The path to the metadata JSON file.
            enable_amharic: Whether to enable Amharic translation features.
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_amharic = enable_amharic

        # Initialize core inference system (with TTS enabled for API)
        self.inference_system = ASLRealTimeInference(
            model_path=model_path,
            metadata_path=metadata_path,
            camera_index=0,  # Not used in API mode
            enable_speech=True,   # Re-enabled for TTS
            use_google_tts=True,  # Re-enabled for better TTS
            show_landmarks=False,
            enable_amharic=enable_amharic
        )

        # Load model and initialize components
        self._initialize_system()

        # Initialize Amharic translator if enabled
        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = SimpleAmharicTranslator(use_translation=True)
            logger.info(f"Amharic translation: {'Available' if self.amharic_translator.translation_available else 'Not available'}")

    def _initialize_system(self):
        """Loads the model, metadata, and initializes MediaPipe."""
        try:
            # Load model and metadata
            self.inference_system.load_model()
            self.inference_system.load_metadata()
            self.inference_system.initialize_mediapipe()

            logger.info("ASL inference engine initialized successfully")
            logger.info(f"Model: {self.model_path}")
            logger.info(f"Classes loaded: {len(self.inference_system.class_mapping)}")

        except Exception as e:
            logger.error(f"Failed to initialize inference system: {e}")
            raise

    def process_frame(self, frame_data: bytes, word_tracker: WordTracker) -> Dict[str, Any]:
        """Processes a single image frame to perform ASL recognition.

        This method takes raw image bytes, decodes them, runs the full
        MediaPipe and TensorFlow Lite inference pipeline, updates the state
        of the provided `WordTracker`, and returns a dictionary of results.

        Args:
            frame_data: The raw byte content of the image file (e.g., JPEG, PNG).
            word_tracker: The stateful `WordTracker` instance for the current
                user/connection.

        Returns:
            A dictionary containing the prediction results, including the
            detected letter, confidence, current word, and translation if a
            word was completed.
        """
        try:
            # Decode image from bytes
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image data")

            # Flip frame horizontally for mirror effect (like mobile camera)
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

            # Update word tracker
            current_word, word_finalized, letter_progress = word_tracker.add_prediction(
                prediction, confidence
            )

            # Get word suggestions
            word_suggestions = word_tracker.get_word_suggestions(current_word, max_suggestions=3)

            # Handle word completion
            word_completed = None
            amharic_translation = None

            if word_finalized and current_word:
                word_completed = current_word

                # Get Amharic translation if enabled
                if self.enable_amharic and self.amharic_translator and self.amharic_translator.translation_available:
                    amharic_translation = self.amharic_translator.translate(current_word)
                    logger.info(f"Translation: '{current_word}' -> '{amharic_translation}'")

                # Handle TTS (both English and Amharic)
                self._handle_word_completion_with_tts(current_word, amharic_translation)

                # Reset word tracker for next word
                word_tracker.reset_word()

                logger.info(f"Word completed: '{word_completed}'" +
                          (f" -> Amharic: '{amharic_translation}'" if amharic_translation else ""))

            return {
                "letter": prediction,
                "confidence": round(confidence, 3),
                "current_word": current_word,
                "word_suggestions": word_suggestions,
                "word_finalized": word_finalized,
                "word_completed": word_completed,
                "amharic_translation": amharic_translation,
                "letter_progress": round(letter_progress, 3),
                "timestamp": datetime.now().isoformat()
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
                "error": str(e)
            }

    def _handle_word_completion_with_tts(self, word: str, amharic_translation: Optional[str]):
        """Triggers Text-to-Speech (TTS) for a completed word and its translation.

        This method is designed to be non-blocking. It queues the English word
        for immediate speech and schedules the Amharic translation to be spoken
        after a short delay.

        Args:
            word: The completed English word.
            amharic_translation: The Amharic translation of the word, if available.
        """
        try:
            # Use the original inference system's TTS if available
            if hasattr(self.inference_system, 'tts_engine') and self.inference_system.tts_engine:
                # Speak English word
                self.inference_system.tts_engine.speak(word, 'en')
                logger.info(f"Speaking English: '{word}'")

                # Speak Amharic translation if available
                if self.enable_amharic and amharic_translation:
                    # Add delay for Amharic speech (as in original)
                    import threading
                    threading.Timer(1.2, lambda: self.inference_system.tts_engine.speak(amharic_translation, 'am')).start()
                    logger.info(f"Speaking Amharic: '{amharic_translation}' (delayed)")
            else:
                # Fallback: try to initialize TTS if not available
                logger.warning("TTS engine not available in inference system")

        except Exception as e:
            logger.error(f"TTS error: {e}")


class ConnectionManager:
    """Manages active WebSocket connections and their associated user states.

    This class acts as a central registry for all connected clients. It assigns
    a unique ID to each connection and maintains a separate `WordTracker`
    instance for each, ensuring that users' word-building states are isolated.

    Attributes:
        active_connections: A dictionary mapping unique connection IDs to a
            dict containing the WebSocket object and the user's `WordTracker`.
    """

    def __init__(self):
        """Initializes the ConnectionManager."""
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accepts a new WebSocket connection and prepares its state.

        This method accepts the connection, generates a unique ID, creates a
        new `WordTracker` for the connection, and stores it in the active
        connections registry.

        Args:
            websocket: The incoming `fastapi.WebSocket` object.

        Returns:
            The unique connection ID string for the new connection.
        """
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        word_tracker = WordTracker(
            window_size=6,
            confidence_threshold=0.6,
            pause_threshold=2.0,
            min_letter_duration=0.8
        )

        self.active_connections[connection_id] = {
            "websocket": websocket,
            "word_tracker": word_tracker,
            "connected_at": datetime.now(),
            "frame_count": 0
        }

        logger.info(f"New WebSocket connection: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        """Removes a connection from the registry upon disconnection.

        Args:
            connection_id: The unique ID of the connection to remove.
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")

    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the data associated with a specific connection ID.

        Args:
            connection_id: The ID of the connection to retrieve.

        Returns:
            A dictionary containing the connection's data (including its
            `WordTracker`), or None if the ID is not found.
        """
        return self.active_connections.get(connection_id)

    def get_connection_count(self) -> int:
        """Returns the current number of active WebSocket connections.

        Returns:
            The total count of active connections.
        """
        return len(self.active_connections)

    def reset_word_tracker(self, connection_id: str) -> bool:
        """Resets the `WordTracker` for a specific connection.

        Args:
            connection_id: The ID of the connection whose tracker should be reset.

        Returns:
            True if the tracker was successfully reset, False if the
            connection ID was not found.
        """
        if connection_id in self.active_connections:
            self.active_connections[connection_id]["word_tracker"].reset_word()
            logger.info(f"Word tracker reset for connection: {connection_id}")
            return True
        return False


# Initialize FastAPI app
app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition with Amharic translation",
    version="1.0.0"
)

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_engine: Optional[ASLInferenceEngine] = None
connection_manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initializes the ASLInferenceEngine when the FastAPI application starts.

    This event handler ensures that the model and all necessary components are
    loaded into memory once, before any requests are served. This avoids the

    overhead of loading the model on each request.

    Raises:
        FileNotFoundError: If the model or metadata files are not found.
    """
    global inference_engine

    # Configure paths (adjust as needed)
    model_path = "export/asl_model.tflite"
    metadata_path = "processed_asl/metadata.json"

    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not Path(metadata_path).exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        # Initialize inference engine with Amharic and TTS enabled
        inference_engine = ASLInferenceEngine(
            model_path=model_path,
            metadata_path=metadata_path,
            enable_amharic=True
        )

        # Test translation availability
        if inference_engine.amharic_translator:
            if inference_engine.amharic_translator.translation_available:
                # Test translation
                test_translation = inference_engine.amharic_translator.translate("hello")
                logger.info(f"Translation test: 'hello' -> '{test_translation}'")
            else:
                logger.error("Amharic translator failed to initialize")
        else:
            logger.error("Amharic translator is None")

        # Test TTS availability
        if hasattr(inference_engine.inference_system, 'tts_engine'):
            if inference_engine.inference_system.tts_engine:
                logger.info("TTS engine initialized successfully")
            else:
                logger.error("TTS engine is None")
        else:
            logger.error("TTS engine not found in inference system")

        logger.info("ASL Recognition API server started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Provides a health check endpoint for monitoring the service.

    Returns:
        A `HealthResponse` object indicating the server's status, whether the
        model is loaded, the current timestamp, and the number of active
        connections.
    """
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        timestamp=datetime.now().isoformat(),
        connections=connection_manager.get_connection_count()
    )


@app.post("/reset/{connection_id}", response_model=ResetResponse)
async def reset_connection(connection_id: str):
    """Resets the word tracker state for a specific connection ID.

    This allows a client to clear the current word being formed without
    disconnecting.

    Args:
        connection_id: The unique ID of the connection.

    Returns:
        A `ResetResponse` indicating success or failure.
    """
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
    """Provides a REST endpoint for single-frame predictions.

    This endpoint serves as a fallback or for use cases where WebSockets are
    not suitable. It accepts an image file and an optional `connection_id` to
    maintain state across calls.

    Args:
        file: An uploaded image file.
        connection_id: An optional string to identify a user session. If provided,
            the server will use the associated `WordTracker`.

    Returns:
        A `PredictionResponse` with the inference results.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    try:
        # Read image data
        image_data = await file.read()

        # Create or get word tracker
        if connection_id:
            connection = connection_manager.get_connection(connection_id)
            if connection:
                word_tracker = connection["word_tracker"]
            else:
                # Create new word tracker for this connection_id
                word_tracker = WordTracker()
        else:
            # Create temporary word tracker
            word_tracker = WordTracker()

        # Process frame
        result = inference_engine.process_frame(image_data, word_tracker)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"REST prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Handles real-time, stateful ASL recognition over a WebSocket connection.

    This is the primary endpoint for mobile applications. It establishes a
    persistent connection, allowing a client to stream frames (as base64-encoded
    strings) and receive immediate feedback. The server maintains a unique
    `WordTracker` for each connection.

    **Client-to-Server Message Types:**
    - `{"type": "frame", "data": "..."}`: Sends a base64 image for processing.
    - `{"type": "reset"}`: Resets the word tracker for the connection.
    - `{"type": "config", "data": {"hold_time": 0.8}}`: (Example) Adjusts settings.

    **Server-to-Client Message Types:**
    - `{"type": "prediction", "data": {...}}`: The main prediction result.
    - `{"type": "error", "message": "..."}`: Sent when an error occurs.
    - `{"type": "reset_confirmed"}`: Confirms that the word tracker was reset.

    Args:
        websocket: The `fastapi.WebSocket` object for the connection.
    """
    if inference_engine is None:
        await websocket.close(code=1013, reason="Inference engine not initialized")
        return

    # Connect and get connection ID
    connection_id = await connection_manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()

            try:
                data = json.loads(message)

                if data.get("type") == "frame":
                    # Decode base64 image data
                    image_base64 = data.get("data", "")
                    if not image_base64:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "No image data provided"
                        }))
                        continue

                    # Decode base64 to bytes
                    try:
                        image_data = base64.b64decode(image_base64)
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Failed to decode base64 data: {str(e)}"
                        }))
                        continue

                    # Get connection and word tracker
                    connection = connection_manager.get_connection(connection_id)
                    if not connection:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Connection not found"
                        }))
                        continue

                    word_tracker = connection["word_tracker"]
                    connection["frame_count"] += 1

                    # Process frame
                    start_time = time.time()
                    result = inference_engine.process_frame(image_data, word_tracker)
                    processing_time = (time.time() - start_time) * 1000  # ms

                    # Add processing stats
                    result["processing_time_ms"] = round(processing_time, 2)
                    result["frame_count"] = connection["frame_count"]

                    # Send prediction result
                    response = {
                        "type": "prediction",
                        "data": result
                    }

                    await websocket.send_text(json.dumps(response))

                    # Log performance
                    if processing_time > 200:
                        logger.warning(f"Slow processing: {processing_time:.1f}ms for frame {connection['frame_count']}")

                elif data.get("type") == "reset":
                    # Reset word tracker
                    connection = connection_manager.get_connection(connection_id)
                    if connection:
                        connection["word_tracker"].reset_word()
                        await websocket.send_text(json.dumps({
                            "type": "reset_confirmed",
                            "message": "Word tracker reset"
                        }))

                elif data.get("type") == "config":
                    # Update configuration settings
                    config_data = data.get("data", {})
                    connection = connection_manager.get_connection(connection_id)

                    if connection and "hold_time" in config_data:
                        hold_time = float(config_data["hold_time"])
                        if 0.1 <= hold_time <= 5.0:  # Reasonable bounds
                            connection["word_tracker"].min_letter_duration = hold_time
                            await websocket.send_text(json.dumps({
                                "type": "config_updated",
                                "message": f"Hold time updated to {hold_time:.1f}s"
                            }))
                            logger.info(f"Hold time updated to {hold_time:.1f}s for connection {connection_id}")
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Hold time must be between 0.1 and 5.0 seconds"
                            }))

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
    """Provides a simple root endpoint with basic API information.

    Returns:
        A JSON object with a welcome message and a list of available endpoints.
    """
    return {
        "message": "ASL Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "websocket": "WS /ws/predict",
            "rest_predict": "POST /predict",
            "reset": "POST /reset/{connection_id}"
        },
        "websocket_example": {
            "connect": "ws://localhost:8000/ws/predict",
            "message_format": {
                "type": "frame",
                "data": "base64_encoded_image"
            }
        }
    }


if __name__ == "__main__":
    # For development - use uvicorn command for production
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )