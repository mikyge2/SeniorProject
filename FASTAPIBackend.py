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
    status: str
    model_loaded: bool
    timestamp: str
    connections: int


class ResetResponse(BaseModel):
    status: str
    message: str


class ASLInferenceEngine:
    """Headless ASL inference engine for API use."""

    def __init__(self, model_path: str, metadata_path: str, enable_amharic: bool = True):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_amharic = enable_amharic

        # Initialize core inference system (headless)
        self.inference_system = ASLRealTimeInference(
            model_path=model_path,
            metadata_path=metadata_path,
            camera_index=0,  # Not used in API mode
            enable_speech=False,  # Disabled for API
            use_google_tts=False,
            show_landmarks=False,
            enable_amharic=enable_amharic
        )

        # Load model and initialize components
        self._initialize_system()

        # Initialize Amharic translator if enabled
        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = SimpleAmharicTranslator(use_translation=True)
            logger.info(
                f"Amharic translation: {'Available' if self.amharic_translator.translation_available else 'Not available'}")

    def _initialize_system(self):
        """Initialize the inference system components."""
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
        """
        Process a single frame and return prediction results.

        Args:
            frame_data: Raw image bytes
            word_tracker: WordTracker instance for this connection

        Returns:
            Dictionary with prediction results
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


class ConnectionManager:
    """Manage WebSocket connections and their word trackers."""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept new WebSocket connection and create word tracker."""
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
        """Remove connection and clean up resources."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")

    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection data by ID."""
        return self.active_connections.get(connection_id)

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def reset_word_tracker(self, connection_id: str) -> bool:
        """Reset word tracker for a connection."""
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
    """Initialize the ASL inference engine on startup."""
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
        # Initialize inference engine
        inference_engine = ASLInferenceEngine(
            model_path=model_path,
            metadata_path=metadata_path,
            enable_amharic=True
        )
        logger.info("ASL Recognition API server started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        timestamp=datetime.now().isoformat(),
        connections=connection_manager.get_connection_count()
    )


@app.post("/reset/{connection_id}", response_model=ResetResponse)
async def reset_connection(connection_id: str):
    """Reset word tracker for a specific connection."""
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
    """
    REST endpoint for frame prediction (fallback option).
    Requires multipart/form-data with image file.
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
    """
    WebSocket endpoint for continuous frame processing.

    Expected message format:
    {
        "type": "frame",
        "data": "base64_encoded_image_data"
    }

    Response format:
    {
        "type": "prediction",
        "data": {
            "letter": "H",
            "confidence": 0.94,
            "current_word": "HE",
            "word_suggestions": ["HELLO", "HELP"],
            "word_finalized": false,
            "word_completed": null,
            "amharic_translation": null,
            "letter_progress": 0.75,
            "timestamp": "2023-..."
        }
    }
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
                        logger.warning(
                            f"Slow processing: {processing_time:.1f}ms for frame {connection['frame_count']}")

                elif data.get("type") == "reset":
                    # Reset word tracker
                    connection = connection_manager.get_connection(connection_id)
                    if connection:
                        connection["word_tracker"].reset_word()
                        await websocket.send_text(json.dumps({
                            "type": "reset_confirmed",
                            "message": "Word tracker reset"
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
    """Root endpoint with API information."""
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
        "FASTAPIBackend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )