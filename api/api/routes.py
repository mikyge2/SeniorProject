from fastapi import APIRouter, WebSocket, HTTPException, UploadFile, File
from typing import Optional
import json
import base64
import time
import logging

from ..models.schemas import PredictionResponse, HealthResponse, ResetResponse
from ..core.engine import ASLInferenceEngine
from ..services.connection_manager import ConnectionManager, PerformanceMonitor
from ..core.inference import WordTracker

router = APIRouter()
logger = logging.getLogger(__name__)

inference_engine: Optional[ASLInferenceEngine] = None
connection_manager = ConnectionManager()

def set_inference_engine(engine: ASLInferenceEngine):
    global inference_engine
    inference_engine = engine

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint."""
    uptime = inference_engine.get_uptime() if inference_engine else None

    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        connections=connection_manager.get_connection_count(),
        uptime_seconds=round(uptime, 1) if uptime else None
    )

@router.get("/stats/{connection_id}")
async def get_session_stats(connection_id: str):
    """Get session statistics for a connection."""
    stats = connection_manager.get_session_stats(connection_id)
    if stats:
        return stats
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Connection {connection_id} not found"
        )

@router.post("/reset/{connection_id}", response_model=ResetResponse)
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

@router.post("/predict", response_model=PredictionResponse)
async def predict_rest(
    file: UploadFile = File(...),
    connection_id: Optional[str] = None
):
    """REST endpoint for frame prediction."""
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


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    if inference_engine is None:
        await websocket.close(code=1013, reason="Inference engine not initialized")
        return

    connection_id = await connection_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("type") == "frame":
                image_base64 = data.get("data", "")
                if not image_base64:
                    continue

                image_data = base64.b64decode(image_base64)
                connection = connection_manager.get_connection(connection_id)
                if not connection:
                    continue

                word_tracker = connection["word_tracker"]
                performance_monitor = connection["performance_monitor"]
                smart_predictor = connection["smart_predictor"]
                connection["frame_count"] += 1

                result = inference_engine.process_frame(image_data, word_tracker, performance_monitor)

                if result.get("is_sign_detected"):
                    connection["letters_recognized"] += 1
                    connection["total_confidence"] += result.get("confidence", 0.0)

                if result.get("word_completed"):
                    connection["words_completed"] += 1
                    smart_predictor.record_completed_word(result["word_completed"])

                if result.get("current_word"):
                    result["word_suggestions"] = smart_predictor.get_smart_suggestions(
                        result["current_word"],
                        result.get("word_suggestions", [])
                    )

                response = {"type": "prediction", "data": result}
                await websocket.send_text(json.dumps(response))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(connection_id)
