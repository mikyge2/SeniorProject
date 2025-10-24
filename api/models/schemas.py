from pydantic import BaseModel
from typing import Dict, Optional, Any, List

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
    """Statistics for a recognition session"""
    total_frames: int
    letters_recognized: int
    words_completed: int
    average_confidence: float
    session_duration_seconds: float
