import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
from fastapi import WebSocket
from ..core.inference import WordTracker

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
    """Advanced word prediction with context awareness"""

    def __init__(self):
        self.word_history = deque(maxlen=10)
        self.common_words = self._load_common_words()

    def _load_common_words(self) -> List[str]:
        """Load common English words for better predictions"""
        return [
            "HELLO", "THANKS", "PLEASE", "SORRY", "YES", "NO", "HELP", "LOVE",
            "FAMILY", "FRIEND", "GOOD", "BAD", "HAPPY", "SAD", "WATER", "FOOD",
            "MORE", "FINISH", "WORK", "HOME", "SCHOOL", "TEACH", "LEARN", "NAME",
            "WHERE", "WHAT", "WHEN", "HOW", "WHY", "WHO", "TIME", "DAY", "NIGHT",
            "MORNING", "AFTERNOON", "EVENING", "TODAY", "TOMORROW", "YESTERDAY"
        ]

    def get_smart_suggestions(self, partial_word: str, base_suggestions: List[str]) -> List[str]:
        """Get context-aware word suggestions"""
        if not partial_word:
            return base_suggestions[:3]

        all_suggestions = list(set(base_suggestions + self.common_words))
        matches = [w for w in all_suggestions if w.startswith(partial_word.upper())]
        matches.sort(key=lambda w: (
            not w.startswith(partial_word.upper()),
            len(w),
            w not in self.common_words
        ))
        return matches[:5]

    def record_completed_word(self, word: str):
        """Record completed word for context"""
        self.word_history.append(word)

class ConnectionManager:
    """Enhanced connection manager with session tracking."""

    def __init__(self):
        self.active_connections: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept new WebSocket connection with enhanced tracking."""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        word_tracker = WordTracker(
            window_size=6,
            confidence_threshold=0.6,
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
        return connection_id

    def disconnect(self, connection_id: str):
        """Remove connection with session summary."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection data by ID."""
        return self.active_connections.get(connection_id)

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_session_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get session statistics for a connection."""
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
        """Reset word tracker for a connection."""
        if connection_id in self.active_connections:
            self.active_connections[connection_id]["word_tracker"].reset_word()
            return True
        return False
