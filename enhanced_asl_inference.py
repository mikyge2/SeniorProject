"""
Enhanced Real-time ASL inference script with modern UI, Google TTS, and improved visual feedback.
Adds circular progress indicators, stylish word completion banners, and Google TTS integration.

Compatible with Python 3.12, TensorFlow Lite, OpenCV, MediaPipe, gTTS, and espeak fallback.
"""

import argparse
import json
import sys
import logging
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, Counter
import threading
import queue
import tempfile
import os
import subprocess
import shutil

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModernProgressBar:
    """Modern circular progress bar for letter hold duration."""

    def __init__(self, center: Tuple[int, int], radius: int = 30, thickness: int = 6):
        """
        Initialize the progress bar.

        Args:
            center: Center point (x, y) of the circle
            radius: Radius of the progress circle
            thickness: Thickness of the progress ring
        """
        self.center = center
        self.radius = radius
        self.thickness = thickness
        self.inner_radius = radius - thickness

    def draw(self, frame: np.ndarray, progress: float, letter: str = "") -> np.ndarray:
        """
        Draw the circular progress bar.

        Args:
            frame: OpenCV frame to draw on
            progress: Progress value from 0.0 to 1.0
            letter: Current letter being held

        Returns:
            Frame with progress bar drawn
        """
        # Clamp progress
        progress = max(0.0, min(1.0, progress))

        # Background circle (dark gray)
        cv2.circle(frame, self.center, self.radius, (50, 50, 50), self.thickness)

        # Progress arc
        if progress > 0:
            # Calculate angle (start from top, go clockwise)
            angle = int(360 * progress)

            # Draw progress arc with gradient effect
            if progress < 1.0:
                # Yellow to green gradient based on progress
                blue = int(50 * (1 - progress))
                green = 255
                red = int(255 * (1 - progress))
                color = (blue, green, red)  # BGR format
            else:
                # Full green when complete
                color = (0, 255, 0)

            # Draw the arc
            if angle > 0:
                # Create points for the arc
                pts = []
                for i in range(angle + 1):
                    angle_rad = math.radians(i - 90)  # Start from top
                    x = int(self.center[0] + self.radius * math.cos(angle_rad))
                    y = int(self.center[1] + self.radius * math.sin(angle_rad))
                    pts.append([x, y])

                if len(pts) > 1:
                    # Draw thick line for arc effect
                    for i in range(len(pts) - 1):
                        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, self.thickness)

        # Center circle with letter
        if letter:
            # Inner circle background
            cv2.circle(frame, self.center, self.inner_radius, (30, 30, 30), -1)
            cv2.circle(frame, self.center, self.inner_radius, (100, 100, 100), 2)

            # Letter text
            text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = self.center[0] - text_size[0] // 2
            text_y = self.center[1] + text_size[1] // 2

            # Text with outline for better visibility
            cv2.putText(frame, letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)  # Black outline
            cv2.putText(frame, letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # White text

        # Progress percentage text
        if progress > 0:
            percent_text = f"{int(progress * 100)}%"
            text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = self.center[0] - text_size[0] // 2
            text_y = self.center[1] + self.radius + 25

            cv2.putText(frame, percent_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame


class WordCompletionBanner:
    """Stylish banner for displaying completed words."""

    def __init__(self, frame_width: int, frame_height: int):
        """Initialize the banner with frame dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.banner_height = 120
        self.banner_y = frame_height // 2 - self.banner_height // 2

    def draw(self, frame: np.ndarray, word: str, flash_progress: float) -> np.ndarray:
        """
        Draw the word completion banner with animation effects.

        Args:
            frame: OpenCV frame to draw on
            word: Completed word to display
            flash_progress: Animation progress from 0.0 to 1.0

        Returns:
            Frame with banner drawn
        """
        if not word or flash_progress <= 0:
            return frame

        # Animation easing (ease out)
        ease_progress = 1 - (1 - flash_progress) ** 3

        # Banner dimensions with animation
        banner_width = int(self.frame_width * 0.8 * ease_progress)
        banner_x = (self.frame_width - banner_width) // 2

        # Animated colors (pulse effect)
        pulse = (math.sin(flash_progress * math.pi * 4) + 1) / 2
        base_color = (20, 20, 20)  # Dark background
        border_color = (int(100 + pulse * 155), int(200 + pulse * 55), int(50 + pulse * 205))

        # Draw banner with rounded rectangle effect
        # Main rectangle
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     base_color, -1)

        # Border with gradient effect
        border_thickness = int(4 + pulse * 2)
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     border_color, border_thickness)

        # Title text
        title = "WORD COMPLETED!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        title_x = banner_x + (banner_width - title_size[0]) // 2
        title_y = self.banner_y + 35

        cv2.putText(frame, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Word text (main focus)
        word_text = word.upper()
        word_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
        word_x = banner_x + (banner_width - word_size[0]) // 2
        word_y = self.banner_y + 85

        # Word with glow effect
        glow_color = (int(pulse * 100), int(pulse * 255), int(pulse * 255))
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, glow_color, 6)  # Glow
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)  # Main text

        return frame


class SimpleFallbackTTS:
    """Ultra-simple fallback TTS using system commands."""

    def __init__(self):
        self.is_working = self._test_system_tts()

    def _test_system_tts(self) -> bool:
        """Test if any system TTS is available."""
        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":  # macOS
                result = subprocess.run(['say', 'test'],
                                      capture_output=True, timeout=3)
                if result.returncode == 0:
                    logger.info("✓ macOS 'say' command available")
                    return True
            elif system == "windows":
                # Windows PowerShell TTS
                ps_cmd = ['powershell', '-Command',
                         'Add-Type -AssemblyName System.Speech; '
                         '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                         '$speak.Speak("test")']
                result = subprocess.run(ps_cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    logger.info("✓ Windows PowerShell TTS available")
                    return True
            elif system == "linux":
                # Try festival
                if shutil.which('festival'):
                    result = subprocess.run(['festival', '--tts'],
                                          input='test', text=True,
                                          capture_output=True, timeout=3)
                    if result.returncode == 0:
                        logger.info("✓ Festival TTS available")
                        return True

                # Try spd-say (speech-dispatcher)
                if shutil.which('spd-say'):
                    result = subprocess.run(['spd-say', 'test'],
                                          capture_output=True, timeout=3)
                    if result.returncode == 0:
                        logger.info("✓ speech-dispatcher available")
                        return True

        except Exception as e:
            logger.debug(f"System TTS test failed: {e}")

        return False

    def speak(self, text: str) -> bool:
        """Speak using system TTS."""
        if not self.is_working or not text.strip():
            return False

        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":  # macOS
                result = subprocess.run(['say', text.strip()],
                                      timeout=10, capture_output=True)
                return result.returncode == 0
            elif system == "windows":
                ps_cmd = ['powershell', '-Command',
                         f'Add-Type -AssemblyName System.Speech; '
                         f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                         f'$speak.Speak("{text.strip()}")']
                result = subprocess.run(ps_cmd, timeout=10, capture_output=True)
                return result.returncode == 0
            elif system == "linux":
                # Try spd-say first
                if shutil.which('spd-say'):
                    result = subprocess.run(['spd-say', text.strip()],
                                          timeout=10, capture_output=True)
                    return result.returncode == 0

                # Try festival
                if shutil.which('festival'):
                    result = subprocess.run(['festival', '--tts'],
                                          input=text.strip(), text=True,
                                          timeout=10, capture_output=True)
                    return result.returncode == 0

        except Exception as e:
            logger.error(f"System TTS failed: {e}")

        return False


class GoogleTTSEngine:
    """Google Text-to-Speech engine with comprehensive fallbacks."""

    def __init__(self, use_google: bool = True, language: str = 'en',
                 slow: bool = False, temp_dir: Optional[str] = None):
        """
        Initialize Google TTS engine with multiple fallbacks.

        Args:
            use_google: Try to use Google TTS first
            language: Language code (e.g., 'en', 'es', 'fr')
            slow: Speak slowly
            temp_dir: Temporary directory for audio files
        """
        self.use_google = use_google
        self.language = language
        self.slow = slow
        self.temp_dir = temp_dir or tempfile.gettempdir()

        self.speech_queue = queue.Queue()
        self.is_running = True
        self.google_available = False
        self.espeak_available = False
        self.system_tts_available = False

        # Initialize fallback TTS
        self.fallback_tts = SimpleFallbackTTS()
        self.system_tts_available = self.fallback_tts.is_working

        # Test TTS engines
        self._test_engines()

        # Start TTS worker thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _test_engines(self):
        """Test availability of TTS engines with detailed diagnostics."""
        logger.info("Testing TTS engines...")

        # Test Google TTS
        if self.use_google:
            try:
                import gtts
                logger.info("✓ gTTS module is installed")

                # Test with a simple phrase
                tts = gtts.gTTS(text="test", lang=self.language, slow=self.slow)
                test_file = os.path.join(self.temp_dir, "tts_test.mp3")
                tts.save(test_file)
                logger.info("✓ Google TTS can generate audio files")

                # Check for Python audio libraries first
                python_audio_available = False

                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.quit()
                    logger.info("✓ pygame audio library available")
                    python_audio_available = True
                except ImportError:
                    logger.debug("pygame not available")
                except Exception as e:
                    logger.debug(f"pygame test failed: {e}")

                if not python_audio_available:
                    try:
                        from pydub import AudioSegment
                        logger.info("✓ pydub audio library available")
                        python_audio_available = True
                    except ImportError:
                        logger.debug("pydub not available")

                if not python_audio_available:
                    try:
                        from playsound import playsound
                        logger.info("✓ playsound audio library available")
                        python_audio_available = True
                    except ImportError:
                        logger.debug("playsound not available")

                # Check system audio players as fallback
                system_players_found = []
                for player in ['mpg123', 'afplay', 'play', 'vlc']:
                    if shutil.which(player):
                        system_players_found.append(player)

                if python_audio_available:
                    self.google_available = True
                    logger.info("✓ Google TTS with Python audio is fully functional")
                elif system_players_found:
                    self.google_available = True
                    logger.info(f"✓ Google TTS with system players: {', '.join(system_players_found)}")
                else:
                    logger.warning("⚠ Google TTS available but no audio playback method found")
                    logger.info("   Install: pip install pygame")
                    logger.info("   Or: sudo apt install mpg123")

                # Clean up test file
                if os.path.exists(test_file):
                    os.remove(test_file)

            except ImportError:
                logger.error("❌ gTTS not installed. Run: pip install gtts")
            except Exception as e:
                logger.error(f"❌ Google TTS test failed: {e}")

        # Test espeak
        try:
            espeak_path = shutil.which('espeak')
            if espeak_path:
                logger.info(f"✓ espeak found at: {espeak_path}")
                result = subprocess.run(['espeak', '--version'],
                                      capture_output=True, timeout=5, text=True)
                if result.returncode == 0:
                    self.espeak_available = True
                    logger.info("✓ espeak is working")
                else:
                    logger.error(f"❌ espeak version check failed: {result.stderr}")
            else:
                logger.info("❌ espeak not found in PATH")
                logger.info("   Ubuntu/Debian: sudo apt install espeak")
                logger.info("   macOS: brew install espeak")
                logger.info("   Windows: Download from http://espeak.sourceforge.net/")
        except Exception as e:
            logger.error(f"❌ espeak test failed: {e}")

        # Final status with recommendations
        if not self.google_available and not self.espeak_available and not self.system_tts_available:
            logger.error("❌ NO TTS ENGINES AVAILABLE!")
            logger.info("QUICK FIX:")
            logger.info("  1. Install pygame: pip install pygame")
            logger.info("  2. OR install espeak: sudo apt install espeak")
            logger.info("  3. OR run with --no-speech to disable TTS")
        elif self.google_available:
            logger.info("🔊 Using Google TTS as primary engine")
        elif self.espeak_available:
            logger.info("🔊 Using espeak as primary engine")
        elif self.system_tts_available:
            logger.info("🔊 Using system TTS as primary engine")

    def _tts_worker(self):
        """TTS worker thread."""
        logger.info("TTS worker thread started")

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text and text.strip():
                    self._speak_text(text.strip())

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    def _speak_text(self, text: str):
        """Speak text using available TTS engine."""
        success = False

        # Try Google TTS first
        if self.google_available and self.use_google:
            success = self._speak_with_google(text)

        # Fallback to espeak
        if not success and self.espeak_available:
            success = self._speak_with_espeak(text)

        if success:
            logger.info(f"🔊 Successfully spoke: '{text}'")
        else:
            logger.error(f"❌ Failed to speak: '{text}'")

    def _speak_with_google(self, text: str) -> bool:
        """Speak using Google TTS with Python audio libraries."""
        try:
            import gtts

            # Create TTS
            tts = gtts.gTTS(text=text, lang=self.language, slow=self.slow)

            # Save to temporary file
            audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")
            tts.save(audio_file)

            # Try multiple Python audio libraries
            success = False

            # Method 1: Try pygame (most reliable)
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                pygame.mixer.quit()
                success = True
                logger.info("Used pygame for audio playback")

            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"pygame failed: {e}")

            # Method 2: Try pydub + simpleaudio
            if not success:
                try:
                    from pydub import AudioSegment
                    from pydub.playback import play

                    audio = AudioSegment.from_mp3(audio_file)
                    play(audio)
                    success = True
                    logger.info("Used pydub for audio playback")

                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"pydub failed: {e}")

            # Method 3: Try playsound
            if not success:
                try:
                    from playsound import playsound
                    playsound(audio_file)
                    success = True
                    logger.info("Used playsound for audio playback")

                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"playsound failed: {e}")

            # Method 4: Fall back to system players
            if not success:
                players = ['mpg123', 'afplay', 'play', 'vlc']
                for player in players:
                    if shutil.which(player):
                        try:
                            if player == 'mpg123':
                                cmd = [player, '-q', audio_file]
                            elif player == 'afplay':
                                cmd = [player, audio_file]
                            elif player == 'vlc':
                                cmd = [player, '--play-and-exit', '--intf', 'dummy', audio_file]
                            else:
                                cmd = [player, '-q', audio_file]

                            result = subprocess.run(cmd, timeout=10, capture_output=True)
                            if result.returncode == 0:
                                success = True
                                logger.info(f"Used {player} for audio playback")
                                break
                        except Exception as e:
                            logger.debug(f"{player} failed: {e}")
                            continue

            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)

            return success

        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            return False

    def _speak_with_espeak(self, text: str) -> bool:
        """Speak using espeak."""
        try:
            cmd = ['espeak', '-s150', '-a80', '-g5', text]
            result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"espeak error: {e}")
            return False

    def speak(self, text: str) -> bool:
        """Queue text for speech."""
        if text and text.strip() and self.is_running:
            try:
                self.speech_queue.put_nowait(text.strip())
                logger.info(f"📝 Queued for speech: '{text.strip()}'")
                return True
            except queue.Full:
                logger.warning("Speech queue full, dropping speech request")
                return False
        return False

    def is_working(self) -> bool:
        """Check if any TTS engine is working."""
        return self.google_available or self.espeak_available

    def stop(self):
        """Stop the TTS engine."""
        logger.info("Stopping TTS engine...")
        self.is_running = False
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)


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
        """Load a comprehensive English word dictionary for predictions."""
        # Expanded common words dictionary
        self.common_words = [
            # Greetings and polite expressions
            "hello", "hi", "goodbye", "bye", "thanks", "thank", "please", "sorry",
            "excuse", "welcome", "nice", "good", "great", "wonderful", "amazing",

            # Basic pronouns and articles
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "hers", "its", "our", "their", "mine", "yours", "ours", "theirs",
            "the", "a", "an", "this", "that", "these", "those", "some", "any", "all", "each", "every",

            # Common verbs
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "will", "would", "could", "should", "might", "may", "can",
            "go", "goes", "went", "going", "come", "comes", "came", "coming", "get", "gets", "got",
            "see", "sees", "saw", "seeing", "look", "looks", "looked", "looking", "hear", "hears",
            "say", "says", "said", "saying", "tell", "tells", "told", "telling", "ask", "asks",
            "know", "knows", "knew", "knowing", "think", "thinks", "thought", "thinking",
            "want", "wants", "wanted", "wanting", "need", "needs", "needed", "needing",
            "like", "likes", "liked", "liking", "love", "loves", "loved", "loving",
            "help", "helps", "helped", "helping", "work", "works", "worked", "working",
            "play", "plays", "played", "playing", "eat", "eats", "ate", "eating",
            "drink", "drinks", "drank", "drinking", "sleep", "sleeps", "slept", "sleeping",
            "walk", "walks", "walked", "walking", "run", "runs", "ran", "running",
            "sit", "sits", "sat", "sitting", "stand", "stands", "stood", "standing",
            "stop", "stops", "stopped", "stopping", "start", "starts", "started", "starting",
            "open", "opens", "opened", "opening", "close", "closes", "closed", "closing",
            "give", "gives", "gave", "giving", "take", "takes", "took", "taking",
            "make", "makes", "made", "making", "use", "uses", "used", "using",
            "find", "finds", "found", "finding", "show", "shows", "showed", "showing",
            "feel", "feels", "felt", "feeling", "seem", "seems", "seemed", "seeming",

            # Common nouns
            "time", "person", "people", "year", "years", "way", "ways", "day", "days",
            "thing", "things", "man", "men", "woman", "women", "child", "children",
            "world", "life", "hand", "hands", "part", "parts", "place", "places",
            "case", "cases", "point", "points", "government", "company", "companies",
            "number", "numbers", "group", "groups", "problem", "problems", "fact", "facts",
            "water", "food", "home", "homes", "house", "houses", "family", "families",
            "friend", "friends", "school", "schools", "work", "job", "jobs", "money",
            "name", "names", "question", "questions", "answer", "answers", "idea", "ideas",
            "book", "books", "room", "rooms", "car", "cars", "door", "doors", "window", "windows",
            "table", "tables", "chair", "chairs", "bed", "beds", "phone", "phones",
            "computer", "computers", "music", "movie", "movies", "game", "games",
            "story", "stories", "picture", "pictures", "color", "colors", "size", "sizes",

            # Common adjectives
            "big", "small", "large", "little", "long", "short", "high", "low", "old", "new",
            "young", "right", "wrong", "good", "bad", "best", "better", "worse", "worst",
            "important", "different", "same", "easy", "hard", "difficult", "simple",
            "happy", "sad", "angry", "tired", "hungry", "thirsty", "sick", "healthy",
            "hot", "cold", "warm", "cool", "fast", "slow", "early", "late", "free", "busy",
            "full", "empty", "clean", "dirty", "quiet", "loud", "dark", "light", "bright",
            "beautiful", "ugly", "strong", "weak", "smart", "stupid", "funny", "serious",

            # Time expressions
            "today", "tomorrow", "yesterday", "now", "then", "before", "after", "later",
            "morning", "afternoon", "evening", "night", "week", "weeks", "month", "months",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",

            # Numbers
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety", "hundred", "thousand", "million", "first", "second", "third",

            # Colors
            "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink",
            "gray", "grey", "silver", "gold", "dark", "light",

            # Body parts
            "head", "face", "eye", "eyes", "ear", "ears", "nose", "mouth", "teeth", "tooth",
            "hand", "hands", "finger", "fingers", "arm", "arms", "leg", "legs", "foot", "feet",
            "body", "hair", "skin", "heart", "mind", "brain",

            # Places
            "house", "home", "room", "kitchen", "bathroom", "bedroom", "office", "school",
            "hospital", "store", "shop", "restaurant", "hotel", "park", "street", "road",
            "city", "town", "country", "state", "world", "america", "europe", "asia",
            "church", "library", "bank", "post", "station", "airport", "bus", "train",

            # Technology and modern words
            "computer", "internet", "website", "email", "phone", "mobile", "app", "software",
            "video", "audio", "camera", "photo", "digital", "online", "social", "media",
            "facebook", "twitter", "google", "youtube", "netflix", "amazon", "apple",

            # Common phrases (as single words for finger spelling)
            "okay", "alright", "maybe", "probably", "definitely", "absolutely", "exactly",
            "especially", "really", "very", "quite", "pretty", "rather", "fairly",
            "always", "never", "sometimes", "often", "usually", "rarely", "seldom",
            "everywhere", "somewhere", "nowhere", "anywhere", "everything", "something",
            "nothing", "anything", "everyone", "someone", "noone", "anyone"
        ]

        # Sort by length for better prefix matching
        self.common_words.sort(key=len)

    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """Get top word suggestions based on partial input."""
        if not partial_word or len(partial_word) < 1:
            return []

        partial_lower = partial_word.lower().strip()
        if not partial_lower:
            return []

        suggestions = []

        # Find words that start with the partial word
        for word in self.common_words:
            if word.startswith(partial_lower) and word != partial_lower:
                suggestions.append(word)
                if len(suggestions) >= max_suggestions:
                    break

        # If we don't have enough suggestions, try fuzzy matching
        if len(suggestions) < max_suggestions and len(partial_lower) > 2:
            # Simple fuzzy matching - find words containing the partial word
            for word in self.common_words:
                if (partial_lower in word and
                    word not in suggestions and
                    word != partial_lower):
                    suggestions.append(word)
                    if len(suggestions) >= max_suggestions:
                        break

        return suggestions[:max_suggestions]

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


class ModernSuggestionsPanel:
    """Modern styled suggestions panel with clean design."""

    def __init__(self, frame_width: int, frame_height: int):
        """Initialize the suggestions panel."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = 280
        self.panel_x = frame_width - self.panel_width - 15
        self.suggestion_height = 45

    def draw(self, frame: np.ndarray, current_word: str, suggestions: List[str]) -> np.ndarray:
        """
        Draw the modern suggestions panel.

        Args:
            frame: OpenCV frame to draw on
            current_word: Currently typed word
            suggestions: List of word suggestions

        Returns:
            Frame with suggestions panel drawn
        """
        if not suggestions or not current_word:
            return frame

        # Calculate panel height
        panel_height = 80 + len(suggestions) * (self.suggestion_height + 8) + 20
        panel_y = 15

        # Main panel background with modern styling
        # Background with rounded corners effect
        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (25, 25, 30), -1)  # Dark background

        # Border with gradient effect
        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (80, 150, 200), 2)  # Blue border

        # Panel title
        title = "WORD SUGGESTIONS"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        title_x = self.panel_x + (self.panel_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 150, 200), 2)

        # Current word display
        current_text = f"'{current_word.upper()}' → "
        cv2.putText(frame, current_text, (self.panel_x + 15, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw suggestion buttons
        for i, suggestion in enumerate(suggestions):
            button_y = panel_y + 70 + i * (self.suggestion_height + 8)

            # Button background with hover effect styling
            button_color = (40, 45, 50)  # Slightly lighter than panel
            border_color = (100, 200, 150)  # Green accent

            # Button rectangle
            cv2.rectangle(frame, (self.panel_x + 10, button_y),
                         (self.panel_x + self.panel_width - 10, button_y + self.suggestion_height),
                         button_color, -1)
            cv2.rectangle(frame, (self.panel_x + 10, button_y),
                         (self.panel_x + self.panel_width - 10, button_y + self.suggestion_height),
                         border_color, 2)

            # Button number
            number_text = f"{i + 1}"
            cv2.putText(frame, number_text, (self.panel_x + 25, button_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)

            # Suggestion text
            suggestion_text = suggestion.upper()
            cv2.putText(frame, suggestion_text, (self.panel_x + 55, button_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


class ASLRealTimeInference:
    """Enhanced Real-time ASL inference with modern UI and Google TTS."""

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0,
                 enable_speech: bool = True, use_google_tts: bool = True,
                 show_landmarks: bool = False):
        """
        Initialize the enhanced ASL inference system.

        Args:
            model_path: Path to the TensorFlow Lite model file
            metadata_path: Path to the metadata JSON file containing class mappings
            camera_index: Camera device index (default: 0)
            enable_speech: Enable text-to-speech output
            use_google_tts: Use Google TTS (with espeak fallback)
            show_landmarks: Show MediaPipe hand landmarks overlay
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.camera_index = camera_index
        self.enable_speech = enable_speech
        self.use_google_tts = use_google_tts
        self.show_landmarks = show_landmarks

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

        # Modern UI components
        self.progress_bar = None
        self.completion_banner = None
        self.suggestions_panel = None

        # TTS Engine
        self.tts_engine = None
        if enable_speech:
            self.tts_engine = GoogleTTSEngine(use_google=use_google_tts)
            # Give TTS time to initialize
            time.sleep(1)

        # Constants
        self.IMAGE_SIZE = (224, 224)
        self.NUM_LANDMARKS = 21
        self.LANDMARK_FEATURES = 42  # 21 landmarks × 2 coordinates (x, y)

        # Display state
        self.last_spoken_word = ""
        self.word_flash_time = 0
        self.flash_duration = 3.0  # Show spoken word for 3 seconds

        # Window size for better UI
        self.window_width = 1280
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
        """Initialize camera capture with optimized settings."""
        try:
            logger.info(f"Initializing camera (index: {self.camera_index})")
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera with index {self.camera_index}")

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Camera initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            sys.exit(1)

    def initialize_ui_components(self, frame_width: int, frame_height: int) -> None:
        """Initialize modern UI components."""
        # Progress bar positioned in top-left area
        progress_center = (100, 100)
        self.progress_bar = ModernProgressBar(progress_center, radius=40, thickness=8)

        # Word completion banner
        self.completion_banner = WordCompletionBanner(frame_width, frame_height)

        # Suggestions panel
        self.suggestions_panel = ModernSuggestionsPanel(frame_width, frame_height)

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
                if 'landmarks' in input_detail['name'].lower():
                    landmarks_idx = input_detail['index']
                elif 'image' in input_detail['name'].lower():
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

    def draw_modern_ui(self, frame: np.ndarray, prediction: str, confidence: float,
                      results, current_word: str, word_finalized: bool,
                      letter_progress: float, word_suggestions: List[str]) -> np.ndarray:
        """
        Draw modern UI with all enhanced components.

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
            Frame with modern UI overlays
        """
        h, w = frame.shape[:2]

        # Initialize UI components if not done
        if self.progress_bar is None:
            self.initialize_ui_components(w, h)

        # Draw hand landmarks if enabled and detected
        if self.show_landmarks and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Main info panel (top-left, below progress bar)
        panel_height = 100
        panel_y = 180
        cv2.rectangle(frame, (10, panel_y), (400, panel_y + panel_height), (20, 25, 30), -1)
        cv2.rectangle(frame, (10, panel_y), (400, panel_y + panel_height), (60, 120, 180), 2)

        # Current prediction with confidence-based coloring
        pred_text = f"Letter: {prediction}"
        conf_text = f"Confidence: {confidence:.2f}"

        # Color based on confidence
        if confidence >= self.word_tracker.confidence_threshold:
            text_color = (0, 255, 0)  # Green for good confidence
        else:
            text_color = (100, 100, 100)  # Gray for low confidence

        cv2.putText(frame, pred_text, (20, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, conf_text, (20, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Current word assembly
        word_color = (100, 200, 255) if current_word else (100, 100, 100)
        word_text = f"Word: {current_word.upper() if current_word else '...'}"
        cv2.putText(frame, word_text, (20, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, word_color, 2)

        # Letter hold progress bar
        if confidence >= self.word_tracker.confidence_threshold and prediction:
            self.progress_bar.draw(frame, letter_progress, prediction)
        else:
            self.progress_bar.draw(frame, 0.0, "")

        # Word suggestions panel
        self.suggestions_panel.draw(frame, current_word, word_suggestions)

        # Word completion banner
        current_time = time.time()
        if word_finalized:
            self.last_spoken_word = current_word
            self.word_flash_time = current_time

        if (self.last_spoken_word and
            current_time - self.word_flash_time < self.flash_duration):
            flash_progress = 1.0 - (current_time - self.word_flash_time) / self.flash_duration
            self.completion_banner.draw(frame, self.last_spoken_word, flash_progress)

        # Status panel (bottom-right)
        stats = self.word_tracker.get_stats()
        status_panel_w, status_panel_h = 320, 140
        status_x = w - status_panel_w - 15
        status_y = h - status_panel_h - 15

        cv2.rectangle(frame, (status_x, status_y),
                     (status_x + status_panel_w, status_y + status_panel_h),
                     (20, 25, 30), -1)
        cv2.rectangle(frame, (status_x, status_y),
                     (status_x + status_panel_w, status_y + status_panel_h),
                     (60, 120, 180), 2)

        # Status text
        total_words_text = f"Total Words: {stats['total_words']}"
        buffer_text = f"Buffer: {stats['buffer_size']}/{self.word_tracker.window_size}"
        landmarks_text = f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}"

        # TTS status
        tts_status = "TTS: Not Available"
        tts_color = (0, 0, 200)

        if self.tts_engine and self.tts_engine.is_working():
            if self.use_google_tts and self.tts_engine.google_available:
                tts_status = "TTS: Google TTS"
                tts_color = (0, 200, 0)
            else:
                tts_status = "TTS: espeak"
                tts_color = (200, 200, 0)
        elif self.tts_engine:
            tts_status = "TTS: Failed"
            tts_color = (0, 0, 200)

        cv2.putText(frame, total_words_text, (status_x + 10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, buffer_text, (status_x + 10, status_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, landmarks_text, (status_x + 10, status_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)
        cv2.putText(frame, tts_status, (status_x + 10, status_y + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 1)

        # Controls (bottom-left)
        controls = [
            "Controls: 'r' = reset word, '1'/'2'/'3' = select suggestion",
            "'l' = toggle landmarks, 'q' = quit",
            "Hold each letter for 2 seconds to add to word"
        ]

        for i, control_text in enumerate(controls):
            cv2.putText(frame, control_text, (15, h - 60 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return frame

    def run(self) -> None:
        """Run the enhanced real-time inference loop with modern UI."""
        logger.info("Starting enhanced ASL inference with modern UI...")
        logger.info("Press 'r' to reset word, '1'/'2'/'3' for suggestions, 'l' for landmarks, 'q' to quit")

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

                # Update word tracker
                current_word, word_finalized, letter_progress = self.word_tracker.add_prediction(
                    prediction, confidence)

                # Get word suggestions
                stats = self.word_tracker.get_stats()
                word_suggestions = stats.get('word_suggestions', [])

                # Draw modern UI
                frame = self.draw_modern_ui(frame, prediction, confidence, results,
                                          current_word, word_finalized, letter_progress,
                                          word_suggestions)

                # Display frame
                cv2.imshow('Enhanced ASL Real-time Inference - Modern UI', frame)

                # Handle word finalization
                if word_finalized and current_word and self.tts_engine:
                    if self.tts_engine.speak(current_word):
                        logger.info(f"Word completed and queued for speech: '{current_word}'")
                    else:
                        logger.error(f"Failed to speak completed word: '{current_word}'")
                    self.word_tracker.reset_word()

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.word_tracker.reset_word()
                    logger.info("Word reset by user")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    logger.info(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('1'):
                    if len(word_suggestions) >= 1:
                        selected_word = self.word_tracker.select_word_suggestion(0)
                        if selected_word and self.tts_engine:
                            if self.tts_engine.speak(selected_word):
                                logger.info(f"Suggestion 1 selected and queued: '{selected_word}'")
                            self.word_tracker.reset_word()
                elif key == ord('2'):
                    if len(word_suggestions) >= 2:
                        selected_word = self.word_tracker.select_word_suggestion(1)
                        if selected_word and self.tts_engine:
                            if self.tts_engine.speak(selected_word):
                                logger.info(f"Suggestion 2 selected and queued: '{selected_word}'")
                            self.word_tracker.reset_word()
                elif key == ord('3'):
                    if len(word_suggestions) >= 3:
                        selected_word = self.word_tracker.select_word_suggestion(2)
                        if selected_word and self.tts_engine:
                            if self.tts_engine.speak(selected_word):
                                logger.info(f"Suggestion 3 selected and queued: '{selected_word}'")
                            self.word_tracker.reset_word()

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


def diagnose_tts_setup():
    """Diagnose TTS setup and provide installation instructions."""
    import platform
    system = platform.system().lower()

    print("\n" + "="*60)
    print("TTS DIAGNOSTIC REPORT")
    print("="*60)

    # Check Python packages
    try:
        import gtts
        print("✓ gTTS Python package: INSTALLED")
        gtts_available = True
    except ImportError:
        print("❌ gTTS Python package: NOT INSTALLED")
        print("   Fix: pip install gtts")
        gtts_available = False

    # Check Python audio libraries
    audio_libs = []

    try:
        import pygame
        audio_libs.append("pygame")
    except ImportError:
        pass

    try:
        from pydub import AudioSegment
        audio_libs.append("pydub")
    except ImportError:
        pass

    try:
        from playsound import playsound
        audio_libs.append("playsound")
    except ImportError:
        pass

    if audio_libs:
        print(f"✓ Python audio libraries: {', '.join(audio_libs)}")
    else:
        print("❌ Python audio libraries: NONE FOUND")
        print("   Fix: pip install pygame  # Recommended")
        print("   Or:  pip install pydub simpleaudio")
        print("   Or:  pip install playsound")

    # Check espeak
    espeak_path = shutil.which('espeak')
    if espeak_path:
        print(f"✓ espeak: FOUND at {espeak_path}")
        espeak_available = True
    else:
        print("❌ espeak: NOT FOUND")
        print("   Container fix: apt update && apt install espeak")
        if system == "linux":
            print("   System fix: sudo apt install espeak")
        elif system == "darwin":
            print("   System fix: brew install espeak")
        espeak_available = False

    # Check audio players for Google TTS (system level)
    players = ['mpg123', 'afplay', 'play', 'vlc']
    found_players = [p for p in players if shutil.which(p)]

    if found_players:
        print(f"✓ System audio players: {', '.join(found_players)}")
    else:
        print("❌ System audio players: NONE FOUND")
        print("   Container fix: apt update && apt install mpg123")
        if system == "linux":
            print("   System fix: sudo apt install mpg123")

    # Check system TTS
    system_tts = SimpleFallbackTTS()
    if system_tts.is_working:
        print("✓ System TTS: AVAILABLE")
    else:
        print("❌ System TTS: NOT AVAILABLE")

    print("="*60)

    # Recommendations for container environment
    total_options = 0
    working_options = []

    if gtts_available and audio_libs:
        total_options += 1
        working_options.append("Google TTS + Python audio")

    if espeak_available:
        total_options += 1
        working_options.append("espeak")

    if system_tts.is_working:
        total_options += 1
        working_options.append("System TTS")

    if total_options == 0:
        print("RECOMMENDATION: Install Python audio library (easiest for containers)")
        print("Quick fixes (choose one):")
        print("1. pip install pygame              # Best for containers")
        print("2. pip install pydub simpleaudio   # Alternative")
        print("3. apt update && apt install espeak # If you have apt access")
        print("4. Run without speech: --no-speech")
    else:
        print(f"RECOMMENDATION: {len(working_options)} TTS option(s) available")
        for i, option in enumerate(working_options, 1):
            print(f"{i}. {option}")

    print("="*60 + "\n")


def main():
    """Main function with enhanced argument parsing and TTS diagnostics."""
    parser = argparse.ArgumentParser(
        description="Enhanced real-time ASL inference with modern UI and Google TTS",
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
        '--no-google-tts',
        action='store_true',
        help='Disable Google TTS, use only espeak'
    )

    parser.add_argument(
        '--show-landmarks',
        action='store_true',
        help='Show MediaPipe hand landmarks overlay at startup'
    )

    parser.add_argument(
        '--diagnose-tts',
        action='store_true',
        help='Run TTS diagnostic and exit'
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

    parser.add_argument(
        '--letter-hold-time',
        type=float,
        default=3.0,
        help='Time (seconds) to hold each letter before acceptance (default: 2.0)'
    )

    args = parser.parse_args()

    # Run TTS diagnostic if requested
    if args.diagnose_tts:
        diagnose_tts_setup()
        sys.exit(0)

    # Validate arguments
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Validate numeric arguments
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        logger.error("Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if args.letter_hold_time < 0.5:
        logger.error("Letter hold time must be at least 0.5 seconds")
        sys.exit(1)

    # Run TTS diagnostic automatically if speech is enabled
    if not args.no_speech:
        diagnose_tts_setup()

    # Create and run enhanced inference system
    try:
        logger.info("=" * 60)
        logger.info("ENHANCED ASL REAL-TIME INFERENCE WITH MODERN UI")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Metadata: {args.metadata}")
        logger.info(f"Camera: {args.camera}")
        logger.info(f"Speech: {'Disabled' if args.no_speech else 'Enabled'}")
        logger.info(f"Google TTS: {'Disabled' if args.no_google_tts else 'Enabled'}")
        logger.info(f"Show Landmarks: {args.show_landmarks}")
        logger.info(f"Confidence Threshold: {args.confidence_threshold}")
        logger.info(f"Letter Hold Time: {args.letter_hold_time}s")
        logger.info("=" * 60)

        inference_system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera,
            enable_speech=not args.no_speech,
            use_google_tts=not args.no_google_tts,
            show_landmarks=args.show_landmarks
        )

        # Configure word tracker with custom parameters
        inference_system.word_tracker = WordTracker(
            window_size=args.window_size,
            confidence_threshold=args.confidence_threshold,
            pause_threshold=args.pause_threshold,
            min_letter_duration=args.letter_hold_time
        )

        # Initialize all components
        inference_system.load_model()
        inference_system.load_metadata()
        inference_system.initialize_mediapipe()
        inference_system.initialize_camera()

        # Display startup information
        logger.info("SYSTEM READY!")
        logger.info("CONTROLS:")
        logger.info("  - Hold each letter for 2+ seconds to add to word")
        logger.info("  - Press '1', '2', or '3' to select word suggestions")
        logger.info("  - Press 'r' to reset current word")
        logger.info("  - Press 'l' to toggle hand landmarks display")
        logger.info("  - Press 'q' to quit")
        logger.info("=" * 60)

        # Run enhanced inference
        inference_system.run()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to initialize enhanced inference system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()