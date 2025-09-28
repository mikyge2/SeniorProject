"""
Enhanced ASL Inference Script with Amharic Translation
Clean, working implementation with proper class organization
"""

import argparse
import json
import sys
import logging
import time
import math
import signal
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


class OnlineDictionaryAPI:
    """Provides word suggestions by querying an online dictionary API.

    This class handles fetching word suggestions for partially completed words.
    It includes a caching mechanism to reduce redundant API calls and a
    fallback list of common words if the API is unavailable.

    Attributes:
        api_available (bool): True if the online dictionary API is accessible.
        cache (Dict[str, List[str]]): A cache to store API results.
        cache_max_size (int): The maximum number of items to store in the cache.
    """

    def __init__(self):
        """Initializes the OnlineDictionaryAPI and tests API availability."""
        self.api_available = self._test_api()
        self.cache = {}
        self.cache_max_size = 200

    def _test_api(self) -> bool:
        """Tests the availability of the Datamuse online dictionary API.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        try:
            import requests
            response = requests.get(
                "https://api.datamuse.com/words?sp=hel*&max=5",
                timeout=3
            )
            if response.status_code == 200:
                logger.info("Online dictionary API available")
                return True
        except ImportError:
            logger.warning("requests library not installed: pip install requests")
        except Exception as e:
            logger.warning(f"Online dictionary API not available: {e}")
        return False

    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """Fetches word suggestions for a given partial word.

        Queries the Datamuse API for words starting with `partial_word`.
        Results are cached. If the API is down or returns no results, a
        fallback list of common English words is used.

        Args:
            partial_word: The partially spelled word to get suggestions for.
            max_suggestions: The maximum number of suggestions to return.

        Returns:
            A list of suggested words, or an empty list if none are found.
        """
        if not partial_word or len(partial_word) < 2:
            return []

        partial_lower = partial_word.lower().strip()
        cache_key = f"{partial_lower}_{max_suggestions}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        suggestions = []

        if self.api_available:
            try:
                import requests
                url = "https://api.datamuse.com/words"
                params = {
                    'sp': f"{partial_lower}*",
                    'max': max_suggestions * 2,
                    'md': 'f'
                }

                response = requests.get(url, params=params, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        word = item.get('word', '').lower()
                        if (word.startswith(partial_lower) and
                            word != partial_lower and
                            len(word) <= 15 and
                            word.isalpha()):
                            suggestions.append(word)
                            if len(suggestions) >= max_suggestions:
                                break
            except Exception as e:
                logger.debug(f"API request failed: {e}")

        # Fallback suggestions
        if not suggestions:
            fallback_words = [
                "hello", "help", "home", "house", "happy", "hand", "head", "heart",
                "good", "great", "green", "go", "get", "give", "girl", "game",
                "work", "water", "want", "walk", "watch", "window", "word", "world",
                "love", "life", "light", "learn", "look", "live", "like", "little",
                "thank", "think", "time", "today", "tomorrow", "table", "talk", "take",
                "family", "friend", "food", "feel", "fast", "father", "first", "find",
                "school", "see", "say", "sit", "stand", "stop", "start", "small",
                "mother", "make", "man", "more", "much", "music", "money", "move",
                "book", "boy", "big", "be", "beautiful", "best", "blue", "black",
                "come", "can", "call", "child", "clean", "close", "color", "cold"
            ]

            for word in fallback_words:
                if word.startswith(partial_lower) and word != partial_lower:
                    suggestions.append(word)
                    if len(suggestions) >= max_suggestions:
                        break

        # Cache management
        if len(self.cache) >= self.cache_max_size:
            items_to_remove = list(self.cache.keys())[:50]
            for key in items_to_remove:
                del self.cache[key]

        self.cache[cache_key] = suggestions
        return suggestions


class WordTracker:
    """Tracks the sequence of letters to form words based on predictions.

    This class manages the logic for building words from a stream of individual
    letter predictions. It uses a sliding window to determine a "stable" letter,
    requires a letter to be held for a minimum duration before being appended
    to the current word, and finalizes words after a pause in confident
    predictions.

    Attributes:
        window_size (int): The number of recent predictions to consider for stability.
        confidence_threshold (float): The minimum confidence for a prediction to be considered valid.
        pause_threshold (float): The duration of inactivity (in seconds) to finalize a word.
        min_letter_duration (float): The time a stable letter must be held before being added to the word.
        prediction_buffer (deque): A buffer holding recent predictions.
        current_word (str): The word currently being formed.
        current_letter (str): The stable letter currently being held.
        dictionary (OnlineDictionaryAPI): An instance for fetching word suggestions.
    """

    def __init__(self, window_size: int = 6, confidence_threshold: float = 0.6,
                 pause_threshold: float = 2.0, min_letter_duration: float = 0.8):
        """Initializes the WordTracker with configurable parameters.

        Args:
            window_size: The size of the prediction buffer.
            confidence_threshold: The minimum prediction confidence to accept.
            pause_threshold: The pause duration (in seconds) to finalize a word.
            min_letter_duration: The time (in seconds) a letter must be held.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.pause_threshold = pause_threshold
        self.min_letter_duration = min_letter_duration

        self.prediction_buffer = deque(maxlen=window_size)
        self.current_word = ""
        self.current_letter = ""
        self.current_letter_start = time.time()
        self.last_prediction_time = time.time()
        self.word_finalized = False
        self.letter_hold_progress = 0.0

        self.total_words = 0
        self.recognized_words = []

        self.dictionary = OnlineDictionaryAPI()

    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """Gets word suggestions for the current partial word.

        Args:
            partial_word: The word to get suggestions for.
            max_suggestions: The maximum number of suggestions to return.

        Returns:
            A list of suggested words.
        """
        return self.dictionary.get_word_suggestions(partial_word, max_suggestions)

    def add_prediction(self, letter: str, confidence: float) -> Tuple[str, bool, float]:
        """Processes a new letter prediction to update the current word.

        This is the core method of the tracker. It updates the internal state
        based on the new prediction, checks for stable letters, manages the
        hold timer, and determines if a word has been finalized.

        Args:
            letter: The predicted letter.
            confidence: The confidence score of the prediction.

        Returns:
            A tuple containing:
            - The current word being formed.
            - A boolean indicating if the word was just finalized.
            - A float from 0.0 to 1.0 representing the progress of holding the current letter.
        """
        current_time = time.time()
        word_finalized = False

        if confidence < self.confidence_threshold:
            if current_time - self.last_prediction_time > self.pause_threshold:
                if self.current_word and not self.word_finalized:
                    word_finalized = True
                    self.word_finalized = True
                    self.recognized_words.append(self.current_word)
                    self.total_words += 1

            self.letter_hold_progress = 0.0
            return self.current_word, word_finalized, self.letter_hold_progress

        self.last_prediction_time = current_time
        self.word_finalized = False
        self.prediction_buffer.append((letter, confidence, current_time))

        stable_letter = self._get_stable_letter()

        if stable_letter == self.current_letter and stable_letter:
            hold_duration = current_time - self.current_letter_start
            self.letter_hold_progress = min(1.0, hold_duration / self.min_letter_duration)

            if hold_duration >= self.min_letter_duration:
                self.current_word += stable_letter.lower()
                logger.info(f"Letter '{stable_letter}' added: '{self.current_word}'")

                self.current_letter = ""
                self.current_letter_start = current_time
                self.letter_hold_progress = 0.0

        elif stable_letter != self.current_letter:
            self.current_letter = stable_letter
            self.current_letter_start = current_time
            self.letter_hold_progress = 0.0

        return self.current_word, word_finalized, self.letter_hold_progress

    def _get_stable_letter(self) -> str:
        """Determines the most stable letter from the prediction buffer.

        It calculates a weighted score for each letter in the buffer, giving
        more weight to recent and high-confidence predictions.

        Returns:
            The letter with the highest score, or an empty string if the
            buffer is empty.
        """
        if not self.prediction_buffer:
            return ""

        letter_scores = Counter()
        current_time = time.time()

        for letter, confidence, timestamp in self.prediction_buffer:
            time_weight = max(0.1, 1.0 - (current_time - timestamp) / 2.0)
            score = confidence * time_weight
            letter_scores[letter] += score

        return letter_scores.most_common(1)[0][0] if letter_scores else ""

    def auto_complete_word(self) -> Optional[str]:
        """Finalizes and returns the current word.

        Returns:
            The completed word, or None if there is no current word.
        """
        if self.current_word and not self.word_finalized:
            self.recognized_words.append(self.current_word)
            self.total_words += 1
            self.word_finalized = True
            logger.info(f"Auto-completed: '{self.current_word}'")
            return self.current_word
        return None

    def select_suggestion(self, index: int) -> Optional[str]:
        """Selects a word from the suggestions list to be the new current word.

        Args:
            index: The 0-based index of the suggestion to select.

        Returns:
            The selected word, or None if the index is invalid.
        """
        suggestions = self.get_word_suggestions(self.current_word)
        if 0 <= index < len(suggestions):
            selected = suggestions[index]
            self.current_word = selected
            self.recognized_words.append(selected)
            self.total_words += 1
            return selected
        return None

    def reset_word(self):
        """Resets the state to begin tracking a new word."""
        self.current_word = ""
        self.word_finalized = False
        self.current_letter = ""
        self.current_letter_start = time.time()
        self.letter_hold_progress = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Returns a dictionary of the current tracking statistics.

        Returns:
            A dictionary containing stats like total words, current word,
            and recent words.
        """
        return {
            "total_words": self.total_words,
            "current_word": self.current_word,
            "current_letter": self.current_letter,
            "letter_progress": self.letter_hold_progress,
            "word_suggestions": self.get_word_suggestions(self.current_word),
            "recent_words": self.recognized_words[-5:],
            "buffer_size": len(self.prediction_buffer)
        }


class SimpleAmharicTranslator:
    """A simple wrapper for translating English text to Amharic.

    This class uses the `deep_translator` library (with a fallback to
    `googletrans`) to provide translations. It includes a cache to avoid
    re-translating the same text.

    Attributes:
        use_translation (bool): Flag to enable or disable translation.
        translator: The translator object from the backing library.
        translation_available (bool): True if a translation service was successfully initialized.
        translation_cache (Dict[str, str]): A cache for translations.
    """

    def __init__(self, use_translation: bool = False):
        """Initializes the translator.

        Args:
            use_translation: If True, attempts to initialize a translation service.
        """
        self.use_translation = use_translation
        self.translator = None
        self.translation_available = False
        self.translation_cache = {}

        if use_translation:
            self._initialize_translator()

    def _initialize_translator(self):
        """Initializes a translation service, trying `deep_translator` first."""
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source='en', target='am')
            self.translation_method = "deep_translator"

            test_result = self.translator.translate("hello")
            if test_result:
                self.translation_available = True
                logger.info("Deep Translator (Google) initialized successfully")
                return
        except ImportError:
            logger.debug("deep-translator not available")
        except Exception as e:
            logger.debug(f"Deep translator failed: {e}")

        try:
            from googletrans import Translator
            self.translator = Translator()
            self.translation_method = "googletrans"

            logger.info("Testing Google Translate connection...")
            test_result = self.translator.translate("hello", dest='am')
            if test_result and test_result.text:
                self.translation_available = True
                logger.info("Google Translate (googletrans) initialized successfully")
                return
            else:
                logger.error("Google Translate test failed - no response")
        except ImportError as e:
            logger.error("googletrans not available - install with: pip install deep-translator")
        except Exception as e:
            logger.error(f"Google Translate initialization failed: {e}")

        logger.error("No translation service available")
        logger.info("Install with: pip install deep-translator")

    def translate(self, text: str) -> Optional[str]:
        """Translates English text to Amharic.

        Args:
            text: The English text to translate.

        Returns:
            The Amharic translation as a string, or None if translation fails
            or is disabled.
        """
        if not self.use_translation or not self.translation_available or not text.strip():
            return None

        text_lower = text.lower().strip()

        if text_lower in self.translation_cache:
            logger.debug(f"Using cached translation for '{text_lower}'")
            return self.translation_cache[text_lower]

        try:
            logger.debug(f"Translating '{text_lower}' to Amharic...")

            if hasattr(self, 'translation_method') and self.translation_method == "deep_translator":
                translation = self.translator.translate(text_lower)
            else:
                result = self.translator.translate(text_lower, dest='am')
                translation = result.text if result and result.text else None

            if translation:
                self.translation_cache[text_lower] = translation
                logger.debug(f"Translation successful: '{text_lower}' -> '{translation}'")
                return translation
            else:
                logger.warning(f"No translation result for '{text_lower}'")

        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")

        return None


class SimpleTTS:
    """A simple, non-blocking Text-to-Speech (TTS) engine.

    This class uses gTTS (Google Text-to-Speech) to generate audio and plays
    it back using either Pygame or a system command-line player (like mpg123).
    Speech requests are handled in a separate thread to prevent blocking the
    main application loop.

    Attributes:
        use_google (bool): Flag to enable Google TTS.
        google_available (bool): True if gTTS and a playback method are available.
        speech_queue (queue.Queue): A queue to hold pending speech requests.
        is_running (bool): A flag to control the worker thread.
    """

    def __init__(self, use_google: bool = True):
        """Initializes the TTS engine and starts the worker thread.

        Args:
            use_google: If True, attempts to use Google TTS.
        """
        self.use_google = use_google
        self.google_available = False
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.temp_dir = tempfile.gettempdir()

        if use_google:
            self._test_google_tts()

        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _test_google_tts(self):
        """Checks for the availability of gTTS and a suitable audio player."""
        try:
            import gtts
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.quit()
                self.google_available = True
                logger.info("Google TTS with pygame available")
                return
            except ImportError:
                pass

            if any(shutil.which(p) for p in ['mpg123', 'afplay']):
                self.google_available = True
                logger.info("Google TTS with system audio available")

        except ImportError:
            logger.info("gTTS not installed")

    def _tts_worker(self):
        """The worker method that runs in a separate thread to process speech requests."""
        while self.is_running:
            try:
                text, lang = self.speech_queue.get(timeout=1.0)
                if text:
                    self._speak_with_google(text, lang)
            except queue.Empty:
                continue

    def _romanize_amharic(self, amharic_text: str) -> str:
        """Converts Amharic text to a romanized (Latin script) form for TTS.

        This is a workaround because gTTS does not have a native Amharic voice.
        It uses a simple dictionary lookup for common words.

        Args:
            amharic_text: The Amharic text to convert.

        Returns:
            A romanized string that can be pronounced by an English TTS voice.
        """
        # Simple Amharic to romanized mapping for common words
        amharic_romanized = {
            'ሰላም': 'selam',
            'ደህና': 'dehna',
            'እንደምን': 'endemin',
            'አመሰግናለሁ': 'ameseginalew',
            'ዓላማ': 'alama',
            'ማዕከል': 'makel',
            'መጽሐፍ': 'metshaf',
            'ሰዓት': 'seat',
            'ቤት': 'bet',
            'ውሃ': 'wuha',
            'ቃል': 'kal',
            'ሰው': 'sew',
            'እጅ': 'ej',
            'ዓይን': 'ayn',
            'ልብ': 'lib',
            'ቀን': 'ken',
            'ለሊት': 'lelit',
            'ምግብ': 'migib',
            'መጠጥ': 'metat',
            'አደራጅ': 'aderaj'
        }

        return amharic_romanized.get(amharic_text, amharic_text)

    def _speak_with_google(self, text: str, lang: str = 'en') -> bool:
        """Generates and plays audio for the given text using Google TTS.

        For Amharic, it first romanizes the text. The generated MP3 file is
        saved to a temporary directory and played back in a non-blocking manner.

        Args:
            text: The text to speak.
            lang: The language of the text ('en' or 'am').

        Returns:
            True if the speech was successfully initiated, False otherwise.
        """
        if not self.google_available:
            return False

        try:
            import gtts

            # Handle Amharic by converting to romanized English pronunciation
            if lang == 'am':
                romanized = self._romanize_amharic(text)
                logger.info(f"Speaking Amharic '{text}' as '{romanized}'")
                tts = gtts.gTTS(text=romanized, lang='en', slow=True)  # Slower for clarity
            else:
                tts = gtts.gTTS(text=text, lang=lang)

            audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time() * 1000)}.mp3")
            tts.save(audio_file)

            # Non-blocking audio playback
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                # Don't block - just start playing and clean up later
                threading.Timer(5.0, lambda: self._cleanup_audio_file(audio_file)).start()
                return True
            except ImportError:
                pass

            # Try system players in background
            for player in ['mpg123', 'afplay']:
                if shutil.which(player):
                    try:
                        # Run in background, don't wait
                        subprocess.Popen([player, audio_file],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        threading.Timer(5.0, lambda: self._cleanup_audio_file(audio_file)).start()
                        return True
                    except Exception:
                        continue

            os.remove(audio_file)
        except Exception as e:
            logger.error(f"TTS error: {e}")

        return False

    def _cleanup_audio_file(self, file_path: str):
        """Deletes a temporary audio file after a delay.

        Args:
            file_path: The path to the audio file to delete.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    def speak(self, text: str, lang: str = 'en') -> bool:
        """Adds a text-to-speech request to the processing queue.

        Args:
            text: The text to be spoken.
            lang: The language of the text ('en' or 'am').

        Returns:
            True if the request was successfully queued, False if the queue is full.
        """
        try:
            self.speech_queue.put_nowait((text, lang))
            return True
        except queue.Full:
            return False

    def stop(self):
        """Stops the TTS worker thread."""
        self.is_running = False


class ModernProgressBar:
    """A UI component to draw a circular progress bar.

    This is used to visualize the hold duration for the current stable letter.

    Attributes:
        center (Tuple[int, int]): The (x, y) coordinates for the center of the bar.
        radius (int): The outer radius of the progress bar.
        thickness (int): The thickness of the progress bar ring.
    """

    def __init__(self, center: Tuple[int, int], radius: int = 35, thickness: int = 8):
        """Initializes the progress bar's dimensions and position.

        Args:
            center: The (x, y) coordinates for the center of the bar.
            radius: The outer radius.
            thickness: The thickness of the bar.
        """
        self.center = center
        self.radius = radius
        self.thickness = thickness
        self.inner_radius = radius - thickness

    def draw(self, frame: np.ndarray, progress: float, letter: str = "") -> np.ndarray:
        """Draws the progress bar onto a frame.

        Args:
            frame: The OpenCV image (numpy array) to draw on.
            progress: The progress value from 0.0 to 1.0.
            letter: The letter to display in the center of the bar.

        Returns:
            The frame with the progress bar drawn on it.
        """
        progress = max(0.0, min(1.0, progress))

        cv2.circle(frame, self.center, self.radius, (40, 40, 40), self.thickness)

        if progress > 0:
            if progress < 0.3:
                color = (0, 100, 255)
            elif progress < 0.7:
                color = (0, 200, 255)
            else:
                color = (0, 255, 100)

            angle = int(360 * progress)
            axes = (self.radius, self.radius)
            cv2.ellipse(frame, self.center, axes, -90, 0, angle, color, self.thickness)

        if letter:
            pulse = 0.7 + 0.3 * math.sin(time.time() * 5)
            inner_color = (int(25 * pulse), int(25 * pulse), int(35 * pulse))

            cv2.circle(frame, self.center, self.inner_radius, inner_color, -1)
            cv2.circle(frame, self.center, self.inner_radius, (80, 80, 100), 2)

            font_scale = 1.0 if len(letter) == 1 else 0.7
            text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = self.center[0] - text_size[0] // 2
            text_y = self.center[1] + text_size[1] // 2

            cv2.putText(frame, letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        return frame


class WordCompletionBanner:
    """A UI component to display an animated banner when a word is completed.

    This banner shows the completed English word and its Amharic translation.
    It uses the Pillow (PIL) library to render Amharic text correctly, as
    OpenCV's default fonts do not support the script well.

    Attributes:
        frame_width (int): The width of the video frame.
        frame_height (int): The height of the video frame.
        pil_available (bool): True if the Pillow library is installed.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """Initializes the banner's dimensions.

        Args:
            frame_width: The width of the target frame.
            frame_height: The height of the target frame.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.banner_height = 140
        self.banner_y = frame_height // 2 - self.banner_height // 2
        self.pil_available = self._test_pil()

    def _test_pil(self) -> bool:
        """Checks if the Pillow library is available.

        Returns:
            True if Pillow can be imported, False otherwise.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            return True
        except ImportError:
            logger.info("PIL not available - Amharic text may not display correctly")
            logger.info("Install with: pip install Pillow")
            return False

    def _draw_amharic_with_pil(self, frame: np.ndarray, amharic_text: str,
                              x: int, y: int, width: int) -> np.ndarray:
        """Draws Amharic text onto a frame using the Pillow library.

        This method is necessary for correct rendering of Unicode scripts like
        Amharic. It searches for a suitable system font to use.

        Args:
            frame: The OpenCV frame to draw on.
            amharic_text: The Amharic text to render.
            x: The starting x-coordinate for the text area.
            y: The starting y-coordinate for the text area.
            width: The width of the text area for centering.

        Returns:
            The frame with the Amharic text rendered on it.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            # Create PIL image from the current frame
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # Try to find a suitable font that supports Unicode
            font = None
            font_size = 28  # Larger for better visibility

            # Font search order - prioritize fonts with good Unicode support
            font_paths = [
                # Windows fonts
                "C:/Windows/Fonts/seguisym.ttf",  # Good Unicode support
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                # macOS fonts
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                # Linux fonts
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"  # Excellent Unicode
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        # Test if the font can render Amharic by trying to get bbox
                        test_bbox = draw.textbbox((0, 0), amharic_text, font=font)
                        if test_bbox[2] > 0:  # Width > 0 means it can render
                            break
                    except Exception:
                        continue

            if font is None:
                try:
                    # Try default font as last resort
                    font = ImageFont.load_default()
                except Exception:
                    # If even default fails, draw placeholder
                    cv2.putText(frame, "[Amharic text - install Noto fonts]",
                              (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
                    return frame

            # Calculate text position for centering
            try:
                bbox = draw.textbbox((0, 0), amharic_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except Exception:
                # Fallback for older PIL versions
                try:
                    text_width, text_height = draw.textsize(amharic_text, font=font)
                except:
                    text_width, text_height = 100, 25  # Fallback dimensions

            text_x = x + (width - text_width) // 2
            text_y = y

            # Ensure coordinates are positive
            text_x = max(0, text_x)
            text_y = max(0, text_y)

            # Draw text with outline for better visibility
            outline_color = (0, 0, 0)  # Black outline
            text_color = (100, 255, 150)  # Light green

            # Draw outline (multiple positions for thickness)
            outline_positions = [(-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2)]
            for dx, dy in outline_positions:
                draw.text((text_x+dx, text_y+dy), amharic_text, font=font, fill=outline_color)

            # Draw main text
            draw.text((text_x, text_y), amharic_text, font=font, fill=text_color)

            # Convert back to OpenCV format
            frame_rgb = np.array(pil_img)
            converted_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Verify the conversion worked
            if converted_frame.shape == frame.shape:
                return converted_frame
            else:
                logger.debug("Frame conversion size mismatch, using original")
                return frame

        except Exception as e:
            logger.debug(f"PIL text rendering failed: {e}")
            # Fallback to OpenCV with transliterated text
            transliterated = self._transliterate_amharic(amharic_text)
            cv2.putText(frame, f"Amharic: {transliterated}",
                      (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)
            return frame

    def _transliterate_amharic(self, amharic_text: str) -> str:
        """Converts Amharic text to a Latin script transliteration.

        This is a fallback for display purposes if PIL is not available.

        Args:
            amharic_text: The Amharic text to transliterate.

        Returns:
            A string containing the Latin-script representation.
        """
        transliteration_map = {
            'ሰላም': 'selam', 'ደህና': 'dehna', 'እንደምን': 'endemin',
            'አመሰግናለሁ': 'amesegnalew', 'ዓላማ': 'alama', 'ማዕከል': 'ma\'ekel',
            'መጽሐፍ': 'metshaf', 'ሰዓት': 'se\'at', 'ቤት': 'bet', 'ውሃ': 'wuha',
            'ቃል': 'qal', 'ሰው': 'sew', 'እጅ': 'ej', 'ዓይን': 'ayin', 'ልብ': 'leb',
            'ቀን': 'qen', 'ለሊት': 'lelit', 'ምግብ': 'megeb', 'መጠጥ': 'metat'
        }
        return transliteration_map.get(amharic_text, amharic_text)

    def draw(self, frame: np.ndarray, word: str, flash_progress: float,
             amharic_translation: Optional[str] = None) -> np.ndarray:
        """Draws the animated word completion banner on the frame.

        Args:
            frame: The OpenCV frame to draw on.
            word: The completed English word.
            flash_progress: A value from 1.0 to 0.0 controlling the banner's
                animation (e.g., fade-out).
            amharic_translation: The optional Amharic translation to display.

        Returns:
            The frame with the banner drawn on it.
        """
        if not word or flash_progress <= 0:
            return frame

        ease_progress = 1 - (1 - flash_progress) ** 2
        banner_width = int(self.frame_width * 0.8 * ease_progress)
        banner_x = (self.frame_width - banner_width) // 2

        pulse = (math.sin(flash_progress * math.pi * 6) + 1) / 2
        bg_color = (15, 15, 25)
        border_color = (int(50 + pulse * 150), int(150 + pulse * 50), int(200 + pulse * 55))

        # Main banner
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     bg_color, -1)

        border_thickness = int(3 + pulse * 2)
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     border_color, border_thickness)

        # Title
        title = "WORD COMPLETED!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_x = banner_x + (banner_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, self.banner_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # English word
        word_text = word.upper()
        word_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        word_x = banner_x + (banner_width - word_size[0]) // 2
        word_y = self.banner_y + 60

        glow_color = (int(pulse * 150), int(pulse * 255), int(pulse * 200))
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, glow_color, 5)
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Amharic translation
        if amharic_translation:
            amharic_y = self.banner_y + 95

            if self.pil_available:
                # Use PIL for proper Amharic rendering
                frame = self._draw_amharic_with_pil(frame, amharic_translation,
                                                  banner_x, amharic_y, banner_width)
            else:
                # Fallback: show placeholder text
                placeholder = "[Amharic: Install 'pip install Pillow' for display]"
                placeholder_size = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                placeholder_x = banner_x + (banner_width - placeholder_size[0]) // 2

                cv2.putText(frame, placeholder, (placeholder_x, amharic_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            # Add label
            label = "(Amharic Translation)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = banner_x + (banner_width - label_size[0]) // 2
            cv2.putText(frame, label, (label_x, self.banner_y + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        return frame


class SuggestionsPanel:
    """A UI component to display word suggestions.

    This panel shows the current word being formed and a list of clickable
    suggestions fetched from the `OnlineDictionaryAPI`.
    """

    def __init__(self, frame_width: int, frame_height: int):
        """Initializes the panel's dimensions and position.

        Args:
            frame_width: The width of the target frame.
            frame_height: The height of the target frame.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = 280
        self.panel_x = frame_width - self.panel_width - 10
        self.button_height = 40

    def draw(self, frame: np.ndarray, current_word: str, suggestions: List[str]) -> np.ndarray:
        """Draws the suggestions panel onto a frame.

        Args:
            frame: The OpenCV frame to draw on.
            current_word: The current word being typed.
            suggestions: A list of word suggestions to display.

        Returns:
            The frame with the panel drawn on it.
        """
        if not current_word:
            return frame

        panel_height = 100 + len(suggestions) * (self.button_height + 5) + 10
        panel_y = 10

        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (20, 20, 30), -1)
        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (100, 150, 200), 2)

        cv2.putText(frame, "WORD OPTIONS", (self.panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 2)

        cv2.putText(frame, f"'{current_word.upper()}'", (self.panel_x + 10, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        button_y = panel_y + 60
        cv2.rectangle(frame, (self.panel_x + 5, button_y),
                     (self.panel_x + self.panel_width - 5, button_y + self.button_height),
                     (40, 80, 40), -1)
        cv2.rectangle(frame, (self.panel_x + 5, button_y),
                     (self.panel_x + self.panel_width - 5, button_y + self.button_height),
                     (80, 255, 80), 2)

        cv2.putText(frame, "0", (self.panel_x + 15, button_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
        cv2.putText(frame, f"COMPLETE: {current_word.upper()}", (self.panel_x + 35, button_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, suggestion in enumerate(suggestions):
            button_y = panel_y + 110 + i * (self.button_height + 5)

            cv2.rectangle(frame, (self.panel_x + 5, button_y),
                         (self.panel_x + self.panel_width - 5, button_y + self.button_height),
                         (40, 50, 60), -1)
            cv2.rectangle(frame, (self.panel_x + 5, button_y),
                         (self.panel_x + self.panel_width - 5, button_y + self.button_height),
                         (100, 200, 150), 2)

            cv2.putText(frame, str(i + 1), (self.panel_x + 15, button_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 150), 2)
            cv2.putText(frame, suggestion.upper(), (self.panel_x + 35, button_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame


class ASLRealTimeInference:
    """Orchestrates the real-time ASL recognition system.

    This is the main class that integrates all other components:
    - Loads the TFLite model and metadata.
    - Initializes MediaPipe for hand tracking.
    - Captures video from the camera.
    - Processes each frame to run inference.
    - Uses `WordTracker` to build words from predictions.
    - Handles user input for controls.
    - Manages translation and TTS.
    - Draws the complete UI with all components.

    Attributes:
        model_path (Path): Path to the TFLite model file.
        metadata_path (Path): Path to the class metadata JSON file.
        interpreter: The TFLite interpreter instance.
        word_tracker (WordTracker): The instance for tracking word formation.
        amharic_translator (Optional[SimpleAmharicTranslator]): The translator instance.
        tts_engine (Optional[SimpleTTS]): The TTS engine instance.
    """

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0,
                 enable_speech: bool = True, use_google_tts: bool = True,
                 show_landmarks: bool = False, enable_amharic: bool = False):
        """Initializes the ASL inference system.

        Args:
            model_path: Path to the `.tflite` model file.
            metadata_path: Path to the metadata JSON file.
            camera_index: The index of the camera to use.
            enable_speech: If True, enables the text-to-speech engine.
            use_google_tts: If True, prioritizes Google TTS.
            show_landmarks: If True, draws MediaPipe hand landmarks on the frame.
            enable_amharic: If True, enables Amharic translation.
        """

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.camera_index = camera_index
        self.enable_speech = enable_speech
        self.use_google_tts = use_google_tts
        self.show_landmarks = show_landmarks
        self.enable_amharic = enable_amharic

        # Core components
        self.interpreter = None
        self.class_mapping = {}
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        self.cap = None

        # Model details
        self.input_details = None
        self.output_details = None
        self.IMAGE_SIZE = (224, 224)
        self.LANDMARK_FEATURES = 42

        # Enhanced components
        self.word_tracker = WordTracker()
        self.progress_bar = None
        self.completion_banner = None
        self.suggestions_panel = None

        # Translation and TTS
        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = SimpleAmharicTranslator(use_translation=True)

        self.tts_engine = None
        if enable_speech:
            self.tts_engine = SimpleTTS(use_google=use_google_tts)

        # Display state
        self.last_spoken_word = ""
        self.last_amharic_translation = ""
        self.word_flash_time = 0
        self.flash_duration = 3.0
        self.window_width = 1024
        self.window_height = 768

        # Performance tracking
        self.fps_counter = deque(maxlen=30)

    def load_model(self) -> None:
        """Loads the TensorFlow Lite model and allocates tensors.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the model does not have the expected number of inputs.
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            logger.info(f"Loading model: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            if len(self.input_details) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(self.input_details)}")

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            sys.exit(1)

    def load_metadata(self) -> None:
        """Loads the class mapping from the metadata JSON file.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If the metadata file does not contain a valid class mapping.
        """
        try:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            if 'class_mapping' in metadata:
                self.class_mapping = {int(k): v for k, v in metadata['class_mapping'].items()}
            elif 'classes' in metadata:
                self.class_mapping = {i: cls for i, cls in enumerate(metadata['classes'])}
            else:
                self.class_mapping = {int(k): v for k, v in metadata.items() if k.isdigit()}

            if not self.class_mapping:
                raise ValueError("No valid class mapping found")

            logger.info(f"Loaded {len(self.class_mapping)} classes")

        except Exception as e:
            logger.error(f"Metadata loading failed: {e}")
            sys.exit(1)

    def initialize_mediapipe(self) -> None:
        """Initializes the MediaPipe Hands solution for hand tracking."""
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe initialized")

        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")
            sys.exit(1)

    def initialize_camera(self) -> None:
        """Initializes the camera capture with specified dimensions.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {self.camera_index} not available")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info("Camera initialized")

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            sys.exit(1)

    def initialize_ui_components(self, frame_width: int, frame_height: int) -> None:
        """Initializes all UI components based on the frame dimensions.

        Args:
            frame_width: The width of the video frame.
            frame_height: The height of the video frame.
        """
        self.progress_bar = ModernProgressBar((80, 80))
        self.completion_banner = WordCompletionBanner(frame_width, frame_height)
        self.suggestions_panel = SuggestionsPanel(frame_width, frame_height)

    def extract_hand_landmarks(self, results) -> Optional[np.ndarray]:
        """Extracts normalized 2D hand landmarks from MediaPipe results.

        Args:
            results: The output from `mediapipe.solutions.hands.process`.

        Returns:
            A numpy array of flattened landmark coordinates (x1, y1, x2, y2, ...),
            or None if no hands were detected.
        """
        if not results.multi_hand_landmarks:
            return None

        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y])

        return np.array(landmarks, dtype=np.float32)

    def crop_hand_region(self, frame: np.ndarray, results) -> Optional[np.ndarray]:
        """Crops the region of the hand from the frame based on landmarks.

        Calculates a bounding box around the detected hand landmarks and crops
        that region from the frame.

        Args:
            frame: The original video frame.
            results: The output from MediaPipe.

        Returns:
            The cropped hand image, resized to the model's expected input size,
            or None if no hand was found.
        """
        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = results.multi_hand_landmarks[0]

        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None

        hand_crop = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(hand_crop, self.IMAGE_SIZE)

    def preprocess_inputs(self, image: Optional[np.ndarray],
                         landmarks: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepares the image and landmark data for the TFLite model.

        This involves normalizing the image, handling cases where no hand is
        detected (by providing zero-arrays), and adding a batch dimension.

        Args:
            image: The cropped hand image.
            landmarks: The array of hand landmarks.

        Returns:
            A tuple containing the processed image and landmark tensors ready
            for the model.
        """
        if image is None:
            processed_image = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.astype(np.float32) / 255.0

        image_input = np.expand_dims(processed_image, axis=0)

        if landmarks is None:
            processed_landmarks = np.zeros(self.LANDMARK_FEATURES, dtype=np.float32)
        else:
            processed_landmarks = landmarks

        landmarks_input = np.expand_dims(processed_landmarks, axis=0)

        return image_input, landmarks_input

    def predict(self, image_input: np.ndarray, landmarks_input: np.ndarray) -> Tuple[str, float]:
        """Runs inference on the TFLite model with the given inputs.

        Args:
            image_input: The preprocessed image tensor.
            landmarks_input: The preprocessed landmarks tensor.

        Returns:
            A tuple containing the predicted class name and the confidence score.
        """
        try:
            landmarks_idx = 0 if 'landmarks' in self.input_details[0]['name'].lower() else 1
            image_idx = 1 - landmarks_idx

            self.interpreter.set_tensor(self.input_details[landmarks_idx]['index'], landmarks_input)
            self.interpreter.set_tensor(self.input_details[image_idx]['index'], image_input)

            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_idx])

            predicted_class = self.class_mapping.get(predicted_idx, f"Unknown_{predicted_idx}")
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def calculate_fps(self) -> float:
        """Calculates the current frames per second (FPS) based on a deque of timestamps.

        Returns:
            The calculated FPS.
        """
        current_time = time.time()
        self.fps_counter.append(current_time)

        if len(self.fps_counter) < 2:
            return 0.0

        time_span = self.fps_counter[-1] - self.fps_counter[0]
        return len(self.fps_counter) / time_span if time_span > 0 else 0.0

    def handle_word_completion(self, word: str):
        """Handles the logic for when a word is completed.

        This includes triggering the Amharic translation and queuing the
        English and Amharic text for text-to-speech.

        Args:
            word: The completed word.
        """
        if not word:
            return

        # Get Amharic translation
        amharic_translation = None
        if self.enable_amharic and self.amharic_translator:
            if self.amharic_translator.translation_available:
                amharic_translation = self.amharic_translator.translate(word)
                if amharic_translation:
                    logger.info(f"Translation successful: '{word}' -> '{amharic_translation}'")
                else:
                    logger.warning(f"Translation returned empty for: '{word}'")
            else:
                logger.warning("Amharic translator not available - check initialization")
        elif self.enable_amharic:
            logger.warning("Amharic enabled but translator not initialized")

        # Immediately update UI state
        self.last_spoken_word = word
        self.last_amharic_translation = amharic_translation
        self.word_flash_time = time.time()

        # Speak English first (non-blocking)
        if self.tts_engine:
            self.tts_engine.speak(word, 'en')

            # Speak Amharic after a short delay if translation is available
            if self.enable_amharic and amharic_translation:
                # Use threading timer for delayed Amharic speech
                threading.Timer(1.2, lambda: self.tts_engine.speak(amharic_translation, 'am')).start()

        # Log completion
        if amharic_translation:
            logger.info(f"Completed word: '{word}' -> Amharic: '{amharic_translation}' (with TTS)")
        else:
            logger.info(f"Completed word: '{word}' (English only)")

    def draw_ui(self, frame: np.ndarray, prediction: str, confidence: float, results,
                current_word: str, word_finalized: bool, letter_progress: float,
                word_suggestions: List[str]) -> np.ndarray:
        """Draws the entire user interface onto the frame.

        This method orchestrates the drawing of all UI components, including
        the info panel, progress bar, suggestions panel, and completion banner.

        Args:
            frame: The main video frame to draw on.
            prediction: The current predicted letter.
            confidence: The confidence of the current prediction.
            results: The raw results from MediaPipe for landmark drawing.
            current_word: The word currently being formed.
            word_finalized: A flag indicating if a word was just finalized.
            letter_progress: The hold progress for the current letter (0.0 to 1.0).
            word_suggestions: A list of suggestions for the current word.

        Returns:
            The frame with the complete UI drawn on it.
        """
        h, w = frame.shape[:2]

        if self.progress_bar is None:
            self.initialize_ui_components(w, h)

        fps = self.calculate_fps()

        if self.show_landmarks and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Main info panel
        panel_x, panel_y = 10, 150
        panel_w, panel_h = 400, 100
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 30), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (80, 120, 160), 2)

        pred_color = (0, 255, 0) if confidence >= self.word_tracker.confidence_threshold else (100, 100, 100)
        cv2.putText(frame, f"Letter: {prediction}", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

        word_color = (100, 200, 255) if current_word else (100, 100, 100)
        cv2.putText(frame, f"Word: {current_word.upper()}", (panel_x + 10, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 2)

        fps_color = (0, 255, 0) if fps >= 25 else (0, 200, 200)
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 280, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)

        if confidence >= self.word_tracker.confidence_threshold and prediction:
            self.progress_bar.draw(frame, letter_progress, prediction)
        else:
            self.progress_bar.draw(frame, 0.0, "")

        self.suggestions_panel.draw(frame, current_word, word_suggestions)

        current_time = time.time()
        if (self.last_spoken_word and
            current_time - self.word_flash_time < self.flash_duration):
            flash_progress = 1.0 - (current_time - self.word_flash_time) / self.flash_duration
            self.completion_banner.draw(frame, self.last_spoken_word, flash_progress,
                                      self.last_amharic_translation)

        # Status panel
        stats = self.word_tracker.get_stats()
        status_x = w - 300
        status_y = h - 120
        cv2.rectangle(frame, (status_x, status_y), (w - 10, h - 10), (20, 20, 30), -1)
        cv2.rectangle(frame, (status_x, status_y), (w - 10, h - 10), (80, 120, 160), 2)

        cv2.putText(frame, f"Words: {stats['total_words']}", (status_x + 10, status_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}",
                   (status_x + 10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)

        amharic_status = "ON" if self.enable_amharic else "OFF"
        amharic_color = (0, 200, 0) if self.enable_amharic else (100, 100, 100)
        cv2.putText(frame, f"Amharic: {amharic_status}", (status_x + 10, status_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, amharic_color, 1)

        if self.enable_amharic and self.amharic_translator:
            trans_status = "Available" if self.amharic_translator.translation_available else "Failed"
            trans_color = (0, 200, 0) if self.amharic_translator.translation_available else (200, 100, 0)
            cv2.putText(frame, f"Translation: {trans_status}", (status_x + 10, status_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, trans_color, 1)

        controls = [
            "Controls: '0'=complete word, 'r'=reset, '1/2/3'=suggestions",
            "'l'=toggle landmarks, 'q'=quit | Hold letters 0.8s to add"
        ]
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, h - 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def run(self) -> None:
        """The main application loop for video processing and inference."""
        logger.info("Starting ASL inference with Amharic translation...")
        logger.info("Controls: '0'=complete, 'r'=reset, '1/2/3'=suggestions, 'l'=landmarks, 'q'=quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb_frame)

                landmarks = self.extract_hand_landmarks(results)
                hand_crop = self.crop_hand_region(frame, results)

                image_input, landmarks_input = self.preprocess_inputs(hand_crop, landmarks)
                prediction, confidence = self.predict(image_input, landmarks_input)

                current_word, word_finalized, letter_progress = self.word_tracker.add_prediction(
                    prediction, confidence)

                stats = self.word_tracker.get_stats()
                word_suggestions = stats.get('word_suggestions', [])

                frame = self.draw_ui(frame, prediction, confidence, results,
                                   current_word, word_finalized, letter_progress, word_suggestions)

                cv2.imshow('ASL Inference with Amharic Translation', frame)

                if word_finalized and current_word:
                    self.handle_word_completion(current_word)
                    self.word_tracker.reset_word()

                # Handle key presses with immediate response
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('0'):  # Auto-complete
                    if self.word_tracker.current_word:
                        completed = self.word_tracker.auto_complete_word()
                        if completed:
                            self.handle_word_completion(completed)
                            self.word_tracker.reset_word()
                            logger.info(f"Manual completion: '{completed}'")
                elif key == ord('r'):  # Reset
                    self.word_tracker.reset_word()
                    logger.info("Word reset")
                elif key == ord('l'):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks
                    logger.info(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('1'):  # Suggestion 1
                    suggestions = self.word_tracker.get_word_suggestions(self.word_tracker.current_word)
                    if len(suggestions) > 0:
                        selected = suggestions[0]
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()
                        logger.info(f"Selected suggestion 1: '{selected}'")
                elif key == ord('2'):  # Suggestion 2
                    suggestions = self.word_tracker.get_word_suggestions(self.word_tracker.current_word)
                    if len(suggestions) > 1:
                        selected = suggestions[1]
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()
                        logger.info(f"Selected suggestion 2: '{selected}'")
                elif key == ord('3'):  # Suggestion 3
                    suggestions = self.word_tracker.get_word_suggestions(self.word_tracker.current_word)
                    if len(suggestions) > 2:
                        selected = suggestions[2]
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()
                        logger.info(f"Selected suggestion 3: '{selected}'")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Releases all resources, such as the camera and TTS engine."""
        logger.info("Cleaning up...")

        if self.tts_engine:
            self.tts_engine.stop()
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

        cv2.destroyAllWindows()


def diagnose_system():
    """Runs a diagnostic check to verify all dependencies are installed and working.

    This function prints a report of required and optional dependencies,
    their status (installed/missing), and instructions for installation.
    It also tests the translation services.
    """
    print("\n" + "="*60)
    print("ASL SYSTEM DIAGNOSTIC")
    print("="*60)

    deps = [
        ('OpenCV', 'cv2', 'pip install opencv-python'),
        ('MediaPipe', 'mediapipe', 'pip install mediapipe'),
        ('TensorFlow', 'tensorflow', 'pip install tensorflow'),
        ('NumPy', 'numpy', 'pip install numpy'),
        ('Requests', 'requests', 'pip install requests')
    ]

    for name, module, install_cmd in deps:
        try:
            __import__(module)
            print(f"✅ {name}: INSTALLED")
        except ImportError:
            print(f"❌ {name}: MISSING - {install_cmd}")

    print("\nOptional Dependencies:")

    try:
        import gtts
        print("✅ Google TTS: INSTALLED")
    except ImportError:
        print("❌ Google TTS: MISSING - pip install gtts")

    try:
        import pygame
        print("✅ Pygame (audio): INSTALLED")
    except ImportError:
        print("❌ Pygame (audio): MISSING - pip install pygame")

    # Check PIL for proper Amharic display
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL/Pillow (Amharic display): INSTALLED")
    except ImportError:
        print("❌ PIL/Pillow (Amharic display): MISSING - pip install Pillow")

    translation_working = False

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='am')
        result = translator.translate("hello")
        if result:
            print("✅ Deep Translator (recommended): WORKING")
            translation_working = True
    except ImportError:
        print("❌ Deep Translator: MISSING - pip install deep-translator")
    except Exception as e:
        print(f"❌ Deep Translator: ERROR - {e}")

    if not translation_working:
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate("hello", dest='am')
            if result and result.text:
                print("✅ Google Translate (googletrans): WORKING")
                translation_working = True
            else:
                print("❌ Google Translate (googletrans): NOT RESPONDING")
        except Exception as e:
            print(f"❌ Google Translate (googletrans): ERROR - {e}")

    print("="*60)
    print("COMPLETE INSTALLATION GUIDE:")
    print("="*60)
    print("1. Core dependencies (REQUIRED):")
    print("   pip install opencv-python mediapipe tensorflow gtts pygame requests")
    print()
    print("2. Amharic text display (RECOMMENDED for proper Unicode display):")
    print("   pip install Pillow")
    print()
    print("3. Translation service (choose one):")
    print("   pip install deep-translator  # Most reliable")
    print("   # OR if above fails:")
    print("   pip install googletrans==4.0.0rc1")
    print()
    print("4. Test your setup:")
    print("   python script.py --diagnose")
    print("   python script.py --translate-amharic")
    print()
    print("KNOWN FEATURES:")
    print("• English words are spoken clearly")
    print("• Amharic translations are displayed AND spoken (romanized)")
    print("• Real-time word suggestions from online dictionary")
    print("• Immediate response to number key selections")
    print("• Proper Unicode display with PIL/Pillow")

    if not translation_working:
        print()
        print("⚠️  WARNING: Amharic translation not working!")
        print("   RECOMMENDED FIX: pip install deep-translator")

    print("="*60 + "\n")


def main():
    """The main entry point for the script.

    Parses command-line arguments and initializes and runs the
    `ASLRealTimeInference` system.
    """
    parser = argparse.ArgumentParser(description="ASL Inference with Amharic Translation")

    parser.add_argument('--model', default='export/asl_model.tflite',
                       help='Path to TensorFlow Lite model')
    parser.add_argument('--metadata', default='processed_asl/metadata.json',
                       help='Path to metadata JSON file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--no-speech', action='store_true', help='Disable TTS')
    parser.add_argument('--no-google-tts', action='store_true', help='Use only system TTS')
    parser.add_argument('--translate-amharic', action='store_true',
                       help='Enable Amharic translation and bilingual TTS')
    parser.add_argument('--show-landmarks', action='store_true', help='Show hand landmarks')
    parser.add_argument('--letter-hold-time', type=float, default=2.0,
                       help='Seconds to hold letter (default: 0.8)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence (default: 0.6)')
    parser.add_argument('--diagnose', action='store_true', help='Run system diagnostic')

    args = parser.parse_args()

    if args.diagnose:
        diagnose_system()
        return

    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    try:
        logger.info("="*60)
        logger.info("ASL INFERENCE WITH AMHARIC TRANSLATION")
        logger.info("="*60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Metadata: {args.metadata}")
        logger.info(f"Amharic Translation: {'ENABLED' if args.translate_amharic else 'DISABLED'}")
        logger.info(f"Speech: {'ENABLED' if not args.no_speech else 'DISABLED'}")
        logger.info("="*60)

        system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera,
            enable_speech=not args.no_speech,
            use_google_tts=not args.no_google_tts,
            show_landmarks=args.show_landmarks,
            enable_amharic=args.translate_amharic
        )

        system.word_tracker = WordTracker(
            min_letter_duration=args.letter_hold_time,
            confidence_threshold=args.confidence_threshold
        )

        system.load_model()
        system.load_metadata()
        system.initialize_mediapipe()
        system.initialize_camera()

        logger.info("System ready!")
        logger.info("- Press '0' to complete current word")
        logger.info("- Press '1', '2', '3' to select word suggestions")
        logger.info(f"- Hold letters for {args.letter_hold_time}s to add them")
        if args.translate_amharic:
            logger.info("- Every completed word will be spoken in English AND Amharic")

        system.run()

    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()