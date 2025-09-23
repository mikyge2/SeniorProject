"""
Enhanced Real-time ASL Inference Script
Features: Auto-completion, Amharic translation, improved responsiveness, modern UI
Compatible with Python 3.8+, TensorFlow Lite, OpenCV, MediaPipe
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


class AmharicTranslator:
    """Handles Amharic translation with offline dictionary and Google Translate fallback."""

    def __init__(self, use_translation: bool = False):
        self.use_translation = use_translation
        self.translator = None
        self.translation_available = False
        self.translation_cache = {}
        self.cache_max_size = 500
        self.translation_method = None

        if use_translation:
            self._initialize_translator()
            logger.info(f"Translation service initialized: {self.translation_available}")

    def _initialize_translator(self):
        """Initialize translation service with better error handling."""
        logger.info("Initializing Amharic translation service...")

        # Load offline dictionary first (always available)
        self._load_offline_dictionary()
        self.translation_available = True  # At minimum, we have offline dictionary

        # Try to initialize online translation
        translation_methods = [
            ("deep_translator", self._try_deep_translator),
            ("googletrans", self._try_googletrans),
        ]

        for method_name, method_func in translation_methods:
            try:
                if method_func():
                    logger.info(f"Online translation initialized with {method_name}")
                    return
            except Exception as e:
                logger.debug(f"{method_name} failed: {e}")
                continue

        logger.info(f"Using offline Amharic dictionary only ({len(self.offline_dict)} words)")

    def _try_deep_translator(self) -> bool:
        """Try to initialize deep-translator."""
        try:
            from deep_translator import GoogleTranslator
            self.translator = GoogleTranslator(source='en', target='am')
            self.translation_method = "deep_translator"

            # Test translation
            test_result = self.translator.translate("hello")
            if test_result and test_result != "hello":
                logger.info("Deep Translator test successful")
                return True
            return False
        except Exception as e:
            logger.debug(f"Deep translator initialization failed: {e}")
            return False

    def _try_googletrans(self) -> bool:
        """Try to initialize googletrans."""
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.translation_method = "googletrans"

            # Test translation with timeout
            test_result = self._translate_with_timeout("hello", timeout=5)
            if test_result and test_result != "hello":
                logger.info("GoogleTrans test successful")
                return True
            return False
        except Exception as e:
            logger.debug(f"GoogleTrans initialization failed: {e}")
            return False

    def _translate_with_timeout(self, text: str, timeout: int = 5) -> Optional[str]:
        """Translate with timeout protection."""
        if not self.translator:
            return None

        try:
            if self.translation_method == "deep_translator":
                result = self.translator.translate(text)
                return result if result and result != text else None
            elif self.translation_method == "googletrans":
                result = self.translator.translate(text, dest='am')
                return result.text if result and result.text and result.text != text else None
        except Exception as e:
            logger.debug(f"Translation error for '{text}': {e}")

        return None

    def _load_offline_dictionary(self):
        """Load comprehensive English to Amharic dictionary."""
        self.offline_dict = {
            # Greetings and common expressions
            "hello": "ሰላም", "hi": "ሰላም", "goodbye": "ደህና ሁን", "bye": "ደህና ሁን",
            "thanks": "አመሰግናለሁ", "thank": "አመሰግናለሁ", "please": "እባክህ",
            "sorry": "ይቅርታ", "yes": "አዎ", "no": "አይ", "ok": "እሺ", "okay": "እሺ",

            # Basic needs
            "water": "ውሃ", "food": "ምግብ", "eat": "በላ", "drink": "ጠጣ",
            "hungry": "ተራብ", "thirsty": "ተጠመ", "tired": "ደከመ", "sleep": "ተኛ",

            # Family
            "family": "ቤተሰብ", "mother": "እናት", "father": "አባት", "sister": "እህት",
            "brother": "ወንድም", "child": "ልጅ", "friend": "ወዳጅ", "person": "ሰው",

            # Common verbs
            "go": "ሂድ", "come": "ና", "see": "ይመልከት", "hear": "ስማ", "speak": "ተናገር",
            "help": "እርዳታ", "work": "ስራ", "play": "ተጫወት", "love": "ፍቅር",
            "like": "እወዳለሁ", "want": "ፈልግ", "need": "ያስፈልጋል", "know": "ያውቃል",

            # Feelings
            "happy": "ደስተኛ", "sad": "አዝኛ", "angry": "ተናደደ", "good": "ጥሩ", "bad": "መጥፎ",

            # Descriptions
            "big": "ትልቅ", "small": "ትንሽ", "hot": "ሞቅ ያለ", "cold": "ቀዝቃዛ",
            "new": "አዲስ", "old": "አሮጌ", "fast": "ፈጣን", "slow": "ቀርፋፋ",

            # Colors
            "red": "ቀይ", "blue": "ሰማያዊ", "green": "አረንጓዴ", "yellow": "ቢጫ",
            "black": "ጥቁር", "white": "ነጭ", "brown": "ቡናማ",

            # Numbers
            "one": "አንድ", "two": "ሁለት", "three": "ሦስት", "four": "አራት", "five": "አምስት",
            "six": "ስድስት", "seven": "ሰባት", "eight": "ስምንት", "nine": "ዘጠኝ", "ten": "አስር",

            # Places
            "home": "ቤት", "house": "ቤት", "school": "ትምህርት ቤት", "work": "ስራ ቦታ",

            # Time
            "time": "ጊዜ", "day": "ቀን", "night": "ሌሊት", "today": "ዛሬ", "tomorrow": "ነገ",

            # Body parts
            "hand": "እጅ", "eye": "ዓይን", "head": "ራስ", "heart": "ልብ",

            # Technology
            "computer": "ኮምፒዩተር", "phone": "ስልክ", "book": "መጽሐፍ"
        }

    def translate(self, text: str) -> Optional[str]:
        """Synchronous translation with caching."""
        if not self.use_translation or not self.translation_available or not text.strip():
            return None

        text_lower = text.lower().strip()

        # Check cache first
        if text_lower in self.translation_cache:
            return self.translation_cache[text_lower]

        translation = None

        # Try offline dictionary first
        if text_lower in self.offline_dict:
            translation = self.offline_dict[text_lower]
            logger.debug(f"Offline translation: '{text}' -> '{translation}'")
        elif self.translator and self.translation_method:
            # Try online translation
            translation = self._translate_with_timeout(text_lower, timeout=3)
            if translation:
                logger.debug(f"Online translation ({self.translation_method}): '{text}' -> '{translation}'")

        # Cache the result (even if None)
        if len(self.translation_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_keys = list(self.translation_cache.keys())[:50]
            for key in oldest_keys:
                del self.translation_cache[key]

        self.translation_cache[text_lower] = translation
        return translation

    def get_translation_status(self) -> str:
        """Get current translation status."""
        if not self.use_translation:
            return "Disabled"
        elif not self.translation_available:
            return "Failed to initialize"
        elif self.translation_method:
            return f"Online + Offline ({self.translation_method})"
        else:
            return f"Offline only ({len(self.offline_dict)} words)"


class SimpleTTS:
    """Simple TTS using system commands."""

    def __init__(self):
        self.working = self._test_system_tts()

    def _test_system_tts(self) -> bool:
        """Test system TTS availability."""
        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":
                result = subprocess.run(['say', 'test'], capture_output=True, timeout=2)
                return result.returncode == 0
            elif system == "linux":
                if shutil.which('espeak'):
                    result = subprocess.run(['espeak', 'test'], capture_output=True, timeout=2)
                    return result.returncode == 0
        except Exception:
            pass
        return False

    def speak(self, text: str) -> bool:
        """Speak text using system TTS."""
        if not self.working:
            return False

        try:
            import platform
            system = platform.system().lower()

            if system == "darwin":
                subprocess.run(['say', text], timeout=5)
                return True
            elif system == "linux" and shutil.which('espeak'):
                subprocess.run(['espeak', text], timeout=5)
                return True
        except Exception:
            pass
        return False


class GoogleTTSEngine:
    """Google TTS with audio playback and fallbacks."""

    def __init__(self, use_google: bool = True):
        self.use_google = use_google
        self.google_available = False
        self.system_tts = SimpleTTS()
        self.speech_queue = queue.Queue(maxsize=10)
        self.is_running = True
        self.temp_dir = tempfile.gettempdir()

        if use_google:
            self._test_google_tts()

        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _test_google_tts(self):
        """Test Google TTS availability."""
        try:
            import gtts
            # Test basic functionality
            test_tts = gtts.gTTS(text="test", lang='en')

            # Check audio playback capabilities
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.quit()
                self.google_available = True
                logger.info("Google TTS with pygame available")
                return
            except ImportError:
                pass

            # Check system audio players
            if any(shutil.which(p) for p in ['mpg123', 'afplay', 'vlc']):
                self.google_available = True
                logger.info("Google TTS with system audio available")

        except ImportError:
            logger.info("gTTS not installed - using system TTS only")

    def _tts_worker(self):
        """TTS worker thread."""
        while self.is_running:
            try:
                item = self.speech_queue.get(timeout=1.0)
                if item is None:
                    break

                text, amharic_text = item

                # Speak English first
                self._speak_text(text)

                # Then speak Amharic if available
                if amharic_text:
                    time.sleep(0.5)  # Brief pause between languages
                    self._speak_amharic(amharic_text)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    def _speak_text(self, text: str):
        """Speak English text."""
        if self.google_available and self.use_google:
            if self._speak_with_google(text, lang='en'):
                return

        # Fallback to system TTS
        self.system_tts.speak(text)

    def _speak_amharic(self, text: str):
        """Speak Amharic text using Google Translate TTS."""
        if not self.google_available or not self.use_google:
            # Fallback announcement
            announcement = f"Amharic translation: {text}"
            self.system_tts.speak(announcement)
            return

        # Try using Google Translate's TTS for Amharic
        if self._speak_amharic_with_google_translate(text):
            logger.info(f"Spoke Amharic via Google Translate TTS: {text}")
            return

        # Try phonetic approximation as fallback
        phonetic_text = self._get_amharic_phonetic(text)
        if phonetic_text and self._speak_with_google(phonetic_text, lang='en'):
            logger.info(f"Spoke Amharic phonetically: {text} -> {phonetic_text}")
            return

        # Final fallback: announcement
        announcement = f"Amharic translation: {text}"
        if self._speak_with_google(announcement, lang='en'):
            logger.info(f"Announced Amharic translation: {text}")
        else:
            self.system_tts.speak(announcement)

    def _speak_amharic_with_google_translate(self, text: str) -> bool:
        """Use Google Translate's TTS API for Amharic."""
        try:
            # Use Google Translate TTS endpoint directly
            import urllib.parse
            import urllib.request

            # Google Translate TTS endpoint
            encoded_text = urllib.parse.quote(text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=am&client=tw-ob&q={encoded_text}"

            # Download audio
            audio_file = os.path.join(self.temp_dir, f"amharic_tts_{int(time.time())}.mp3")

            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

            with urllib.request.urlopen(req, timeout=5) as response:
                with open(audio_file, 'wb') as f:
                    f.write(response.read())

            # Play the audio file
            played = False

            # Try pygame first
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
                played = True
            except ImportError:
                pass

            # Try system players if pygame failed
            if not played:
                for player in ['mpg123', 'afplay', 'vlc']:
                    if shutil.which(player):
                        try:
                            cmd = [player]
                            if player == 'vlc':
                                cmd.extend(['--intf', 'dummy', '--play-and-exit'])
                            cmd.append(audio_file)

                            subprocess.run(cmd, timeout=10, capture_output=True)
                            played = True
                            break
                        except Exception:
                            continue

            # Cleanup
            if os.path.exists(audio_file):
                os.remove(audio_file)

            return played

        except Exception as e:
            logger.debug(f"Google Translate TTS failed for Amharic: {e}")
            return False

    def _get_amharic_phonetic(self, amharic_text: str) -> Optional[str]:
        """Convert Amharic to English phonetic approximation."""
        # Basic phonetic mapping for common Amharic words
        phonetic_map = {
            "ሰላም": "selam",  # hello
            "ደህና ሁን": "dehna hun",  # goodbye
            "አመሰግናለሁ": "ameseginalehu",  # thanks
            "እባክህ": "ebakish",  # please
            "ይቅርታ": "yikirta",  # sorry
            "አዎ": "awo",  # yes
            "አይ": "ay",  # no
            "እሺ": "eshi",  # ok
            "ውሃ": "wiha",  # water
            "ምግብ": "migib",  # food
            "በላ": "bela",  # eat
            "ጠጣ": "teta",  # drink
            "ቤተሰብ": "beteseb",  # family
            "እናት": "enat",  # mother
            "አባት": "abat",  # father
            "እህት": "ehit",  # sister
            "ወንድም": "wendim",  # brother
            "ልጅ": "lij",  # child
            "ወዳጅ": "wedaj",  # friend
            "ሂድ": "hid",  # go
            "ና": "na",  # come
            "እርዳታ": "eridata",  # help
            "ስራ": "sira",  # work
            "ፍቅር": "fikir",  # love
            "ደስተኛ": "desitegna",  # happy
            "አዝኛ": "azegna",  # sad
            "ጥሩ": "tiru",  # good
            "መጥፎ": "metafo",  # bad
            "ትልቅ": "tilik",  # big
            "ትንሽ": "tinish",  # small
            "ቀይ": "key",  # red
            "ሰማያዊ": "semayawi",  # blue
            "አረንጓዴ": "arengwade",  # green
            "ቢጫ": "bicha",  # yellow
            "ጥቁር": "tikur",  # black
            "ነጭ": "nech",  # white
            "አንድ": "and",  # one
            "ሁለት": "hulet",  # two
            "ሦስት": "sost",  # three
            "አራት": "arat",  # four
            "አምስት": "amist",  # five
            "ቤት": "bet",  # home/house
            "ትምህርት ቤት": "timhirt bet",  # school
            "ጊዜ": "gize",  # time
            "ቀን": "ken",  # day
            "ሌሊት": "lelit",  # night
            "ዛሬ": "zare",  # today
            "ነገ": "nege",  # tomorrow
            "እጅ": "ij",  # hand
            "ዓይን": "ayin",  # eye
            "ራስ": "ras",  # head
            "ልብ": "lib",  # heart
        }

        return phonetic_map.get(amharic_text)

    def _speak_with_google(self, text: str, lang: str = 'en') -> bool:
        """Speak using Google TTS."""
        try:
            import gtts

            # Check if language is supported
            supported_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
            if lang not in supported_langs:
                logger.warning(f"Language '{lang}' not supported by Google TTS")
                return False

            tts = gtts.gTTS(text=text, lang=lang, slow=False)
            audio_file = os.path.join(self.temp_dir, f"tts_{int(time.time())}_{hash(text)}.mp3")
            tts.save(audio_file)

            # Try pygame first
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
                os.remove(audio_file)
                return True
            except ImportError:
                pass

            # Try system players
            for player in ['mpg123', 'afplay', 'vlc']:
                if shutil.which(player):
                    try:
                        cmd = [player]
                        if player == 'vlc':
                            cmd.extend(['--intf', 'dummy', '--play-and-exit'])
                        cmd.append(audio_file)

                        subprocess.run(cmd, timeout=10, capture_output=True)
                        os.remove(audio_file)
                        return True
                    except Exception as e:
                        logger.debug(f"{player} failed: {e}")
                        continue

            # Cleanup if no player worked
            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            logger.debug(f"Google TTS error for '{text}' (lang: {lang}): {e}")

        return False

    def speak(self, text: str, amharic_text: Optional[str] = None) -> bool:
        """Queue text for speech."""
        try:
            self.speech_queue.put_nowait((text, amharic_text))
            return True
        except queue.Full:
            logger.warning("TTS queue full, skipping speech")
            return False

    def stop(self):
        """Stop TTS engine."""
        self.is_running = False
        try:
            self.speech_queue.put(None, timeout=1.0)
        except queue.Full:
            pass


class ModernProgressBar:
    """Circular progress bar for letter hold duration."""

    def __init__(self, center: Tuple[int, int], radius: int = 35, thickness: int = 8):
        self.center = center
        self.radius = radius
        self.thickness = thickness
        self.inner_radius = radius - thickness

    def draw(self, frame: np.ndarray, progress: float, letter: str = "") -> np.ndarray:
        """Draw animated circular progress bar."""
        progress = max(0.0, min(1.0, progress))

        # Background circle
        cv2.circle(frame, self.center, self.radius, (40, 40, 40), self.thickness)

        if progress > 0:
            # Progress arc with color transition
            if progress < 0.3:
                color = (0, 100, 255)  # Red
            elif progress < 0.7:
                color = (0, 200, 255)  # Orange
            else:
                color = (0, 255, 100)  # Green

            # Draw progress arc
            angle = int(360 * progress)
            axes = (self.radius, self.radius)
            cv2.ellipse(frame, self.center, axes, -90, 0, angle, color, self.thickness)

        # Center circle with letter
        if letter:
            # Pulsing inner circle
            pulse = 0.7 + 0.3 * math.sin(time.time() * 5)
            inner_color = (int(25 * pulse), int(25 * pulse), int(35 * pulse))

            cv2.circle(frame, self.center, self.inner_radius, inner_color, -1)
            cv2.circle(frame, self.center, self.inner_radius, (80, 80, 100), 2)

            # Letter text
            font_scale = 1.0 if len(letter) == 1 else 0.7
            text_size = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = self.center[0] - text_size[0] // 2
            text_y = self.center[1] + text_size[1] // 2

            cv2.putText(frame, letter, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        return frame


class WordCompletionBanner:
    """Animated banner for completed words with Amharic translation."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.banner_height = 160  # Increased height for better Amharic display
        self.banner_y = frame_height // 2 - self.banner_height // 2

    def draw(self, frame: np.ndarray, word: str, flash_progress: float,
             amharic_translation: Optional[str] = None) -> np.ndarray:
        """Draw animated completion banner."""
        if not word or flash_progress <= 0:
            return frame

        # Smooth animation
        ease_progress = 1 - (1 - flash_progress) ** 2
        banner_width = int(self.frame_width * 0.85 * ease_progress)
        banner_x = (self.frame_width - banner_width) // 2

        # Pulsing colors
        pulse = (math.sin(flash_progress * math.pi * 4) + 1) / 2
        bg_color = (15, 15, 25)
        border_color = (int(50 + pulse * 150), int(150 + pulse * 50), int(200 + pulse * 55))

        # Main banner
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     bg_color, -1)

        # Animated border
        border_thickness = int(3 + pulse * 2)
        cv2.rectangle(frame, (banner_x, self.banner_y),
                     (banner_x + banner_width, self.banner_y + self.banner_height),
                     border_color, border_thickness)

        # Title
        title = "WORD COMPLETED!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_x = banner_x + (banner_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, self.banner_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # English word
        word_text = word.upper()
        word_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        word_x = banner_x + (banner_width - word_size[0]) // 2
        word_y = self.banner_y + 70

        glow_color = (int(pulse * 150), int(pulse * 255), int(pulse * 200))
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, glow_color, 5)
        cv2.putText(frame, word_text, (word_x, word_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Amharic translation
        if amharic_translation:
            # Amharic text
            amharic_size = cv2.getTextSize(amharic_translation, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            amharic_x = banner_x + (banner_width - amharic_size[0]) // 2
            amharic_y = self.banner_y + 110

            # Amharic glow effect
            cv2.putText(frame, amharic_translation, (amharic_x, amharic_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 100), 3)
            cv2.putText(frame, amharic_translation, (amharic_x, amharic_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 255, 150), 2)

            # Label
            label = "(Amharic)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_x = banner_x + (banner_width - label_size[0]) // 2
            cv2.putText(frame, label, (label_x, self.banner_y + 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 160, 120), 1)
        else:
            # Show "Translation not available" message
            no_trans = "Translation not available"
            no_trans_size = cv2.getTextSize(no_trans, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            no_trans_x = banner_x + (banner_width - no_trans_size[0]) // 2
            cv2.putText(frame, no_trans, (no_trans_x, self.banner_y + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        return frame


class WordTracker:
    """Enhanced word tracker with faster response and auto-completion."""

    def __init__(self, window_size: int = 6, confidence_threshold: float = 0.6,
                 pause_threshold: float = 2.0, min_letter_duration: float = 0.8):
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

        # Load word dictionary
        self.common_words = [
            "hello", "hi", "goodbye", "thanks", "please", "sorry", "help", "love",
            "family", "friend", "mother", "father", "water", "food", "home", "work",
            "happy", "sad", "good", "bad", "yes", "no", "go", "come", "see", "hear",
            "want", "need", "like", "know", "think", "feel", "big", "small", "hot", "cold",
            "red", "blue", "green", "yellow", "one", "two", "three", "four", "five",
            "today", "tomorrow", "time", "day", "night", "morning", "school", "book"
        ]
        self.common_words.sort()

    def get_word_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """Get word suggestions."""
        if not partial_word:
            return []

        partial_lower = partial_word.lower()
        suggestions = []

        for word in self.common_words:
            if word.startswith(partial_lower) and word != partial_lower:
                suggestions.append(word)
                if len(suggestions) >= max_suggestions:
                    break

        return suggestions

    def add_prediction(self, letter: str, confidence: float) -> Tuple[str, bool, float]:
        """Add prediction and track word progress."""
        current_time = time.time()
        word_finalized = False

        # Check for pause
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

        # Get stable letter
        stable_letter = self._get_stable_letter()

        if stable_letter == self.current_letter and stable_letter:
            # Same letter - check hold duration
            hold_duration = current_time - self.current_letter_start
            self.letter_hold_progress = min(1.0, hold_duration / self.min_letter_duration)

            # Accept letter if held long enough
            if hold_duration >= self.min_letter_duration:
                self.current_word += stable_letter.lower()
                logger.info(f"Letter '{stable_letter}' added: '{self.current_word}'")

                # Reset for next letter
                self.current_letter = ""
                self.current_letter_start = current_time
                self.letter_hold_progress = 0.0

        elif stable_letter != self.current_letter:
            # Letter changed
            self.current_letter = stable_letter
            self.current_letter_start = current_time
            self.letter_hold_progress = 0.0

        return self.current_word, word_finalized, self.letter_hold_progress

    def _get_stable_letter(self) -> str:
        """Get most stable letter from buffer."""
        if not self.prediction_buffer:
            return ""

        letter_scores = Counter()
        current_time = time.time()

        for letter, confidence, timestamp in self.prediction_buffer:
            # Weight by recency and confidence
            time_weight = max(0.1, 1.0 - (current_time - timestamp) / 2.0)
            score = confidence * time_weight
            letter_scores[letter] += score

        return letter_scores.most_common(1)[0][0] if letter_scores else ""

    def auto_complete_word(self) -> Optional[str]:
        """Auto-complete current word."""
        if self.current_word and not self.word_finalized:
            self.recognized_words.append(self.current_word)
            self.total_words += 1
            self.word_finalized = True
            logger.info(f"Auto-completed: '{self.current_word}'")
            return self.current_word
        return None

    def select_suggestion(self, index: int) -> Optional[str]:
        """Select word suggestion."""
        suggestions = self.get_word_suggestions(self.current_word)
        if 0 <= index < len(suggestions):
            selected = suggestions[index]
            self.current_word = selected
            self.recognized_words.append(selected)
            self.total_words += 1
            return selected
        return None

    def reset_word(self):
        """Reset current word."""
        self.current_word = ""
        self.word_finalized = False
        self.current_letter = ""
        self.current_letter_start = time.time()
        self.letter_hold_progress = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            "total_words": self.total_words,
            "current_word": self.current_word,
            "current_letter": self.current_letter,
            "letter_progress": self.letter_hold_progress,
            "word_suggestions": self.get_word_suggestions(self.current_word),
            "recent_words": self.recognized_words[-5:],
            "buffer_size": len(self.prediction_buffer)
        }


class SuggestionsPanel:
    """Modern suggestions panel with auto-complete option."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = 280
        self.panel_x = frame_width - self.panel_width - 10
        self.button_height = 40

    def draw(self, frame: np.ndarray, current_word: str, suggestions: List[str]) -> np.ndarray:
        """Draw suggestions panel."""
        if not current_word:
            return frame

        panel_height = 100 + len(suggestions) * (self.button_height + 5) + 10
        panel_y = 10

        # Panel background
        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (20, 20, 30), -1)
        cv2.rectangle(frame, (self.panel_x, panel_y),
                     (self.panel_x + self.panel_width, panel_y + panel_height),
                     (100, 150, 200), 2)

        # Title
        cv2.putText(frame, "WORD OPTIONS", (self.panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 2)

        # Current word
        cv2.putText(frame, f"'{current_word.upper()}'", (self.panel_x + 10, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Auto-complete button (0)
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

        # Suggestion buttons
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
    """Main ASL inference system with enhanced features."""

    def __init__(self, model_path: str, metadata_path: str, camera_index: int = 0,
                 enable_speech: bool = True, use_google_tts: bool = True,
                 show_landmarks: bool = False, enable_amharic: bool = False):

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

        # Translation and TTS - Initialize properly
        self.amharic_translator = None
        if enable_amharic:
            self.amharic_translator = AmharicTranslator(use_translation=True)
            logger.info(f"Amharic translator status: {self.amharic_translator.get_translation_status()}")

        self.tts_engine = None
        if enable_speech:
            self.tts_engine = GoogleTTSEngine(use_google=use_google_tts)

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
        """Load TensorFlow Lite model."""
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
        """Load class mapping metadata."""
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
        """Initialize MediaPipe hands detection."""
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
        """Initialize camera."""
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
        """Initialize UI components."""
        self.progress_bar = ModernProgressBar((80, 80))
        self.completion_banner = WordCompletionBanner(frame_width, frame_height)
        self.suggestions_panel = SuggestionsPanel(frame_width, frame_height)

    def extract_hand_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract hand landmarks as flat array."""
        if not results.multi_hand_landmarks:
            return None

        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y])

        return np.array(landmarks, dtype=np.float32)

    def crop_hand_region(self, frame: np.ndarray, results) -> Optional[np.ndarray]:
        """Crop hand region from frame."""
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
        """Preprocess model inputs."""
        # Image preprocessing
        if image is None:
            processed_image = np.zeros((224, 224, 3), dtype=np.float32)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.astype(np.float32) / 255.0

        image_input = np.expand_dims(processed_image, axis=0)

        # Landmarks preprocessing
        if landmarks is None:
            processed_landmarks = np.zeros(self.LANDMARK_FEATURES, dtype=np.float32)
        else:
            processed_landmarks = landmarks

        landmarks_input = np.expand_dims(processed_landmarks, axis=0)

        return image_input, landmarks_input

    def predict(self, image_input: np.ndarray, landmarks_input: np.ndarray) -> Tuple[str, float]:
        """Run model inference."""
        try:
            # Set inputs (determine correct order)
            landmarks_idx = 0 if 'landmarks' in self.input_details[0]['name'].lower() else 1
            image_idx = 1 - landmarks_idx

            self.interpreter.set_tensor(self.input_details[landmarks_idx]['index'], landmarks_input)
            self.interpreter.set_tensor(self.input_details[image_idx]['index'], image_input)

            # Run inference
            self.interpreter.invoke()

            # Get prediction
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_idx])

            predicted_class = self.class_mapping.get(predicted_idx, f"Unknown_{predicted_idx}")
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def calculate_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time)

        if len(self.fps_counter) < 2:
            return 0.0

        time_span = self.fps_counter[-1] - self.fps_counter[0]
        return len(self.fps_counter) / time_span if time_span > 0 else 0.0

    def handle_word_completion(self, word: str):
        """Handle completed word with translation and TTS."""
        if not word:
            return

        logger.info(f"Handling word completion: '{word}'")

        # Get Amharic translation
        amharic_translation = None
        if self.enable_amharic and self.amharic_translator:
            try:
                amharic_translation = self.amharic_translator.translate(word)
                if amharic_translation:
                    logger.info(f"Translated '{word}' to '{amharic_translation}'")
                else:
                    logger.info(f"No translation available for '{word}'")
            except Exception as e:
                logger.error(f"Translation error: {e}")

        # Speak the word (both English and Amharic)
        if self.tts_engine:
            try:
                success = self.tts_engine.speak(word, amharic_translation)
                if success:
                    logger.info(f"Queued for speech: '{word}'" +
                               (f" + '{amharic_translation}'" if amharic_translation else ""))
                else:
                    logger.warning("Failed to queue speech")
            except Exception as e:
                logger.error(f"TTS error: {e}")

        # Update display state
        self.last_spoken_word = word
        self.last_amharic_translation = amharic_translation
        self.word_flash_time = time.time()

        logger.info(f"Word completion handled: '{word}'" +
                   (f" (Amharic: '{amharic_translation}')" if amharic_translation else " (no translation)"))

    def draw_ui(self, frame: np.ndarray, prediction: str, confidence: float, results,
                current_word: str, word_finalized: bool, letter_progress: float,
                word_suggestions: List[str]) -> np.ndarray:
        """Draw modern UI overlay."""
        h, w = frame.shape[:2]

        # Initialize UI components if needed
        if self.progress_bar is None:
            self.initialize_ui_components(w, h)

        # Calculate FPS
        fps = self.calculate_fps()

        # Draw hand landmarks if enabled
        if self.show_landmarks and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Main info panel
        panel_x, panel_y = 10, 150
        panel_w, panel_h = 450, 120
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 30), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (80, 120, 160), 2)

        # Current prediction
        pred_color = (0, 255, 0) if confidence >= self.word_tracker.confidence_threshold else (100, 100, 100)
        cv2.putText(frame, f"Letter: {prediction}", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (panel_x + 10, panel_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

        # Current word
        word_color = (100, 200, 255) if current_word else (100, 100, 100)
        cv2.putText(frame, f"Word: {current_word.upper()}", (panel_x + 10, panel_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 2)

        # FPS display
        fps_color = (0, 255, 0) if fps >= 25 else (0, 200, 200)
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 320, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)

        # Translation status
        if self.enable_amharic and self.amharic_translator:
            status = self.amharic_translator.get_translation_status()
            status_color = (0, 255, 0) if "Online" in status else (200, 200, 0)
            cv2.putText(frame, f"Translation: {status}", (panel_x + 10, panel_y + 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # Progress bar for letter hold
        if confidence >= self.word_tracker.confidence_threshold and prediction:
            self.progress_bar.draw(frame, letter_progress, prediction)
        else:
            self.progress_bar.draw(frame, 0.0, "")

        # Suggestions panel
        self.suggestions_panel.draw(frame, current_word, word_suggestions)

        # Word completion banner
        current_time = time.time()
        if (self.last_spoken_word and
            current_time - self.word_flash_time < self.flash_duration):
            flash_progress = 1.0 - (current_time - self.word_flash_time) / self.flash_duration
            self.completion_banner.draw(frame, self.last_spoken_word, flash_progress,
                                      self.last_amharic_translation)

        # Status panel
        stats = self.word_tracker.get_stats()
        status_x = w - 300
        status_y = h - 140
        cv2.rectangle(frame, (status_x, status_y), (w - 10, h - 10), (20, 20, 30), -1)
        cv2.rectangle(frame, (status_x, status_y), (w - 10, h - 10), (80, 120, 160), 2)

        cv2.putText(frame, f"Words: {stats['total_words']}", (status_x + 10, status_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}",
                   (status_x + 10, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 150), 1)
        cv2.putText(frame, f"Amharic: {'ON' if self.enable_amharic else 'OFF'}",
                   (status_x + 10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (0, 200, 0) if self.enable_amharic else (100, 100, 100), 1)

        # Speech status
        cv2.putText(frame, f"Speech: {'ON' if self.enable_speech else 'OFF'}",
                   (status_x + 10, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (0, 200, 0) if self.enable_speech else (100, 100, 100), 1)

        # Recent words
        if stats['recent_words']:
            recent_text = "Recent: " + ", ".join(stats['recent_words'])
            cv2.putText(frame, recent_text[:35] + "..." if len(recent_text) > 35 else recent_text,
                       (status_x + 10, status_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 200), 1)

        # Controls info
        controls = [
            "Controls: '0'=complete word, 'r'=reset, '1/2/3'=suggestions",
            "'l'=toggle landmarks, 'q'=quit | Hold letters 0.8s to add"
        ]
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, h - 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def run(self) -> None:
        """Main inference loop."""
        logger.info("Starting ASL inference with Amharic translation...")
        logger.info("Controls: '0'=complete word, 'r'=reset, '1/2/3'=suggestions, 'l'=landmarks, 'q'=quit")

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue

                frame = cv2.flip(frame, 1)  # Mirror effect
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe processing
                results = self.hands.process(rgb_frame)

                # Extract features
                landmarks = self.extract_hand_landmarks(results)
                hand_crop = self.crop_hand_region(frame, results)

                # Preprocess and predict
                image_input, landmarks_input = self.preprocess_inputs(hand_crop, landmarks)
                prediction, confidence = self.predict(image_input, landmarks_input)

                # Update word tracker
                current_word, word_finalized, letter_progress = self.word_tracker.add_prediction(
                    prediction, confidence)

                # Get suggestions
                stats = self.word_tracker.get_stats()
                word_suggestions = stats.get('word_suggestions', [])

                # Draw UI
                frame = self.draw_ui(frame, prediction, confidence, results,
                                   current_word, word_finalized, letter_progress, word_suggestions)

                # Display frame
                cv2.imshow('ASL with Amharic Translation', frame)

                # Handle word completion
                if word_finalized and current_word:
                    self.handle_word_completion(current_word)
                    self.word_tracker.reset_word()

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('0'):  # Auto-complete
                    completed = self.word_tracker.auto_complete_word()
                    if completed:
                        self.handle_word_completion(completed)
                        self.word_tracker.reset_word()
                elif key == ord('r'):  # Reset
                    logger.info("Resetting current word")
                    self.word_tracker.reset_word()
                elif key == ord('l'):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks
                    logger.info(f"Landmarks {'enabled' if self.show_landmarks else 'disabled'}")
                elif key == ord('1'):  # Suggestion 1
                    selected = self.word_tracker.select_suggestion(0)
                    if selected:
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()
                elif key == ord('2'):  # Suggestion 2
                    selected = self.word_tracker.select_suggestion(1)
                    if selected:
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()
                elif key == ord('3'):  # Suggestion 3
                    selected = self.word_tracker.select_suggestion(2)
                    if selected:
                        self.handle_word_completion(selected)
                        self.word_tracker.reset_word()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up...")

        if self.tts_engine:
            self.tts_engine.stop()
        if self.amharic_translator:
            # No cleanup needed for current implementation
            pass
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

        cv2.destroyAllWindows()


def diagnose_system():
    """Run comprehensive system diagnostic with better error handling."""
    print("\n" + "="*60)
    print("ASL SYSTEM DIAGNOSTIC")
    print("="*60)

    # Check core dependencies
    deps = [
        ('OpenCV', 'cv2', 'pip install opencv-python'),
        ('MediaPipe', 'mediapipe', 'pip install mediapipe'),
        ('TensorFlow', 'tensorflow', 'pip install tensorflow'),
        ('NumPy', 'numpy', 'pip install numpy')
    ]

    for name, module, install_cmd in deps:
        try:
            __import__(module)
            print(f"✓ {name}: INSTALLED")
        except ImportError:
            print(f"✗ {name}: MISSING - {install_cmd}")

    # Check optional dependencies
    print("\nOptional Dependencies:")

    # TTS dependencies
    try:
        import gtts
        print("✓ Google TTS: INSTALLED")
    except ImportError:
        print("✗ Google TTS: MISSING - pip install gtts")

    try:
        import pygame
        print("✓ Pygame (audio): INSTALLED")
    except ImportError:
        print("✗ Pygame (audio): MISSING - pip install pygame")

    # Translation with better error handling
    translation_available = False

    # Try deep-translator first (more reliable)
    try:
        from deep_translator import GoogleTranslator
        print("✓ Deep Translator: INSTALLED")

        # Test with timeout
        try:
            translator = GoogleTranslator(source='en', target='am')
            # Use a simple timeout mechanism
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Translation test timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(3)  # 3 second timeout

            test_result = translator.translate("hello")
            signal.alarm(0)  # Cancel alarm

            if test_result and test_result != "hello":
                print("✓ Deep Translator: WORKING (online test passed)")
                translation_available = True
            else:
                print("✓ Deep Translator: INSTALLED (online test failed - may work offline)")
                translation_available = True  # Still mark as available

        except (TimeoutError, Exception) as e:
            signal.alarm(0)  # Cancel alarm
            print(f"✓ Deep Translator: INSTALLED (network test failed: {str(e)[:50]}...)")
            translation_available = True  # Still mark as available

    except ImportError:
        print("✗ Deep Translator: MISSING - pip install deep-translator")

    # Try googletrans as backup
    if not translation_available:
        try:
            from googletrans import Translator
            print("✓ Google Translate (googletrans): INSTALLED")

            # Test with timeout protection
            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Translation test timeout")

                translator = Translator()
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(3)  # 3 second timeout

                result = translator.translate("hello", dest='am')
                signal.alarm(0)  # Cancel alarm

                if result and result.text and result.text != "hello":
                    print("✓ Google Translate (googletrans): WORKING (online test passed)")
                    translation_available = True
                else:
                    print("✓ Google Translate (googletrans): INSTALLED (test inconclusive)")
                    translation_available = True

            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel alarm
                if "httpcore" in str(e):
                    print("✗ Google Translate (googletrans): DEPENDENCY CONFLICT")
                    print("  Fix: pip install httpcore==0.15.0")
                else:
                    print(f"✓ Google Translate (googletrans): INSTALLED (network test failed)")
                    translation_available = True  # Still mark as available for offline use

        except AttributeError as e:
            if "httpcore" in str(e):
                print("✗ Google Translate (googletrans): DEPENDENCY CONFLICT")
                print("  Fix: pip install httpcore==0.15.0")
            else:
                print(f"✗ Google Translate (googletrans): ERROR - {e}")
        except ImportError:
            print("✗ Google Translate (googletrans): MISSING - pip install googletrans==4.0.0rc1")
        except Exception as e:
            print(f"✗ Google Translate (googletrans): ERROR - {e}")

    # System TTS
    espeak_available = bool(shutil.which('espeak'))
    say_available = bool(shutil.which('say'))
    print(f"{'✓' if espeak_available else '✗'} espeak: {'AVAILABLE' if espeak_available else 'MISSING'}")
    print(f"{'✓' if say_available else '✗'} say (macOS): {'AVAILABLE' if say_available else 'MISSING'}")

    print("="*60)

    print("Recommended setup commands:")
    print("# Core dependencies")
    print("pip install opencv-python mediapipe tensorflow gtts pygame")
    print()
    print("# Translation (recommended - more stable):")
    print("pip install deep-translator")
    print()
    print("# Translation (alternative if above fails):")
    print("pip install googletrans==4.0.0rc1 httpcore==0.15.0")
    print()
    print("# System TTS (Linux)")
    print("sudo apt install espeak")
    print("="*60)

    if not translation_available:
        print("\nWARNING: No translation service available.")
        print("Amharic translation will use offline dictionary only (limited vocabulary).")
    else:
        print("\n✓ Translation service is available!")

    print("="*60 + "\n")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Enhanced ASL Inference with Auto-Complete and Amharic")

    # File arguments
    parser.add_argument('--model', default='export/asl_model.tflite',
                       help='Path to TensorFlow Lite model')
    parser.add_argument('--metadata', default='processed_asl/metadata.json',
                       help='Path to metadata JSON file')

    # Hardware
    parser.add_argument('--camera', type=int, default=0, help='Camera index')

    # Features
    parser.add_argument('--no-speech', action='store_true', help='Disable TTS')
    parser.add_argument('--no-google-tts', action='store_true', help='Use only system TTS')
    parser.add_argument('--translate-amharic', action='store_true', help='Enable Amharic translation')
    parser.add_argument('--show-landmarks', action='store_true', help='Show hand landmarks')

    # Performance tuning
    parser.add_argument('--letter-hold-time', type=float, default=0.8,
                       help='Seconds to hold letter (default: 0.8)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Minimum confidence (default: 0.6)')

    # Utilities
    parser.add_argument('--diagnose', action='store_true', help='Run system diagnostic')

    args = parser.parse_args()

    if args.diagnose:
        diagnose_system()
        return

    # Validate files
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Run diagnostic if using advanced features
    if not args.no_speech or args.translate_amharic:
        print("\nRunning system diagnostic...")
        diagnose_system()

    # Create and run inference system
    try:
        logger.info("="*60)
        logger.info("ENHANCED ASL INFERENCE SYSTEM WITH AMHARIC TRANSLATION")
        logger.info("="*60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Metadata: {args.metadata}")
        logger.info(f"Features: Speech={'ON' if not args.no_speech else 'OFF'}, "
                   f"Amharic={'ON' if args.translate_amharic else 'OFF'}")
        logger.info(f"Performance: Letter hold={args.letter_hold_time}s, "
                   f"Confidence={args.confidence_threshold}")
        logger.info("="*60)

        # Initialize system
        system = ASLRealTimeInference(
            model_path=args.model,
            metadata_path=args.metadata,
            camera_index=args.camera,
            enable_speech=not args.no_speech,
            use_google_tts=not args.no_google_tts,
            show_landmarks=args.show_landmarks,
            enable_amharic=args.translate_amharic
        )

        # Configure word tracker
        system.word_tracker = WordTracker(
            min_letter_duration=args.letter_hold_time,
            confidence_threshold=args.confidence_threshold
        )

        # Initialize components
        system.load_model()
        system.load_metadata()
        system.initialize_mediapipe()
        system.initialize_camera()

        logger.info("System ready! Enhanced features:")
        logger.info("- Press '0' to auto-complete current word instantly")
        logger.info("- Press '1', '2', '3' to select word suggestions")
        logger.info(f"- Hold letters for {args.letter_hold_time}s to add them")
        if args.translate_amharic:
            logger.info("- English words translated to Amharic automatically")
            logger.info("- Both English and Amharic pronunciation via TTS")

        # Run main loop
        system.run()

    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()