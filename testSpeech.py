"""
Text-to-Speech (TTS) Test Script.

This is a simple script to verify that the `gTTS` (Google Text-to-Speech)
library is installed and that an audio player is available on the system to
play the output.

It performs two actions:
1.  Uses `gTTS` to generate an MP3 file named `hello.mp3` from the text
    "Hello world".
2.  Uses the `os.system` command to play the generated MP3 file. You may need
    to change "mpv" to your preferred command-line audio player (e.g., "mpg123",
    "afplay" on macOS).

This script is useful for quickly diagnosing TTS setup issues.

Requires:
- `gtts`: Install with `pip install gtts`.
- A command-line audio player (e.g., `mpv`, `mpg123`, `afplay`).
"""
from gtts import gTTS
import os

# The text to be converted to speech.
text_to_speak = "Hello world"

# The language of the text.
language = "en"

# The output filename for the MP3 file.
output_filename = "hello.mp3"

print(f"Generating speech for: '{text_to_speak}'")

# Create a gTTS object.
tts = gTTS(text_to_speak, lang=language)

# Save the speech to an MP3 file.
tts.save(output_filename)
print(f"Speech saved to {output_filename}")

# Play the generated audio file using a system command.
# Note: You might need to change "mpv" to a player available on your system,
# such as "mpg123" (Linux) or "afplay" (macOS).
player_command = f"mpv {output_filename}"
print(f"Attempting to play audio with command: '{player_command}'")
try:
    os.system(player_command)
    print("Playback command executed.")
except Exception as e:
    print(f"Could not play audio. Please ensure you have a command-line MP3 player installed. Error: {e}")

# Optional: Clean up the generated MP3 file.
# import time
# time.sleep(2) # Wait for playback to likely finish
# os.remove(output_filename)
# print(f"Cleaned up {output_filename}.")