from gtts import gTTS
import os

tts = gTTS("Hello world", lang="en")
tts.save("hello.mp3")
os.system("mpv hello.mp3")  # or any available audio player
