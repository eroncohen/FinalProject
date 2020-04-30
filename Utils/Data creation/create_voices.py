from gtts import gTTS
import os

tts = gTTS(text="Ha Ha Ha", lang='en')
tts.save("smile_voice.mp3")

tts = gTTS(text="Smile it is all good", lang='en')
tts.save("sad_voice.mp3")
# to start the file from python
os.system("start smile_voice.mp3")