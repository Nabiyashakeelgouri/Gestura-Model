import pyttsx3

engine = pyttsx3.init()

def speak(text: str):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS Error:", e)