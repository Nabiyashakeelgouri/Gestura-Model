from fastapi import APIRouter
from ai_engine.tts.tts_engine import speak

router = APIRouter()

@router.post("/speak")
def speak_text(text: str):

    speak(text)

    return {
        "status": "spoken",
        "text": text
    }