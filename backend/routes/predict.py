from fastapi import APIRouter
from ai_engine.recognition.unified_realtime import predict_gesture
from ai_engine.nlp.sentence_builder import correct_sentence
from gtts import gTTS
import uuid
import os

router = APIRouter()

sentence_buffer = []

def speak(text):
    try:
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
        return filename
    except:
        return None


@router.get("/predict-dynamic")
def run_dynamic():

    word = predict_gesture()

    if word:
        sentence_buffer.append(word)

    clean_sentence = correct_sentence(sentence_buffer)

    audio = speak(clean_sentence)

    return {
        "word": word,
        "sentence": clean_sentence,
        "audio": audio
    }


@router.get("/predict-static")
def run_static():

    letter = predict_gesture()

    audio = speak(letter)

    return {
        "letter": letter,
        "audio": audio
    }