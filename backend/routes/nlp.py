from fastapi import APIRouter
from ai_engine.nlp.sentence_builder import correct_sentence

router = APIRouter()

sentence_buffer = []

@router.post("/nlp/add-word")
def add_word(word: str):

    sentence_buffer.append(word)

    corrected = correct_sentence(sentence_buffer)

    return {
        "sentence": corrected
    }


@router.get("/nlp/get")
def get_sentence():

    corrected = correct_sentence(sentence_buffer)

    return {
        "sentence": corrected
    }


@router.post("/nlp/reset")
def reset_sentence():

    sentence_buffer.clear()

    return {
        "status": "sentence cleared"
    }