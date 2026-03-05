from fastapi import APIRouter
from ai_engine.recognition.unified_realtime import predict_gesture

router = APIRouter()

@router.get("/predict")
def predict():
    result = predict_gesture()
    return {"prediction": result}