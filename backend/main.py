from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2

# Import your inference pipeline
from ai_engine.recognition.realtime_dynamic_predict import predict_dynamic
from ai_engine.preprocessing.landmark_extractor import extract_landmarks

app = FastAPI()


class ImageData(BaseModel):
    image: str


@app.post("/predict")
async def predict(data: ImageData):

    # Remove base64 header
    image_data = data.image.split(",")[1]

    # Decode base64
    img_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(img_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Extract landmarks
    landmarks = extract_landmarks(frame)

    if landmarks is None:
        return {"prediction": "No hand detected"}

    # Run model prediction
    prediction = predict_dynamic(landmarks)

    return {"prediction": prediction}