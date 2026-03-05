from fastapi import FastAPI
from .database import engine, Base
from .auth import router as auth_router
from fastapi import APIRouter
from ai_engine.recognition.unified_realtime import predict_gesture
from backend.routes.inference import router as inference_router
from backend.routes.predict import router as predict_router
from backend.routes.predict_static import router as static_router
from backend.routes.nlp import router as nlp_router
from backend.routes.tts import router as tts_router
from backend.routes.mode import router as mode_router
from backend.routes.nlp import router as nlp_router
from backend.routes.user import router as user_router



router = APIRouter()

@router.get("/predict")
def predict():
    result = predict_gesture()
    return {"prediction": result}

app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

# Include authentication routes
app.include_router(auth_router)

app.include_router(inference_router)

@app.get("/")
def root():
    return {"message": "Gestura Backend Running"}

app.include_router(predict_router)

app.include_router(static_router)

app.include_router(nlp_router)

app.include_router(tts_router)

app.include_router(mode_router)

app.include_router(nlp_router)

app.include_router(user_router)
