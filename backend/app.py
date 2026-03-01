from fastapi import FastAPI
from .database import engine
from .models import Base
from .auth import router as auth_router

app = FastAPI()
app.include_router(auth_router)

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"message": "Gestura Backend Running"}