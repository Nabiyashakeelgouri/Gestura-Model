from fastapi import FastAPI
from .database import engine, Base
from .auth import router as auth_router

app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

# Include authentication routes
app.include_router(auth_router)

@app.get("/")
def root():
    return {"message": "Gestura Backend Running"}