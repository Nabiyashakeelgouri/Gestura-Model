import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import models  # noqa: F401 - required so SQLAlchemy registers models
from backend.auth import router as auth_router
from backend.config import settings
from backend.database import Base, engine
from backend.routes.inference import router as inference_router
from backend.routes.mode import router as mode_router
from backend.routes.user import router as user_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Gestura Backend", version="1.0.0")

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

media_dir = Path(__file__).resolve().parent / "media"
media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_dir), name="media")

app.include_router(auth_router)
app.include_router(mode_router)
app.include_router(inference_router)
app.include_router(user_router)


@app.get("/")
def root() -> dict:
    return {"message": "Gestura backend running", "status": "ok"}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logging.getLogger("gestura.api").exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )
