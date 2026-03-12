from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: int
    username: str
    email: EmailStr


class ModeUpdate(BaseModel):
    mode: Literal["live", "record"]


class ModeResponse(BaseModel):
    mode: Literal["live", "record"]


class FrameRequest(BaseModel):
    image: str
    session_id: str = Field(default="default", min_length=1, max_length=64)
    recognition_mode: Literal["hybrid", "static", "dynamic"] = "hybrid"



class DeleteConfirmRequest(BaseModel):
    token: str = Field(min_length=16, max_length=512)


class DeleteRequestResponse(BaseModel):
    status: str
    message: str


class InferenceResponse(BaseModel):
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    hand_bbox: Optional[List[float]] = None
    hand_detected: Optional[bool] = None
    mode: Literal["live", "record"]
    status: str
    sentence: str
    audio_url: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    processing_ms: Optional[int] = None
    frames_processed: Optional[int] = None

