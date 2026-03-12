import base64
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ai_engine.recognition.predictor_service import process_frame, process_video
from backend.auth import get_current_user
from backend.config import settings
from backend.models import User
from backend.schemas import FrameRequest, InferenceResponse
from backend.services.secure_file_service import decrypt_to_temp_video, secure_delete, write_encrypted_upload
from backend.services.tts_service import synthesize_to_audio_url
from backend.state import get_mode

router = APIRouter(prefix="/inference", tags=["inference"])
logger = logging.getLogger("gestura.inference")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _decode_frame(image_data: str):
    payload = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")

    return frame


def _build_frame_response(result: dict, mode: str) -> InferenceResponse:
    audio_url = synthesize_to_audio_url(result.get("speak_text", ""))

    return InferenceResponse(
        prediction=result.get("prediction"),
        confidence=result.get("confidence"),
        hand_bbox=result.get("hand_bbox"),
        hand_detected=result.get("hand_detected"),
        mode=mode,
        status=result.get("status", "standby"),
        sentence=result.get("sentence", ""),
        audio_url=audio_url,
        errors=result.get("errors", []),
    )


@router.post("/frame", response_model=InferenceResponse)
def infer_frame(
    payload: FrameRequest,
    _: User = Depends(get_current_user),
) -> InferenceResponse:
    frame = _decode_frame(payload.image)
    mode = get_mode()

    result = process_frame(
        frame=frame,
        session_id=payload.session_id,
        activation_hold_frames=settings.activation_hold_frames,
        no_hand_cooldown_frames=settings.no_hand_cooldown_frames,
        cooldown_seconds=settings.cooldown_seconds,
        require_checkmark_activation=settings.require_checkmark_activation,
        recognition_mode=payload.recognition_mode,
    )

    return _build_frame_response(result=result, mode=mode)


@router.post("/trial/frame", response_model=InferenceResponse)
def infer_trial_frame(payload: FrameRequest) -> InferenceResponse:
    frame = _decode_frame(payload.image)

    result = process_frame(
        frame=frame,
        session_id=f"trial:{payload.session_id}",
        activation_hold_frames=settings.activation_hold_frames,
        no_hand_cooldown_frames=settings.no_hand_cooldown_frames,
        cooldown_seconds=settings.cooldown_seconds,
        require_checkmark_activation=False,
        recognition_mode=payload.recognition_mode,
    )

    return _build_frame_response(result=result, mode="live")


@router.post("/video", response_model=InferenceResponse)
def infer_video(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> InferenceResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type. Allowed: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
        )

    start = time.perf_counter()
    encrypted_path = None
    plain_path = None

    try:
        upload_bytes = file.file.read()
        encrypted_path, fernet = write_encrypted_upload(upload_bytes, suffix=suffix)
        plain_path = decrypt_to_temp_video(encrypted_path, fernet, suffix=suffix)

        result = process_video(plain_path)
        audio_url = synthesize_to_audio_url(result.get("sentence", ""))

        processing_ms = int((time.perf_counter() - start) * 1000)

        return InferenceResponse(
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            hand_bbox=result.get("hand_bbox"),
            hand_detected=result.get("hand_detected"),
            mode=get_mode(),
            status=result.get("status", "completed"),
            sentence=result.get("sentence", ""),
            audio_url=audio_url,
            errors=result.get("errors", []),
            processing_ms=processing_ms,
            frames_processed=result.get("frames_processed"),
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass

        if plain_path and os.path.exists(plain_path):
            secure_delete(plain_path)
            logger.info("Securely deleted temporary plaintext video: %s", plain_path)

        if encrypted_path and os.path.exists(encrypted_path):
            secure_delete(encrypted_path)
            logger.info("Securely deleted encrypted uploaded video: %s", encrypted_path)



