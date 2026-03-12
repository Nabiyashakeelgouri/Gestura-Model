from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.models import User
from backend.schemas import ModeResponse, ModeUpdate
from backend.state import get_mode, set_mode

router = APIRouter(tags=["mode"])


@router.get("/mode", response_model=ModeResponse)
def get_mode_state(_: User = Depends(get_current_user)) -> ModeResponse:
    return ModeResponse(mode=get_mode())


@router.post("/mode", response_model=ModeResponse)
def set_mode_state(
    payload: ModeUpdate,
    _: User = Depends(get_current_user),
) -> ModeResponse:
    try:
        mode = set_mode(payload.mode)
        return ModeResponse(mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/mode/live", response_model=ModeResponse)
def set_live_mode(_: User = Depends(get_current_user)) -> ModeResponse:
    return ModeResponse(mode=set_mode("live"))


@router.post("/mode/record", response_model=ModeResponse)
def set_record_mode(_: User = Depends(get_current_user)) -> ModeResponse:
    return ModeResponse(mode=set_mode("record"))
