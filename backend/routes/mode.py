from fastapi import APIRouter

router = APIRouter()

current_mode = "live"

@router.get("/mode")
def get_mode():
    return {"mode": current_mode}


@router.post("/mode/live")
def set_live_mode():
    global current_mode
    current_mode = "live"
    return {"mode": "live activated"}


@router.post("/mode/record")
def set_record_mode():
    global current_mode
    current_mode = "record"
    return {"mode": "record activated"}