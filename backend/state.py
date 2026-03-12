from threading import Lock


_mode_lock = Lock()
_current_mode = "live"


def get_mode() -> str:
    with _mode_lock:
        return _current_mode


def set_mode(mode: str) -> str:
    if mode not in {"live", "record"}:
        raise ValueError("Mode must be 'live' or 'record'.")

    global _current_mode
    with _mode_lock:
        _current_mode = mode
        return _current_mode
