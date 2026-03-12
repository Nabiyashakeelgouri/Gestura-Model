import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _parse_csv(value: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if not value:
        return default

    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items) if items else default


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_dotenv_file()


@dataclass(frozen=True)
class Settings:
    jwt_secret: str
    jwt_algorithm: str
    access_token_expire_minutes: int

    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    smtp_use_tls: bool

    delete_confirm_base_url: str
    delete_token_expire_minutes: int
    allow_delete_link_fallback: bool

    cors_origins: Tuple[str, ...]

    activation_hold_frames: int
    no_hand_cooldown_frames: int
    cooldown_seconds: float
    require_checkmark_activation: bool

    secure_temp_dir: str
    record_encryption_key: str
    secure_delete_passes: int


def load_settings() -> Settings:
    return Settings(
        jwt_secret=os.getenv("JWT_SECRET", "change-me-in-env"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "28800")),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_pass=os.getenv("SMTP_PASS", ""),
        smtp_use_tls=_parse_bool(os.getenv("SMTP_USE_TLS"), True),
        delete_confirm_base_url=os.getenv(
            "DELETE_CONFIRM_BASE_URL",
            "http://127.0.0.1:8000/user/delete/confirm",
        ),
        delete_token_expire_minutes=int(os.getenv("DELETE_TOKEN_EXPIRE_MINUTES", "30")),
        allow_delete_link_fallback=_parse_bool(os.getenv("ALLOW_DELETE_LINK_FALLBACK"), False),
        cors_origins=_parse_csv(
            os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500,null"),
            ("http://127.0.0.1:5500", "http://localhost:5500", "null"),
        ),
        activation_hold_frames=int(os.getenv("ACTIVATION_HOLD_FRAMES", "6")),
        no_hand_cooldown_frames=int(os.getenv("NO_HAND_COOLDOWN_FRAMES", "12")),
        cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "1.5")),
        require_checkmark_activation=_parse_bool(
            os.getenv("REQUIRE_CHECKMARK_ACTIVATION"),
            False,
        ),
        secure_temp_dir=os.getenv("SECURE_TEMP_DIR", ""),
        record_encryption_key=os.getenv("RECORD_ENCRYPTION_KEY", ""),
        secure_delete_passes=int(os.getenv("SECURE_DELETE_PASSES", "1")),
    )


settings = load_settings()

