import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from backend.config import settings

logger = logging.getLogger("gestura.secure_files")
_missing_crypto_warned = False


def _secure_temp_root() -> Path:
    if settings.secure_temp_dir:
        root = Path(settings.secure_temp_dir)
    else:
        root = Path(__file__).resolve().parents[1] / "secure_temp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_fernet() -> Optional[object]:
    global _missing_crypto_warned
    try:
        from cryptography.fernet import Fernet
    except ModuleNotFoundError:
        if not _missing_crypto_warned:
            logger.warning(
                "cryptography is not installed. Record-mode uploads will use plaintext temp files. "
                "Install with: pip install cryptography"
            )
            _missing_crypto_warned = True
        return None

    if settings.record_encryption_key:
        key = settings.record_encryption_key.encode("utf-8")
    else:
        # Ephemeral key for this process if no env key is configured.
        key = Fernet.generate_key()
        logger.warning(
            "RECORD_ENCRYPTION_KEY is not set. Using ephemeral key for runtime-only encryption."
        )
    return Fernet(key)


def _random_prefix() -> str:
    return f"gestura_{secrets.token_hex(8)}_"


def write_encrypted_upload(upload_bytes: bytes, suffix: str) -> Tuple[str, Optional[object]]:
    secure_root = _secure_temp_root()
    fernet = _resolve_fernet()
    encrypted_payload = fernet.encrypt(upload_bytes) if fernet else upload_bytes

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"{suffix}.enc" if fernet else suffix,
        prefix=_random_prefix(),
        dir=str(secure_root),
    ) as temp_file:
        temp_file.write(encrypted_payload)
        encrypted_path = temp_file.name

    return encrypted_path, fernet


def decrypt_to_temp_video(encrypted_path: str, fernet: Optional[object], suffix: str) -> str:
    secure_root = _secure_temp_root()
    encrypted_bytes = Path(encrypted_path).read_bytes()
    decrypted_bytes = fernet.decrypt(encrypted_bytes) if fernet else encrypted_bytes

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=_random_prefix(),
        dir=str(secure_root),
    ) as temp_file:
        temp_file.write(decrypted_bytes)
        plain_video_path = temp_file.name

    return plain_video_path


def secure_delete(path: str, passes: int = None) -> None:
    if not path:
        return

    file_path = Path(path)
    if not file_path.exists():
        return

    overwrite_passes = settings.secure_delete_passes if passes is None else max(1, passes)

    try:
        size = file_path.stat().st_size
        for _ in range(overwrite_passes):
            with open(file_path, "r+b") as handle:
                handle.seek(0)
                handle.write(os.urandom(size))
                handle.flush()
        file_path.unlink(missing_ok=True)
    except Exception:
        # Fall back to standard deletion when overwrite is not possible.
        file_path.unlink(missing_ok=True)
