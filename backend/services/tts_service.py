from pathlib import Path
import time
import uuid
from typing import Optional

from gtts import gTTS


ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT_DIR / "backend" / "media" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_audio_files(max_age_seconds: int = 3600) -> None:
    cutoff = time.time() - max_age_seconds
    for item in AUDIO_DIR.glob("*.mp3"):
        try:
            if item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
        except OSError:
            continue


def synthesize_to_audio_url(text: str) -> Optional[str]:
    clean_text = text.strip()
    if not clean_text:
        return None

    _cleanup_audio_files()

    filename = f"speech_{int(time.time())}_{uuid.uuid4().hex}.mp3"
    output_path = AUDIO_DIR / filename

    try:
        tts = gTTS(text=clean_text, lang="en")
        tts.save(str(output_path))
    except Exception:
        output_path.unlink(missing_ok=True)
        return None

    return f"/media/audio/{filename}"
