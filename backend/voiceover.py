import asyncio
import hashlib
import logging
import shutil
import time
from pathlib import Path

from mutagen.mp3 import MP3

from . import database

logger = logging.getLogger("uvicorn.error")


VOICES = {
    ("English", "female"): "en-US-JennyNeural",
    ("English", "male"): "en-US-GuyNeural",
    ("Hindi", "female"): "hi-IN-SwaraNeural",
    ("Hindi", "male"): "hi-IN-MadhurNeural",
}


def generate_voice(job: dict, scene: dict, index: int, destination: Path) -> float:
    if destination.exists() and destination.stat().st_size > 1024:
        return MP3(destination).info.length
    rate = "+12%" if job["mode"] == "shorts" else "+0%"
    voice = VOICES[(job["language"], job["voice"])]
    identity = "|".join((scene["narration_text"], voice, rate))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    cache_path = database.DATA_DIR.parent / "voice_cache" / f"{digest}.mp3"
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_path, destination)
        return MP3(destination).info.length
    import edge_tts

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
    communicator = edge_tts.Communicate(scene["narration_text"], voice, rate=rate)
    for attempt in range(4):
        try:
            asyncio.run(communicator.save(str(temporary)))
            duration = MP3(temporary).info.length
            temporary.replace(destination)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached_temporary = cache_path.with_name(f"{cache_path.stem}.{index}.partial.mp3")
            shutil.copyfile(destination, cached_temporary)
            cached_temporary.replace(cache_path)
            return duration
        except Exception:
            logger.exception("Voice generation attempt %d failed for scene %d", attempt + 1, index + 1)
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Voice generation failed after retries")