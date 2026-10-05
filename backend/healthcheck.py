import asyncio
import io
import logging
import subprocess
import tempfile
import threading
from pathlib import Path

import edge_tts
import httpx
from mutagen.mp3 import MP3
from PIL import Image

from .image_providers import PollinationsProvider
from .providers import GEMINI_MODEL, GROQ_MODEL, _gemini_api_key, _groq_api_key
from .renderer import _ffmpeg, _ffprobe

logger = logging.getLogger("uvicorn.error")
_lock = threading.Lock()
_checks = {
    name: {"status": "checking", "detail": "Self-test has not completed."}
    for name in ("ffmpeg", "ffprobe", "edge_tts", "pollinations", "gemini", "groq")
}
_health_status = "checking"


def get_health_status() -> dict:
    with _lock:
        return {"status": _health_status, "checks": {name: dict(result) for name, result in _checks.items()}}


def _set_check(name: str, status: str, detail: str) -> None:
    global _health_status
    with _lock:
        _checks[name] = {"status": status, "detail": detail}
        if all(result["status"] != "checking" for result in _checks.values()):
            _health_status = "ok" if all(result["status"] == "passed" for result in _checks.values()) else "degraded"


def _probe_executable(executable: str | None, label: str) -> tuple[str, str]:
    if not executable:
        return "failed", f"{label} executable was not found."
    result = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError(f"{label} returned {result.returncode}: {result.stderr}")
    version = (result.stdout or result.stderr).splitlines()[0]
    return "passed", f"{version} ({executable})"


def _find_ffprobe() -> str | None:
    try:
        return _ffprobe()
    except RuntimeError:
        return None


def _test_tts() -> str:
    with tempfile.TemporaryDirectory(prefix="prompttovideo-health-") as folder:
        source_path = Path(folder) / "edge-tts-test.mp3"
        audio_path = Path(folder) / "health-test.mp3"
        communicator = edge_tts.Communicate(
            "Test.",
            "en-US-JennyNeural",
        )
        asyncio.run(communicator.save(str(source_path)))
        command = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source_path), "-t", "1", "-c:a", "libmp3lame", "-q:a", "9", str(audio_path),
        ]
        logger.info("One-second Edge TTS self-test command: %s", subprocess.list2cmdline(command))
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
        if result.returncode:
            logger.error("One-second Edge TTS trim failed with exit code %s; stderr:\n%s", result.returncode, result.stderr)
            raise RuntimeError(f"Could not create one second of test audio: {result.stderr}")
        duration = MP3(audio_path).info.length
        if not 0.9 <= duration <= 1.2:
            raise RuntimeError(f"Edge TTS produced an invalid audio duration: {duration}")
        return f"Generated and decoded {duration:.2f}s of Edge TTS audio."


def _test_pollinations() -> str:
    provider = PollinationsProvider()
    image_bytes = provider.generate_image("A blue sky", 96, 64)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
        return f"Generated a valid {image.width}x{image.height} test image."


def _test_text_provider(name: str, url: str, key: str | None, model: str, secret_header: str) -> tuple[str, str]:
    if not key:
        return "not_configured", "API key is not configured."
    headers = {"x-goog-api-key": key} if secret_header == "google" else {"Authorization": f"Bearer {key}"}
    if secret_header == "google":
        payload = {
            "contents": [{"parts": [{"text": "Reply OK."}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 4},
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "temperature": 0,
            "max_tokens": 4,
        }
    response = httpx.post(url, headers=headers, json=payload, timeout=httpx.Timeout(20, connect=5))
    response.raise_for_status()
    return "passed", f"Key accepted by {name} ({model})."


def _run_check(name: str, check) -> None:
    try:
        status, detail = check()
    except Exception as error:
        logger.exception("Startup self-test failed: %s", name)
        if isinstance(error, httpx.HTTPStatusError):
            secret = _gemini_api_key() if name == "gemini" else _groq_api_key()
            body = error.response.text.replace(secret, "[REDACTED]") if secret else error.response.text
            detail = f"HTTP {error.response.status_code}: {body}"
        else:
            detail = f"{type(error).__name__}: {error}"
        _set_check(name, "failed", detail)
    else:
        _set_check(name, status, detail)
    logger.info("Startup self-test %s: %s", name, get_health_status()["checks"][name])


def run_startup_self_test() -> None:
    import imageio_ffmpeg

    _run_check("ffmpeg", lambda: _probe_executable(imageio_ffmpeg.get_ffmpeg_exe(), "FFmpeg"))
    _run_check("ffprobe", lambda: _probe_executable(_find_ffprobe(), "ffprobe"))
    _run_check("edge_tts", lambda: ("passed", _test_tts()))
    _run_check("pollinations", lambda: ("passed", _test_pollinations()))
    _run_check(
        "gemini",
        lambda: _test_text_provider(
            "Gemini",
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            _gemini_api_key(),
            GEMINI_MODEL,
            "google",
        ),
    )
    _run_check(
        "groq",
        lambda: _test_text_provider(
            "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            _groq_api_key(),
            GROQ_MODEL,
            "bearer",
        ),
    )
