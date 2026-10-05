import hashlib
import io
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock

import httpx
from PIL import Image

from . import database
from .config import STYLE_PREFIXES
from .image_providers import ImageProvider, configured_providers

logger = logging.getLogger("uvicorn.error")
MAX_PROMPT_LENGTH = 300
RETRY_DELAYS = (0.5, 1.0)


@dataclass
class ImageResult:
    path: Path
    provider: str
    placeholder: bool = False
    failures: list[str] | None = None
    attribution: dict | None = None


class ImageProviderChain:
    def __init__(self, providers: list[ImageProvider] | None = None):
        self.providers = providers if providers is not None else configured_providers()
        self.disabled: dict[str, str] = {}
        self.lock = Lock()
        self.semaphores = {provider.name: BoundedSemaphore(1) for provider in self.providers}

    def generate(self, prompt: str, width: int, height: int) -> tuple[bytes, str, dict | None, list[str]]:
        failures = []
        for provider in self.providers:
            with self.semaphores[provider.name]:
                with self.lock:
                    disabled_reason = self.disabled.get(provider.name)
                if disabled_reason:
                    failures.append(f"{provider.name}: skipped for this job after HTTP {disabled_reason}")
                    continue
                try:
                    image = _generate_with_retries(provider, prompt, width, height)
                    return image, provider.name, getattr(provider, "attribution", None), failures
                except Exception as error:
                    message = _failure_message(provider, error)
                    failures.append(message)
                    detail = _full_failure_detail(provider, error)
                    logger.exception("Image provider failed: %s", detail)
                    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (402, 403, 429):
                        with self.lock:
                            self.disabled[provider.name] = str(error.response.status_code)
                        logger.warning("Image circuit breaker opened for %s after HTTP %s", provider.name, error.response.status_code)
        return b"", "placeholder", None, failures


def clean_image_prompt(job: dict, scene: dict) -> str:
    style = STYLE_PREFIXES.get(job.get("style"), STYLE_PREFIXES["cinematic"])
    framing = "vertical portrait composition" if job.get("mode") == "shorts" else "wide cinematic landscape composition"
    scene_description = scene.get("image_prompt") or scene.get("title") or "cinematic scene"
    prompt = f"{style} {scene_description}, {framing}, no text, no watermark"
    prompt = re.sub(r"[^A-Za-z0-9 ,.;:!?()'/-]", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH].rsplit(" ", 1)[0]
    return prompt


def _cache_paths(prompt: str, width: int, height: int) -> tuple[Path, Path]:
    digest = hashlib.sha256(f"{width}x{height}:{prompt}".encode("utf-8")).hexdigest()
    cache_dir = database.DATA_DIR.parent / "image_cache"
    return cache_dir / f"{digest}.jpg", cache_dir / f"{digest}.json"


def _write_image(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(output.getvalue())
    temporary.replace(path)


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _placeholder(width: int, height: int) -> bytes:
    top = (25, 42, 50)
    bottom = (93, 119, 101)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        blend = y / max(height - 1, 1)
        color = tuple(round(top[channel] * (1 - blend) + bottom[channel] * blend) for channel in range(3))
        for x in range(width):
            pixels[x, y] = color
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()


def _failure_message(provider: ImageProvider, error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        reasons = {
            401: "API key missing or invalid",
            402: "payment required or usage limit reached",
            403: "access denied or provider quota unavailable",
            404: "model or endpoint not available",
            408: "request timed out",
            429: "rate limit or provider quota exceeded",
        }
        status = response.status_code
        detail = reasons.get(status, "provider temporarily unavailable" if status >= 500 else response.reason_phrase)
        message = f"{provider.name}: HTTP {status} ({detail})"
    elif isinstance(error, (httpx.TimeoutException, TimeoutError)):
        message = f"{provider.name}: request timed out"
    else:
        message = f"{provider.name}: {type(error).__name__}: {str(error)[:180]}"
    key = getattr(provider, "key", None) or getattr(provider, "token", None)
    if key:
        message = message.replace(key, "[REDACTED]")
    return re.sub(r"https?://\S+", "provider endpoint", message)


def _full_failure_detail(provider: ImageProvider, error: Exception) -> str:
    detail = f"{provider.name}: {type(error).__name__}: {error}"
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        detail += f"; HTTP {response.status_code}; response body: {response.text}"
    key = getattr(provider, "key", None) or getattr(provider, "token", None)
    return detail.replace(key, "[REDACTED]") if key else detail


def _generate_with_retries(provider: ImageProvider, prompt: str, width: int, height: int) -> bytes:
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            image = provider.generate_image(prompt, width, height)
            if not image:
                raise ValueError("Provider returned an empty image")
            return image
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            retryable = status == 408 or status >= 500
            if not retryable or attempt == len(RETRY_DELAYS):
                raise
        except (httpx.TimeoutException, httpx.NetworkError, TimeoutError):
            if attempt == len(RETRY_DELAYS):
                raise
        if attempt < len(RETRY_DELAYS):
            time.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError("Image provider retries exhausted")


def generate_image(job: dict, scene: dict, index: int, destination: Path, provider_chain: ImageProviderChain | None = None) -> ImageResult:
    width, height = (768, 1280) if job["mode"] == "shorts" else (1280, 720)
    prompt = clean_image_prompt(job, scene)
    cache_image, cache_metadata = _cache_paths(prompt, width, height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene_metadata = destination.with_suffix(".json")

    if destination.exists() and destination.stat().st_size > 1024 and scene_metadata.exists():
        metadata = json.loads(scene_metadata.read_text(encoding="utf-8"))
        if not metadata.get("placeholder"):
            provider_name = metadata.get("provider", "cached")
            logger.info("Reused image for scene %s from cache (%s)", index + 1, provider_name)
            return ImageResult(destination, provider_name, attribution=metadata.get("attribution"))
    if cache_image.exists() and cache_image.stat().st_size > 1024:
        metadata = json.loads(cache_metadata.read_text(encoding="utf-8")) if cache_metadata.exists() else {}
        shutil.copyfile(cache_image, destination)
        _write_json(scene_metadata, metadata)
        provider_name = metadata.get("provider", "disk cache")
        logger.info("Reused image for scene %s from shared cache (%s)", index + 1, provider_name)
        return ImageResult(destination, provider_name, attribution=metadata.get("attribution"))

    chain = provider_chain or ImageProviderChain()
    image, provider_name, attribution, failures = chain.generate(prompt, width, height)
    if image:
        _write_image(cache_image, image)
        metadata = {"provider": provider_name, "prompt_hash": cache_image.stem, "attribution": attribution}
        _write_json(cache_metadata, metadata)
        shutil.copyfile(cache_image, destination)
        _write_json(scene_metadata, metadata)
        logger.info("Image provider succeeded for scene %s: %s", index + 1, provider_name)
        return ImageResult(destination, provider_name, attribution=attribution, failures=failures)
    for failure in failures:
        logger.warning("Image provider unavailable for scene %s: %s", index + 1, failure)

    image = _placeholder(width, height)
    _write_image(destination, image)
    metadata = {"provider": "placeholder", "placeholder": True, "failures": failures}
    _write_json(scene_metadata, metadata)
    logger.warning("Using a text-free gradient for scene %s after all image providers failed: %s", index + 1, "; ".join(failures))
    return ImageResult(destination, "placeholder", placeholder=True, failures=failures)