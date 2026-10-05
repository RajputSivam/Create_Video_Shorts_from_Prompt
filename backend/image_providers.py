import base64
import logging
import os
from typing import Protocol
from urllib.parse import quote

import httpx

from .providers import _gemini_api_key, _read_key

logger = logging.getLogger("uvicorn.error")
HTTP_TIMEOUT = httpx.Timeout(120, connect=20)


class ImageProvider(Protocol):
    name: str
    attribution: dict | None

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        ...


def _safe_body(response: httpx.Response, key: str | None = None) -> str:
    body = response.text
    return body.replace(key, "[REDACTED]") if key else body


def _raise_for_status(name: str, response: httpx.Response, key: str | None = None) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("%s image API HTTP %s response: %s", name, response.status_code, _safe_body(response, key))
        raise


def _json_image(response: httpx.Response) -> bytes:
    payload = response.json()
    image = payload.get("result", payload).get("image")
    if not image:
        raise ValueError("Provider response did not contain an image")
    return base64.b64decode(image)


class PollinationsProvider:
    name = "Pollinations"
    attribution = None

    def __init__(self):
        self.key = _read_key("POLLINATIONS_API_KEY")

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        base_url = os.getenv("POLLINATIONS_BASE_URL", "https://gen.pollinations.ai/image").strip().rstrip("/")
        if "image.pollinations.ai" in base_url:
            base_url = "https://gen.pollinations.ai/image"
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                f"{base_url}/{quote(prompt, safe='')}",
                params={"model": "flux", "width": width, "height": height},
                headers=headers,
            )
        _raise_for_status(self.name, response, self.key)
        if "json" in response.headers.get("content-type", ""):
            return _json_image(response)
        return response.content


class CloudflareProvider:
    name = "Cloudflare Workers AI"
    attribution = None

    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(
                url,
                headers={"Authorization": f"Bearer {self.token}"},
                json={"prompt": prompt, "steps": 4},
            )
        _raise_for_status(self.name, response, self.token)
        if "json" in response.headers.get("content-type", ""):
            return _json_image(response)
        return response.content


class GeminiImageProvider:
    name = "Gemini"
    attribution = None

    def __init__(self, key: str):
        self.key = key
        self.model = _read_key("GEMINI_IMAGE_MODEL") or "gemini-3.1-flash-image"

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        aspect_ratio = "9:16" if height > width else "16:9"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(
                url,
                headers={"x-goog-api-key": self.key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": "1K"},
                    },
                },
            )
        _raise_for_status(self.name, response, self.key)
        for part in response.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
            image = part.get("inlineData") or part.get("inline_data")
            if image and image.get("data"):
                return base64.b64decode(image["data"])
        raise ValueError("Gemini response did not contain generated image data")


class PexelsProvider:
    name = "Pexels"

    def __init__(self, key: str):
        self.key = key
        self.attribution: dict | None = None

    def generate_image(self, prompt: str, width: int, height: int) -> bytes:
        orientation = "portrait" if height > width else "landscape"
        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            response = client.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": self.key},
                params={"query": prompt, "orientation": orientation, "per_page": 1},
            )
            _raise_for_status(self.name, response, self.key)
            photos = response.json().get("photos", [])
            if not photos:
                raise ValueError("Pexels returned no photos for this scene")
            photo = photos[0]
            image_url = photo.get("src", {}).get(orientation) or photo.get("src", {}).get("large")
            if not image_url:
                raise ValueError("Pexels response did not contain a downloadable photo")
            image = client.get(image_url)
            _raise_for_status(self.name, image, self.key)
        self.attribution = {
            "photographer": photo.get("photographer"),
            "photographer_url": photo.get("photographer_url"),
            "photo_url": photo.get("url"),
        }
        return image.content


def configured_providers() -> list[ImageProvider]:
    providers: list[ImageProvider] = []
    if _read_key("POLLINATIONS_API_KEY"):
        providers.append(PollinationsProvider())
    cloudflare_token = _read_key("CLOUDFLARE_API_TOKEN")
    cloudflare_account = _read_key("CLOUDFLARE_ACCOUNT_ID")
    if cloudflare_token and cloudflare_account:
        providers.append(CloudflareProvider(cloudflare_token, cloudflare_account))
    gemini_key = _gemini_api_key()
    if gemini_key:
        providers.append(GeminiImageProvider(gemini_key))
    pexels_key = _read_key("PEXELS_API_KEY")
    if pexels_key:
        providers.append(PexelsProvider(pexels_key))
    return providers