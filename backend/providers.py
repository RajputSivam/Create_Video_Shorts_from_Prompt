import asyncio
import json
import logging
import os
import random
import re
import time

import httpx
from dotenv import load_dotenv

from .database import ROOT

load_dotenv(ROOT / ".env")
HTTP_TIMEOUT = httpx.Timeout(150, connect=20)
logger = logging.getLogger("uvicorn.error")


def _read_key(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        if value:
            return value
    return None


def _gemini_api_key() -> str | None:
    return _read_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _groq_api_key() -> str | None:
    return _read_key("GROQ_API_KEY")


GEMINI_MODEL = _read_key("GEMINI_MODEL") or "gemini-3.1-flash-lite"
GROQ_MODEL = _read_key("GROQ_MODEL") or "openai/gpt-oss-20b"




def log_provider_key_status() -> None:
    logger.info(
        "Provider API keys: Gemini=%s, Groq=%s",
        "loaded" if _gemini_api_key() else "missing",
        "loaded" if _groq_api_key() else "missing",
    )


def _extract_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def _safe_response_body(response: httpx.Response, key: str | None) -> str:
    body = response.text
    return body.replace(key, "[REDACTED]") if key else body


def _response_error_message(response: httpx.Response, key: str | None) -> str:
    body = _safe_response_body(response, key)
    try:
        error = response.json().get("error", {})
        message = error.get("message") if isinstance(error, dict) else None
    except ValueError:
        message = None
    if isinstance(message, str) and message.strip():
        return message.replace(key, "[REDACTED]") if key else message
    return body


def _raise_for_provider_status(provider: str, response: httpx.Response, key: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error(
            "%s API HTTP %s response: %s",
            provider,
            response.status_code,
            _safe_response_body(response, key),
        )
        raise


async def _gemini(prompt: str) -> str:
    key = _gemini_api_key()
    if not key:
        raise RuntimeError("Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in the project-root .env file.")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.75}},
        )
        _raise_for_provider_status("Gemini", response, key)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _groq(prompt: str) -> str:
    key = _groq_api_key()
    if not key:
        raise RuntimeError("Groq API key is missing. Set GROQ_API_KEY in the project-root .env file.")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.75},
        )
        _raise_for_provider_status("Groq", response, key)
        return response.json()["choices"][0]["message"]["content"]


async def generate_text(prompt: str) -> str:
    providers = [_gemini, _groq]
    failures = []
    for provider in providers:
        for attempt in range(4):
            try:
                return await provider(prompt)
            except httpx.HTTPStatusError as error:
                name = "Gemini" if provider is _gemini else "Groq"
                key = _gemini_api_key() if provider is _gemini else _groq_api_key()
                message = _response_error_message(error.response, key)
                failures.append(f"{name} returned HTTP {error.response.status_code}: {message}")
                if error.response.status_code not in (408, 429, 500, 502, 503, 504):
                    break
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt + random.random(), 15))
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                failures.append(f"{provider.__name__}: {error.__class__.__name__}")
                if attempt < 3:
                    await asyncio.sleep(min(2 ** attempt + random.random(), 15))
            except (KeyError, IndexError, ValueError, RuntimeError) as error:
                failures.append(f"{provider.__name__}: {error}")
                break
    if not _gemini_api_key() and not _groq_api_key():
        raise RuntimeError(
            "No provider API key is loaded: Gemini needs GEMINI_API_KEY or GOOGLE_API_KEY, "
            "and Groq needs GROQ_API_KEY in the project-root .env file."
        )
    raise RuntimeError("Text generation failed with all configured providers. " + "; ".join(failures[-4:]))


def generate_json(prompt: str):
    return _extract_json(asyncio.run(generate_text(prompt)))


def request_with_retry(url: str, destination, params: dict | None = None) -> None:
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(5):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                with client.stream("GET", url, params=params) as response:
                    response.raise_for_status()
                    with open(temporary, "wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
            temporary.replace(destination)
            return
        except (httpx.HTTPError, OSError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt + random.random(), 20))