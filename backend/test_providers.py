import asyncio
import sys

import httpx

from .providers import (
    GEMINI_MODEL,
    GROQ_MODEL,
    _gemini,
    _gemini_api_key,
    _groq,
    _groq_api_key,
    _safe_response_body,
)

PROMPT = "Reply with exactly: provider test successful."


async def _test_provider(name: str, model: str, call, key: str | None) -> bool:
    print(f"Testing {name} model {model}...", flush=True)
    try:
        result = await call(PROMPT)
        print(f"{name}: success: {result.strip()}", flush=True)
        return True
    except httpx.HTTPStatusError as error:
        body = _safe_response_body(error.response, key)
        print(f"{name}: HTTP {error.response.status_code}: {body}", flush=True)
    except Exception as error:
        print(f"{name}: {type(error).__name__}: {error}", flush=True)
    return False


async def main() -> int:
    results = [
        await _test_provider("Gemini", GEMINI_MODEL, _gemini, _gemini_api_key()),
        await _test_provider("Groq", GROQ_MODEL, _groq, _groq_api_key()),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))