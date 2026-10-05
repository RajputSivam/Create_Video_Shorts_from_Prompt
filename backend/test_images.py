import re
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

from . import database
from .image_generation import RETRY_DELAYS, _generate_with_retries, _write_image
from .image_providers import _safe_body, configured_providers

PROMPT = "A cinematic mountain lake at sunrise, soft mist, natural colors"


def _provider_key(provider) -> str | None:
    return getattr(provider, "key", None) or getattr(provider, "token", None)


def _print_failure(provider, error: Exception) -> None:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        print(f"{provider.name}: HTTP {response.status_code} {response.reason_phrase}")
        print(_safe_body(response, _provider_key(provider)))
        return
    detail = str(error)
    key = _provider_key(provider)
    if key:
        detail = detail.replace(key, "[REDACTED]")
    detail = re.sub(r"https?://\S+", "[provider URL]", detail)
    print(f"{provider.name}: {type(error).__name__}: {detail}")


def main() -> int:
    providers = configured_providers()
    if not providers:
        print("No image providers are configured.")
        return 1

    output_dir = database.ROOT / "data" / "image-provider-tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    print("Configured image providers: " + ", ".join(provider.name for provider in providers))

    for index, provider in enumerate(providers):
        print(f"Testing {provider.name}...", flush=True)
        try:
            image_bytes = _generate_with_retries(provider, PROMPT, 640, 360)
            image_path = output_dir / f"{re.sub(r'[^a-z0-9]+', '-', provider.name.lower()).strip('-')}.jpg"
            _write_image(image_path, image_bytes)
            with Image.open(image_path) as image:
                image.verify()
            print(f"{provider.name}: success ({image_path.relative_to(database.ROOT)})", flush=True)
        except Exception as error:
            failures += 1
            _print_failure(provider, error)
        if index + 1 < len(providers):
            time.sleep(0.75)

    print(f"Provider test complete: {len(providers) - failures} succeeded, {failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())