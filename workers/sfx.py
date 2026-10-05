from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

from .common import job_directory, log


def _make_effect(path: Path, effect: str, seed: int) -> None:
    sample_rate = 24000
    random.seed(seed)
    count = int(sample_rate * 1.4)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        samples = []
        for index in range(count):
            t = index / sample_rate
            envelope = (1 - t / 1.4) ** (2 if effect == "whoosh" else 0.6)
            noise = random.uniform(-1, 1)
            tone = math.sin(2 * math.pi * (130 + t * 180) * t)
            value = (noise * 0.4 + tone * 0.6 if effect == "whoosh" else tone * 0.35) * envelope
            samples.append(struct.pack("<h", int(max(-1, min(1, value * 0.12)) * 32767)))
        audio.writeframes(b"".join(samples))


def generate_sfx(job: dict) -> dict[str, str]:
    target = job_directory(job) / "sfx"
    target.mkdir(parents=True, exist_ok=True)
    effects = {name: target / f"{name}.wav" for name in ("whoosh", "ambience")}
    for index, (name, path) in enumerate(effects.items()):
        if not path.exists():
            _make_effect(path, name, int(job.get("seed", 0)) + index)
    log("sfx", "Generated local transition and ambience effects", count=len(effects))
    return {name: str(path) for name, path in effects.items()}


if __name__ == "__main__":
    from .common import cli_main

    cli_main(generate_sfx)
