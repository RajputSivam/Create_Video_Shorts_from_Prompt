from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

from .common import job_directory, log

CHORDS = {
    "calm": (220.00, 261.63, 329.63),
    "epic": (110.00, 164.81, 220.00),
    "happy": (261.63, 329.63, 392.00),
    "sad": (174.61, 207.65, 261.63),
    "mystery": (146.83, 174.61, 220.00),
}


def _write_music(path: Path, duration: float, mood: str, seed: int) -> None:
    sample_rate = 24000
    frequencies = CHORDS.get(mood, CHORDS["calm"])
    random.seed(seed)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        chunk = 24000
        for offset in range(0, int(duration * sample_rate), chunk):
            frames = []
            for sample in range(offset, min(offset + chunk, int(duration * sample_rate))):
                t = sample / sample_rate
                swell = min(1, t / 1.5, (duration - t) / 1.5)
                value = sum(math.sin(2 * math.pi * frequency * t) for frequency in frequencies) / len(frequencies)
                value *= max(0, swell) * 0.12
                packed = struct.pack("<h", int(max(-1, min(1, value)) * 32767))
                frames.append(packed + packed)
            audio.writeframes(b"".join(frames))


def create_music(job: dict, script: dict, total_duration: float) -> Path:
    mood = job.get("music_mood", "auto")
    if mood == "auto":
        moods = [scene.get("mood", "calm") for scene in script["scenes"]]
        mood = max(set(moods), key=moods.count) if moods else "calm"
    if mood not in CHORDS:
        raise ValueError(f"Unsupported music mood: {mood}")
    path = job_directory(job) / "music.wav"
    if not path.exists():
        _write_music(path, total_duration + 1, mood, int(job.get("seed", 0)))
    log("music", "Generated local synthesized mood bed", provider="procedural-local", mood=mood, duration=round(total_duration, 2))
    return path


if __name__ == "__main__":
    from .common import cli_main, load_job

    def _run(job):
        directory = job_directory(job)
        voices = load_job(directory / "voice.json")
        return str(create_music(job, load_job(directory / "script.json"), sum(item["duration"] for item in voices)))

    cli_main(_run)
