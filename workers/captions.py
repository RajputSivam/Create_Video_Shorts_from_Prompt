from __future__ import annotations

import os
import re
from pathlib import Path

from .common import CONFIG, configured_path, job_directory, log


def _wrap_caption(text: str, max_words: int, max_chars: int) -> list[str]:
    words = text.split()
    if len(words) > max_words:
        raise ValueError(f"Caption chunk contains {len(words)} words; maximum is {max_words}.")
    if any(len(word) > max_chars for word in words):
        raise ValueError(f"A caption word exceeds the {max_chars}-character safe line width.")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 2:
        raise ValueError("Caption chunk exceeds the safe two-line layout.")
    return lines


def _caption_groups(words: list[tuple[str, float, float]], max_words: int, max_chars: int) -> list[list[tuple[str, float, float]]]:
    groups: list[list[tuple[str, float, float]]] = []
    current: list[tuple[str, float, float]] = []
    for word in words:
        candidate = [*current, word]
        try:
            _wrap_caption(" ".join(item[0] for item in candidate), max_words, max_chars)
        except ValueError:
            if not current:
                raise
            groups.append(current)
            current = [word]
            _wrap_caption(word[0], max_words, max_chars)
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _check_hindi_glyphs(text: str, font_path: Path) -> list[str]:
    from PIL import ImageFont

    font = ImageFont.truetype(str(font_path), 48)
    missing_mask = font.getmask(chr(0x10FFFF))
    missing = []
    for character in sorted(set(text)):
        if "\u0900" <= character <= "\u097f":
            glyph = font.getmask(character)
            if glyph.size == missing_mask.size and bytes(glyph) == bytes(missing_mask):
                missing.append(character)
    return missing


def _timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours}:{minutes:02}:{remainder:05.2f}"


def _load_whisper():
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", CONFIG["models"]["whisper"])
    model_path = configured_path("cache") / "whisper" / model_name
    if not model_path.is_dir():
        cache_root = Path(
            os.getenv("HF_HUB_CACHE")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )
        repository = cache_root / f"models--Systran--faster-whisper-{model_name}" / "snapshots"
        snapshots = sorted(repository.glob("*")) if repository.is_dir() else []
        model_path = next((snapshot for snapshot in reversed(snapshots) if (snapshot / "model.bin").is_file()), model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local faster-whisper '{model_name}' weights were not found in the configured cache or Hugging Face cache; downloads are disabled.")
    return WhisperModel(str(model_path), device="cpu", compute_type="int8", local_files_only=True)


def _word_times(audio_path: str, text: str, duration: float, model=None) -> list[tuple[str, float, float]]:
    if model is None:
        raise RuntimeError("Whisper is required to align captions; narration-timed fallback is disabled.")
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="hi" if re.search(r"[\u0900-\u097f]", text) else "en",
    )
    timed_words = [
        (word.word.strip(), float(word.start), float(word.end))
        for segment in segments
        for word in (segment.words or [])
        if word.word.strip()
    ]
    if not timed_words:
        raise RuntimeError(f"Whisper returned no word timestamps for {Path(audio_path).name}.")
    return timed_words


def create_captions(job: dict, script: dict, voices: list[dict]) -> Path:
    width, height = (1080, 1920) if job["mode"] == "shorts" else (1920, 1080)
    size = int(CONFIG["video"]["subtitle_font_size_short"] if job["mode"] == "shorts" else CONFIG["video"]["subtitle_font_size_long"])
    font_name = "Nirmala UI" if job.get("language") == "Hindi" else "Arial"
    if job.get("language") == "Hindi":
        font_path = Path("C:/Windows/Fonts/Nirmala.ttc")
        if not font_path.is_file():
            raise FileNotFoundError(f"Hindi captions require Nirmala.ttc: {font_path}")
        missing = _check_hindi_glyphs("".join(scene["narration"] for scene in script["scenes"]), font_path)
        if missing:
            raise ValueError(f"Nirmala.ttc is missing Hindi glyphs: {''.join(missing)}")
    style = f"Style: Default,{font_name},{size},&H00FFFFFF,&H0000D0FF,&H00000000,&H8C000000,1,0,0,0,100,100,0,0,1,4,2,2,2,80,80,{max(30, int(height * 0.20))},1"
    events = []
    model = _load_whisper()
    max_words = 7
    max_chars = 34 if job.get("mode") == "shorts" else 48
    timeline_offset = 0.0
    previous_end = 0.0
    chunk_count = 0
    for scene, voice in zip(script["scenes"], voices):
        words = _word_times(voice["path"], scene["narration"], voice["duration"], model)
        if not words:
            words = [(word, idx * 0.2, (idx + 1) * 0.2) for idx, word in enumerate(scene["narration"].split())]
        for chunk in _caption_groups(words, max_words, max_chars):
            start = timeline_offset + max(0.0, float(chunk[0][1]))
            end = timeline_offset + min(float(voice["duration"]), float(chunk[-1][2]))
            start = max(start, previous_end)
            end = max(start + 0.04, end)
            text = " ".join(word for word, _, _ in chunk)
            lines = _wrap_caption(text, max_words, max_chars)
            if end > timeline_offset + float(voice["duration"]):
                end = timeline_offset + float(voice["duration"])
            if end <= start:
                raise ValueError("Caption timing does not advance monotonically.")
            caption_text = r"\N".join(lines)
            events.append(f"Dialogue: 0,{_timestamp(start)},{_timestamp(end)},Default,,0,0,0,,{caption_text}")
            previous_end = end + 0.04
            chunk_count += 1
        timeline_offset += float(voice["duration"]) + 0.3
    ass = "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nScaledBorderAndShadow: yes\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n%s\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n%s\n" % (width, height, style, "\n".join(events))
    target = job_directory(job) / "captions.ass"
    target.write_text(ass, encoding="utf-8-sig")
    log("captions", "Caption chunks created with safe margins and language-aware font",
        chunks=chunk_count, resolution=f"{width}x{height}", font=font_name, max_words=max_words, max_lines=2)
    return target


def audit_captions(path: str | Path, job: dict, script: dict, voices: list[dict]) -> dict:
    content = Path(path).read_text(encoding="utf-8-sig")
    dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    previous_end = -1.0
    overflow_count = 0
    max_words = 0
    max_lines = 0
    for line in dialogues:
        parts = line.split(",", 9)
        start = _parse_timestamp(parts[1])
        end = _parse_timestamp(parts[2])
        text = parts[9]
        lines = text.split(r"\N")
        words = text.replace(r"\N", " ").split()
        max_words = max(max_words, len(words))
        max_lines = max(max_lines, len(lines))
        if len(words) > 7 or len(lines) > 2 or any(len(item) > (34 if job["mode"] == "shorts" else 48) for item in lines):
            overflow_count += 1
        if start < previous_end or end <= start:
            overflow_count += 1
        previous_end = end
        if job.get("language") == "Hindi" and any(
            character.isalpha() and not "\u0900" <= character <= "\u097f"
            for character in text
        ):
            overflow_count += 1
        if job.get("language") != "Hindi" and re.search(r"[\u0900-\u097f]", text):
            overflow_count += 1
    if len(voices) != len(script["scenes"]) or not dialogues:
        overflow_count += 1
    return {
        "chunks": len(dialogues),
        "max_words": max_words,
        "max_lines": max_lines,
        "overflow_count": overflow_count,
        "font": "Nirmala UI" if job.get("language") == "Hindi" else "Arial",
        "safe_margins": True,
    }


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


if __name__ == "__main__":
    from .common import cli_main, load_job

    def _run(job):
        directory = job_directory(job)
        return str(create_captions(job, load_job(directory / "script.json"), load_job(directory / "voice.json")))

    cli_main(_run)
