from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

from .common import CONFIG, emit, ffmpeg_path, job_directory, log, save_json


def _voice_model(job: dict) -> Path:
    voice = job.get("voice", "female").lower()
    language = job.get("language", "English").lower()
    env_key = f"PIPER_MODEL_{'HI' if language.startswith('hindi') else 'EN'}_{voice.upper()}"
    if language.startswith("hindi"):
        default_name = f"hi_IN-{'priyamvada' if voice == 'female' else 'rohan'}-medium.onnx"
    else:
        default_name = f"en_US-{'amy' if voice == 'female' else 'joe'}-medium.onnx"
    configured = os.getenv(env_key)
    path = Path(configured) if configured else Path(CONFIG["paths"]["voices"]) / default_name
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _piper_executable() -> str:
    configured = os.getenv("PIPER_EXECUTABLE")
    executable = configured or shutil.which("piper")
    if not executable:
        local = Path(__file__).resolve().parents[1] / ".venv" / ("Scripts/piper.exe" if os.name == "nt" else "bin/piper")
        if local.is_file():
            executable = str(local)
    if not executable:
        raise RuntimeError("Local Piper TTS is not installed. Run scripts/setup.ps1 and download configured Piper voice models.")
    return executable


def _audio_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=False, timeout=15,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
        raise RuntimeError(f"ffprobe could not measure narration duration: {result.stderr}")
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _mean_volume(path: Path) -> float:
    result = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", os.devnull],
        capture_output=True, text=True, check=False, timeout=30,
    )
    match = re.search(r"mean_volume:\s*(-?inf|-?[\d.]+)\s*dB", result.stderr, re.IGNORECASE)
    if result.returncode or not match:
        raise RuntimeError(f"Could not measure audio level for {path.name}: {result.stderr.strip()}")
    return float("-inf") if "inf" in match.group(1).lower() else float(match.group(1))


def generate_voice(job: dict, script: dict) -> list[dict]:
    executable = _piper_executable()
    model = _voice_model(job)
    language = job.get("language", "English")
    if not model.is_file():
        raise FileNotFoundError(f"Local Piper voice model is missing: {model}")
    if language == "Hindi" and "hi_in" not in model.name.lower():
        raise ValueError(f"Language mismatch: job language is Hindi but selected Piper voice model is {model.name}.")
    if language != "Hindi" and "hi_in" in model.name.lower():
        raise ValueError(f"Language mismatch: job language is {language} but selected Piper voice model is {model.name}.")
    expected_name = {
        ("Hindi", "female"): "priyamvada",
        ("Hindi", "male"): "rohan",
        ("English", "female"): "amy",
        ("English", "male"): "joe",
    }.get((language, job.get("voice", "female")))
    if expected_name and expected_name not in model.name.lower():
        raise ValueError(
            f"Voice mismatch: job selected {language} {job.get('voice')} but configured Piper voice is {model.name}."
        )
    log("voice", "Selected Piper voice", language=language, voice=job.get("voice", "female"), voice_file=model.name)

    output = job_directory(job) / "voice"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, scene in enumerate(script["scenes"]):
        if language == "Hindi" and not re.search(r"[\u0900-\u097f]", scene["narration"]):
            raise ValueError(f"Scene {index + 1} narration is not Hindi even though the selected language is Hindi.")
        if language != "Hindi" and re.search(r"[\u0900-\u097f]", scene["narration"]):
            raise ValueError(f"Scene {index + 1} narration is Hindi but the selected language is {language}.")

        target = output / f"scene-{index + 1:03}.wav"
        if target.exists() and target.stat().st_size > 4096:
            duration = _audio_duration(target)
            mean_volume = _mean_volume(target)
            if duration <= 0.5 or mean_volume <= -45:
                raise ValueError(f"Scene {index + 1} cached voice is silent or too short ({duration:.2f}s, {mean_volume:.1f} dB).")
            results.append({"path": str(target), "duration": duration, "mean_volume_db": mean_volume})
            continue
        temporary = target.with_suffix(".partial.wav")
        command = [executable, "--model", str(model), "--output_file", str(temporary)]
        started = time.monotonic()
        result = subprocess.run(command, input=scene["narration"], capture_output=True, text=True, check=False, timeout=180)
        if result.returncode or not temporary.exists():
            temporary.unlink(missing_ok=True)
            log("voice", "Piper TTS failed", scene=index + 1, command=subprocess.list2cmdline(command), exit_code=result.returncode, stderr=result.stderr)
            raise RuntimeError(f"Piper TTS failed for scene {index + 1} (exit {result.returncode}): {result.stderr}")
        duration = _audio_duration(temporary)
        mean_volume = _mean_volume(temporary)
        if duration <= 0.5:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Scene {index + 1} voice audio is too short ({duration:.2f}s; minimum 0.5s).")
        if mean_volume <= -45:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Scene {index + 1} voice audio is silent ({mean_volume:.1f} dB mean volume).")
        temporary.replace(target)
        results.append({"path": str(target), "duration": duration, "mean_volume_db": mean_volume})
        log("voice", "Narration synthesized", scene=index + 1, provider="piper-local", language=language, voice_model=model.name, seconds=round(time.monotonic() - started, 2), audio_seconds=round(duration, 2))
        emit("stage", job_id=job["id"], stage="voice", status="running", progress=index + 1, total=len(script["scenes"]))
    metadata = {
        "language": language,
        "voice": job.get("voice", "female"),
        "voice_model": str(model),
        "voice_file": model.name,
        "voice_files": [str(item["path"]) for item in results],
    }
    save_json(job_directory(job) / "language_voice.json", metadata)
    return results


if __name__ == "__main__":
    from .common import cli_main

    def _run(job):
        from .common import load_job
        script_path = job_directory(job) / "script.json"
        return generate_voice(job, load_job(script_path))

    cli_main(_run)
