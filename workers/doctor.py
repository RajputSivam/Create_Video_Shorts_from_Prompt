from __future__ import annotations

import argparse
import base64
import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

from .common import CONFIG, ROOT, configured_path


def _check(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def _ffprobe_path() -> str | None:
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    candidate = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(candidate.glob("Gyan.FFmpeg.Shared*/*/bin/ffprobe.exe"))
    return str(matches[-1]) if matches else None


def _hf_cache_root() -> Path:
    cache = os.getenv("HF_HUB_CACHE")
    if cache:
        return Path(cache)
    home = os.getenv("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_snapshot(model_id: str, required_file: str) -> Path | None:
    roots = [configured_path("cache") / "huggingface", _hf_cache_root()]
    for root in dict.fromkeys(roots):
        repository = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
        snapshots = sorted(repository.glob("*")) if repository.is_dir() else []
        snapshot = next((item for item in reversed(snapshots) if (item / required_file).is_file()), None)
        if snapshot:
            return snapshot
    return None


def _available_memory_bytes() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Windows could not report available memory.")
        return int(status.avail_phys)
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemAvailable", "MemFree", "Buffers", "Cached"}:
                values[key] = int(value.strip().split()[0]) * 1024
        if "MemAvailable" in values:
            return values["MemAvailable"]
        return sum(values.get(key, 0) for key in ("MemFree", "Buffers", "Cached"))
    raise OSError("Available memory measurement is unsupported on this platform.")


def _job_voice_path(job: dict) -> Path:
    language = job.get("language")
    voice = job.get("voice")
    voice_key = f"PIPER_MODEL_{'HI' if language == 'Hindi' else 'EN'}_{str(voice).upper()}"
    default_name = {
        ("Hindi", "female"): "hi_IN-priyamvada-medium.onnx",
        ("Hindi", "male"): "hi_IN-rohan-medium.onnx",
        ("English", "female"): "en_US-amy-medium.onnx",
        ("English", "male"): "en_US-joe-medium.onnx",
    }.get((language, voice))
    if not default_name:
        raise ValueError("Choose Hindi or English and a male or female voice.")
    configured = os.getenv(voice_key)
    path = Path(configured) if configured else configured_path("voices") / default_name
    return path if path.is_absolute() else ROOT / path


def preflight_job(job: dict) -> dict:
    checks: dict[str, dict[str, str]] = {}

    def add(name: str, operation) -> None:
        try:
            checks[name] = _check("passed", operation())
        except Exception as error:
            checks[name] = _check("failed", f"{type(error).__name__}: {error}")

    def valid_config() -> str:
        mode = job.get("mode")
        duration = job.get("duration")
        if mode not in {"shorts", "video"}:
            raise ValueError("Mode must be shorts or video.")
        min_duration, max_duration = (15, 60) if mode == "shorts" else (60, 900)
        if not isinstance(duration, int) or not min_duration <= duration <= max_duration:
            raise ValueError(f"Duration must be {min_duration}-{max_duration} seconds for {mode}.")
        if job.get("language") not in {"English", "Hindi"}:
            raise ValueError("Language must be English or Hindi.")
        if job.get("voice") not in {"male", "female"}:
            raise ValueError("Voice must be male or female.")
        scenes = job.get("scene_count")
        if scenes is not None and (not isinstance(scenes, int) or not 2 <= scenes <= 30):
            raise ValueError("Scene count must be an integer from 2 to 30.")
        return "Job duration, scenes, language, and voice values are valid."

    add("config", valid_config)

    def check_ollama() -> str:
        endpoint = os.getenv("OLLAMA_URL", CONFIG["models"]["ollama_url"]).rstrip("/")
        model = os.getenv("OLLAMA_MODEL", CONFIG["models"]["llm"])
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=5) as response:
            names = [item.get("name", "") for item in json.loads(response.read()).get("models", [])]
        if not any(name == model or name.startswith(model + ":") for name in names):
            raise RuntimeError(f"Ollama is reachable but configured model '{model}' is not installed.")
        return f"Ollama reachable; configured model {model} is installed."

    add("ollama", check_ollama)

    def check_voice() -> str:
        path = _job_voice_path(job)
        metadata = Path(str(path) + ".json")
        expected_name = {
            ("Hindi", "female"): "priyamvada",
            ("Hindi", "male"): "rohan",
            ("English", "female"): "amy",
            ("English", "male"): "joe",
        }.get((job.get("language"), job.get("voice")))
        if expected_name and expected_name not in path.name.lower():
            raise ValueError(
                f"Selected {job['language']} {job['voice']} voice does not match configured Piper model {path.name}."
            )
        if not path.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Selected Piper voice requires both files: {path} and {metadata}.")
        piper = os.getenv("PIPER_EXECUTABLE") or shutil.which("piper")
        if not piper:
            candidate = ROOT / ".venv" / ("Scripts/piper.exe" if os.name == "nt" else "bin/piper")
            if candidate.is_file():
                piper = str(candidate)
        if not piper:
            raise FileNotFoundError("Piper executable is missing.")
        if job["language"] == "Hindi" and not Path("C:/Windows/Fonts/Nirmala.ttc").is_file():
            raise FileNotFoundError("Hindi caption font Nirmala.ttc is missing from C:/Windows/Fonts.")
        return f"{job['language']} {job['voice']} voice: {path.name} and {metadata.name}."

    add("voice", check_voice)

    def check_media_tools() -> str:
        import imageio_ffmpeg

        ffmpeg = os.getenv("FFMPEG_PATH") or imageio_ffmpeg.get_ffmpeg_exe()
        if subprocess.run([ffmpeg, "-version"], capture_output=True, check=False, timeout=15).returncode:
            raise RuntimeError(f"ffmpeg did not run: {ffmpeg}")
        ffprobe = os.getenv("FFPROBE_PATH") or _ffprobe_path()
        if not ffprobe or subprocess.run([ffprobe, "-version"], capture_output=True, check=False, timeout=15).returncode:
            raise RuntimeError("ffprobe is missing or could not run.")
        return "ffmpeg and ffprobe are available."

    add("media_tools", check_media_tools)

    def check_models() -> str:
        if importlib.util.find_spec("diffusers") is None:
            raise RuntimeError("The diffusers package is not installed.")
        if importlib.util.find_spec("faster_whisper") is None:
            raise RuntimeError("The faster-whisper package is not installed.")
        model_id = os.getenv("SDXL_MODEL", CONFIG["models"]["image"])
        if not _cached_snapshot(model_id, "model_index.json"):
            raise FileNotFoundError(f"SDXL model {model_id} is not cached locally; downloads are disabled.")
        whisper_name = os.getenv("WHISPER_MODEL", CONFIG["models"]["whisper"])
        whisper_dir = configured_path("cache") / "whisper" / whisper_name
        if not whisper_dir.is_dir():
            whisper_dir = _cached_snapshot(f"Systran/faster-whisper-{whisper_name}", "model.bin")
        if whisper_dir is None or not whisper_dir.exists():
            raise FileNotFoundError(f"Whisper model {whisper_name} is not cached locally; downloads are disabled.")
        return f"SDXL {model_id} and Whisper {whisper_name} are cached locally."

    add("models", check_models)

    def check_resources() -> str:
        disk = shutil.disk_usage(ROOT).free
        memory = _available_memory_bytes()
        if disk < 5 * 1024**3:
            raise RuntimeError(f"Only {disk / 1024**3:.1f} GiB disk space is free; at least 5 GiB is required.")
        if memory < 6 * 1024**3:
            raise RuntimeError(f"Only {memory / 1024**3:.1f} GiB RAM is free; at least 6 GiB is required.")
        return f"{disk / 1024**3:.1f} GiB free disk; {memory / 1024**3:.1f} GiB free RAM."

    add("resources", check_resources)
    return {
        "status": "passed" if all(check["status"] == "passed" for check in checks.values()) else "failed",
        "checks": checks,
    }


def run_doctor() -> dict:
    checks = {}
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            checks["gpu"] = _check("passed", f"{gpu}; {vram} GiB VRAM; PyTorch CUDA {torch.version.cuda}.")
        else:
            checks["gpu"] = _check("failed", "PyTorch is installed, but CUDA is unavailable.")
    except ImportError:
        checks["gpu"] = _check("missing", "PyTorch is not installed. Run scripts/setup.ps1 -InstallML.")

    try:
        import imageio_ffmpeg

        ffmpeg = os.getenv("FFMPEG_PATH") or imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=False, timeout=15)
        checks["ffmpeg"] = _check("passed" if result.returncode == 0 else "failed", (result.stdout or result.stderr).splitlines()[0])
    except Exception as error:
        checks["ffmpeg"] = _check("failed", str(error))
    ffprobe = _ffprobe_path()
    checks["ffprobe"] = _check("passed", ffprobe) if ffprobe else _check("missing", "Install FFmpeg and put ffprobe on PATH.")

    ollama = os.getenv("OLLAMA_URL", CONFIG["models"]["ollama_url"]).rstrip("/")
    try:
        with urllib.request.urlopen(f"{ollama}/api/tags", timeout=5) as response:
            tags = json.loads(response.read()).get("models", [])
        names = [item["name"] for item in tags]
        wanted = os.getenv("OLLAMA_MODEL", CONFIG["models"]["llm"])
        found = any(name == wanted or name.startswith(wanted + ":") for name in names)
        checks["ollama"] = _check("passed" if found else "missing", f"{wanted} available." if found else f"Ollama reachable; model {wanted} not pulled.")
    except Exception as error:
        checks["ollama"] = _check("missing", f"Ollama not reachable at {ollama}: {error}")

    for module, name in (("diffusers", "sdxl"), ("faster_whisper", "whisper"), ("piper", "piper")):
        installed = importlib.util.find_spec(module) is not None
        checks[name] = _check("passed" if installed else "missing", "Python package installed." if installed else f"Python package '{module}' is not installed.")
    whisper_name = os.getenv("WHISPER_MODEL", CONFIG["models"]["whisper"])
    whisper_model = Path(CONFIG["paths"]["cache"]) / "whisper" / whisper_name
    if not whisper_model.is_absolute():
        whisper_model = ROOT / whisper_model
    if not whisper_model.is_dir():
        whisper_model = _cached_snapshot(f"Systran/faster-whisper-{whisper_name}", "model.bin") or whisper_model
    if checks["whisper"]["status"] == "passed" and not whisper_model.is_dir():
        checks["whisper"] = _check("missing", f"Local Whisper weights are not cached at {whisper_model}. Run scripts/setup.ps1 -Models.")
    model_id = os.getenv("SDXL_MODEL", CONFIG["models"]["image"])
    model_directory = _cached_snapshot(model_id, "model_index.json")
    if checks["sdxl"]["status"] == "passed" and model_directory is None:
        checks["sdxl"] = _check("missing", f"SDXL weights are not cached yet ({model_id}); the first image job downloads them.")
    piper_executable = os.getenv("PIPER_EXECUTABLE") or shutil.which("piper")
    if not piper_executable:
        local_piper = ROOT / ".venv" / ("Scripts/piper.exe" if os.name == "nt" else "bin/piper")
        if local_piper.is_file():
            piper_executable = str(local_piper)
    if checks["piper"]["status"] == "passed" and not piper_executable:
        checks["piper"] = _check("missing", "Piper is installed but its executable was not found.")
    voice_defaults = {
        "PIPER_MODEL_EN_FEMALE": "en_US-amy-medium.onnx",
        "PIPER_MODEL_EN_MALE": "en_US-joe-medium.onnx",
        "PIPER_MODEL_HI_FEMALE": "hi_IN-priyamvada-medium.onnx",
        "PIPER_MODEL_HI_MALE": "hi_IN-rohan-medium.onnx",
    }
    voice_models = [
        Path(os.getenv(key, str(Path(CONFIG["paths"]["voices"]) / filename)))
        for key, filename in voice_defaults.items()
    ]
    voice_models = [path if path.is_absolute() else ROOT / path for path in voice_models]
    voice_missing = [str(path) for path in voice_models if not path.is_file()]
    checks["voices"] = _check("passed", "English and Hindi male/female Piper models installed.") if not voice_missing else _check("missing", f"Missing voice model(s): {', '.join(voice_missing)}")
    if piper_executable and not voice_missing:
        try:
            with tempfile.TemporaryDirectory(prefix="prompttovideo-doctor-") as folder:
                sample = Path(folder) / "piper-test.wav"
                tested = subprocess.run(
                    [piper_executable, "--model", str(voice_models[0]), "--output_file", str(sample)],
                    input="Local voice test.",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                if tested.returncode or not sample.is_file():
                    raise RuntimeError(f"exit {tested.returncode}: {tested.stderr}")
                with wave.open(str(sample), "rb") as audio:
                    audio_seconds = audio.getnframes() / audio.getframerate()
                if audio_seconds <= 0:
                    raise RuntimeError("Piper generated an empty audio file.")
                checks["piper"] = _check("passed", f"Generated local speech test ({audio_seconds:.1f}s).")
        except Exception as error:
            checks["piper"] = _check("failed", f"Piper speech self-test failed: {error}")

    custom_font = Path(os.getenv("FONT_PATH", CONFIG["paths"]["fonts"] + "/DejaVuSans-Bold.ttf"))
    english_fonts = [custom_font, Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
    hindi_fonts = [Path("C:/Windows/Fonts/Nirmala.ttc"), Path("C:/Windows/Fonts/Nirmala.ttf"), Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf")]
    english_font = next((path if path.is_absolute() else ROOT / path for path in english_fonts if (path if path.is_absolute() else ROOT / path).is_file()), None)
    hindi_font = next((path for path in hindi_fonts if path.is_file()), None)
    checks["fonts"] = _check("passed", f"English: {english_font}; Hindi: {hindi_font}") if english_font and hindi_font else _check("missing", "Install Arial/DejaVu Sans and Nirmala UI/Noto Sans Devanagari fonts for English/Hindi captions.")
    ok = all(check["status"] == "passed" for check in checks.values())
    return {"status": "ready" if ok else "setup_needed", "checks": checks, "python": sys.version.split()[0], "local_only": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--preflight-json")
    args = parser.parse_args()
    if args.preflight_json:
        job = json.loads(base64.urlsafe_b64decode(args.preflight_json.encode()).decode("utf-8"))
        report = preflight_job(job)
    else:
        report = run_doctor()
    print(json.dumps(report, ensure_ascii=False, indent=2 if not args.json else None))
    expected = "passed" if args.preflight_json else "ready"
    if report["status"] != expected:
        sys.exit(1)
