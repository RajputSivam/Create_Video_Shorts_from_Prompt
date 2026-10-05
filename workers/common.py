from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
JOB_STATE_LOCK = threading.RLock()


def _retry_permission_error(operation):
    delay = 0.2
    for attempt in range(10):
        try:
            return operation()
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(delay)
            delay += 0.2


def configured_path(name: str) -> Path:
    value = Path(os.getenv(f"{name.upper()}_DIR", CONFIG["paths"][name]))
    return value if value.is_absolute() else ROOT / value


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, "time": time.time(), **values}, ensure_ascii=False), flush=True)


def log(stage: str, message: str, **values: Any) -> None:
    emit("log", stage=stage, message=message, **values)


def load_job(path: str | Path) -> dict:
    target = Path(path)
    if target.name == "job.json":
        with JOB_STATE_LOCK:
            contents = _retry_permission_error(lambda: target.read_text(encoding="utf-8"))
    else:
        contents = target.read_text(encoding="utf-8")
    return json.loads(contents)


def save_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    if target.name == "job.json":
        serialized = json.dumps(data, ensure_ascii=False, indent=2)
        temporary = target.with_name(
            f"{target.name}.{os.getpid()}.{threading.get_ident()}.part"
        )
        with JOB_STATE_LOCK:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)

                def atomic_write() -> None:
                    temporary.write_text(serialized, encoding="utf-8")
                    temporary.replace(target)

                _retry_permission_error(atomic_write)
                return
            except Exception as error:
                log("job-state", "Atomic job-state save failed; falling back to direct write",
                    path=str(target), error=f"{type(error).__name__}: {error}")
                try:
                    target.write_text(serialized, encoding="utf-8")
                    return
                except Exception as fallback_error:
                    log("job-state", "Job-state save failed; continuing without a progress snapshot",
                        path=str(target), error=f"{type(fallback_error).__name__}: {fallback_error}")
                    return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def job_directory(job: dict) -> Path:
    return configured_path("jobs") / job["id"]


def output_directory(job: dict) -> Path:
    return configured_path("outputs") / job["id"]


def ffmpeg_path() -> str:
    import imageio_ffmpeg

    return os.getenv("FFMPEG_PATH") or imageio_ffmpeg.get_ffmpeg_exe()


def required_file(path: Path, message: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")
    return path


def cli_main(function) -> None:
    try:
        result = function(load_job(sys.argv[1]))
        emit("worker_complete", result=result)
    except Exception as error:
        emit("worker_error", error=f"{type(error).__name__}: {error}")
        raise
