from __future__ import annotations

from faster_whisper import WhisperModel

from .common import CONFIG, configured_path, log


def download_models() -> None:
    model_name = CONFIG["models"]["whisper"]
    target = configured_path("cache") / "whisper" / model_name
    target.parent.mkdir(parents=True, exist_ok=True)
    WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(target.parent))
    log("setup", "Cached faster-whisper CPU model locally", model=model_name, path=str(target))


if __name__ == "__main__":
    download_models()
