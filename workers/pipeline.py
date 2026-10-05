from __future__ import annotations

import shutil
import sys
import time
import traceback

from .captions import create_captions
from .common import emit, job_directory, load_job, log, output_directory, save_json
from .effects import prepare_effects
from .images import generate_images
from .music import create_music
from .qa import check, save_qa_report
from .render import render
from .script_llm import generate_script
from .sfx import generate_sfx
from .tts import generate_voice
from .upscale import upscale_images
from .doctor import preflight_job

_STAGE_STARTED: dict[str, float] = {}


def _set_stage(job: dict, name: str, status: str, progress: int = 0, total: int = 1, detail: str | None = None) -> None:
    path = job_directory(job) / "job.json"
    current = load_job(path)
    stages = current.setdefault("stage_results", {})
    stages[name] = {"status": status, "detail": detail}
    current.update({
        "stage": name,
        "stage_status": status,
        "progress": progress,
        "total": total,
        "updated_at": time.time(),
    })
    save_json(path, current)
    if status == "running":
        _STAGE_STARTED[name] = time.monotonic()
    elif status in ("completed", "degraded") and name in _STAGE_STARTED:
        elapsed = round(time.monotonic() - _STAGE_STARTED.pop(name), 2)
        log(name, "Stage completed", seconds=elapsed)
    emit("stage", job_id=job["id"], stage=name, status=status, progress=progress, total=total, detail=detail)


def process_job(job: dict) -> dict:
    directory = job_directory(job)
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    current = load_job(job_path) if job_path.exists() else dict(job)
    current.update({"status": "running", "error": None, "updated_at": time.time()})
    save_json(job_path, current)
    started = time.monotonic()
    qa_checks = {}
    qa_metrics = {}
    try:
        save_json(directory / "language_voice.json", {
            "language": job.get("language"),
            "voice": job.get("voice"),
            "voice_file": None,
        })
        _set_stage(job, "preflight", "running")
        preflight = {"checks": job["preflight_checks"]} if job.get("preflight_checks") else preflight_job(job)
        voice_check = preflight["checks"].get("voice", {})
        language_voice = load_job(directory / "language_voice.json")
        language_voice["voice_file"] = voice_check.get("detail")
        save_json(directory / "language_voice.json", language_voice)
        failed_preflight = {
            name: item["detail"] for name, item in preflight["checks"].items()
            if item["status"] != "passed"
        }
        qa_checks["preflight"] = check(
            "PASS" if not failed_preflight else "FAIL",
            "All offline job prerequisites are available." if not failed_preflight else "; ".join(
                f"{name}: {reason}" for name, reason in failed_preflight.items()
            ),
        )
        qa_metrics["preflight"] = preflight["checks"]
        save_qa_report(job, qa_checks, qa_metrics)
        _set_stage(job, "preflight", "completed" if not failed_preflight else "failed",
                   1, 1, qa_checks["preflight"]["reason"])
        if failed_preflight:
            raise RuntimeError(f"Preflight failed: {qa_checks['preflight']['reason']}")

        _set_stage(job, "script", "running")
        script_path = directory / "script.json"
        if script_path.exists():
            script = load_job(script_path)
            from .script_llm import expected_scene_count, validate_script

            if script.get("provider") != "ollama":
                raise RuntimeError("Script generation failed: saved script is not verified as Ollama-generated.")
            script = validate_script(script, job, expected_scene_count(job), allow_enriched=True)
        else:
            script = generate_script(job)
        from .script_llm import verify_ollama_unloaded

        verify_ollama_unloaded(required=script.get("provider") == "ollama")
        total_script_words = sum(len(scene["narration"].split()) for scene in script["scenes"])
        devanagari_letters = sum(
            1 for scene in script["scenes"] for character in scene["narration"]
            if "\u0900" <= character <= "\u097f" and character.isalpha()
        )
        narration_letters = sum(
            1 for scene in script["scenes"] for character in scene["narration"] if character.isalpha()
        )
        language_ratio = devanagari_letters / max(1, narration_letters) if job["language"] == "Hindi" else 1.0
        qa_checks["script"] = check(
            "PASS",
            "Narration language, scene count, placeholder, and image prompt validation passed.",
            scenes=len(script["scenes"]), narration_words=total_script_words,
            language_ratio=round(language_ratio, 3),
        )
        qa_metrics.update({"scenes": len(script["scenes"]), "language_ratio": round(language_ratio, 3)})
        save_qa_report(job, qa_checks, qa_metrics)
        _set_stage(job, "script", "completed", 1, 1, script.get("warning"))

        _set_stage(job, "voice", "running")
        voices = generate_voice(job, script)
        save_json(directory / "voice.json", voices)
        voice_duration = sum(float(item["duration"]) for item in voices)
        voice_levels = [float(item["mean_volume_db"]) for item in voices]
        target_duration = float(job["duration"])
        duration_match = target_duration * 0.75 <= voice_duration <= target_duration * 1.25
        qa_checks["voice"] = check(
            "PASS" if duration_match else "WARN",
            "Voice files are present, audible, and match the requested duration." if duration_match
            else f"Total narration is {voice_duration:.1f}s versus {target_duration:.1f}s target (outside ±25%).",
            scenes=len(voices), duration_seconds=round(voice_duration, 2),
            min_scene_seconds=round(min(float(item["duration"]) for item in voices), 2),
            min_mean_volume_db=round(min(voice_levels), 2),
        )
        qa_metrics.update({"voice_duration_seconds": round(voice_duration, 2), "audio_level_db": round(min(voice_levels), 2)})
        save_qa_report(job, qa_checks, qa_metrics)
        _set_stage(job, "voice", "completed", len(voices), len(voices))

        _set_stage(job, "images", "running")
        images = generate_images(job, script)
        if len(images) != len(script["scenes"]) or any(item.get("placeholder") for item in images):
            raise RuntimeError(f"Image QA failed: expected {len(script['scenes'])} real images, received {len(images)}.")
        real = len(images)
        warnings = []
        clip_scores = [float(item["clip_similarity"]) for item in images if item.get("clip_similarity") is not None]
        qa_checks["images"] = check(
            "PASS",
            "All scene images passed blank-image and perceptual-duplicate checks."
            + (" CLIP prompt matching passed." if clip_scores else " Optional CLIP matching skipped because no CLIP weights are cached locally."),
            image_count=len(images), duplicate_images=0, clip_scores=clip_scores,
        )
        qa_metrics.update({"image_count": len(images), "duplicate_images": 0, "clip_similarity_scores": clip_scores or None})
        save_qa_report(job, qa_checks, qa_metrics)
        preview_root = output_directory(job) / "scenes"
        preview_root.mkdir(parents=True, exist_ok=True)
        for item in images:
            preview = preview_root / f"scene-{item['scene']:03}.jpg"
            shutil.copyfile(item["path"], preview)
            item["preview_url"] = f"/outputs/{job['id']}/scenes/{preview.name}"
        save_json(directory / "images.json", images)
        current = load_job(job_path)
        assets = current.setdefault("assets", {})
        assets.update({"script": script, "images": images})
        current["warning"] = "; ".join([*warnings, *([script["warning"]] if script.get("warning") else [])]) or None
        save_json(job_path, current)
        _set_stage(job, "images", "completed" if real else "degraded", len(images), len(images), "; ".join(warnings) or None)

        _set_stage(job, "upscale", "running")
        images = upscale_images(job, images)
        save_json(directory / "images.json", images)
        _set_stage(job, "upscale", "completed", len(images), len(images))

        _set_stage(job, "captions", "running")
        captions = create_captions(job, script, voices)
        save_json(directory / "captions.json", {"path": str(captions)})
        from .captions import audit_captions

        caption_audit = audit_captions(captions, job, script, voices)
        qa_checks["captions"] = check(
            "PASS" if caption_audit["overflow_count"] == 0 else "FAIL",
            "Caption chunks meet word, line, timing, font, and margin constraints."
            if caption_audit["overflow_count"] == 0 else "One or more caption chunks overflow or overlap.",
            **caption_audit,
        )
        qa_metrics.update({"caption_overflow": caption_audit["overflow_count"], "caption_chunks": caption_audit["chunks"]})
        save_qa_report(job, qa_checks, qa_metrics)
        _set_stage(job, "captions", "completed", 1, 1)

        total_duration = sum(item["duration"] for item in voices)
        _set_stage(job, "music", "running")
        music = create_music(job, script, total_duration)
        save_json(directory / "music.json", {"path": str(music)})
        _set_stage(job, "music", "completed", 1, 1)

        _set_stage(job, "sfx", "running")
        effects_audio = generate_sfx(job)
        save_json(directory / "sfx.json", effects_audio)
        _set_stage(job, "sfx", "completed", 1, 1)

        _set_stage(job, "effects", "running")
        effects = prepare_effects(job)
        _set_stage(job, "effects", "completed", 1, 1)

        _set_stage(job, "render", "running")
        result = render(job, script, images, voices, effects_audio, music, captions, effects)
        qa_checks.update(result.pop("qa_checks", {}))
        qa_metrics.update(result.pop("qa_metrics", {}))
        report = save_qa_report(job, qa_checks, qa_metrics)
        _set_stage(job, "render", "completed", 1, 1)
        current = load_job(job_path)
        final_status = "completed" if report["status"] == "pass" else "needs_review"
        current.update({
            "status": final_status,
            "stage": "Ready to watch" if final_status == "completed" else "Needs review",
            "stage_status": "completed" if final_status == "completed" else "needs_review",
            "progress": 1,
            "total": 1,
            "output_file": result["video"],
            "video_url": f"/outputs/{job['id']}/video.mp4",
            "thumbnail_url": f"/outputs/{job['id']}/thumbnail.jpg",
            "metadata_url": f"/outputs/{job['id']}/metadata.json",
            "assets": {"script": script, "images": images, "voices": voices, "probe": result},
            "warning": "; ".join([*warnings, *([script["warning"]] if script.get("warning") else [])]) or None,
            "qa_report": report,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "updated_at": time.time(),
        })
        save_json(job_path, current)
        emit("completed", job_id=job["id"], result=current)
        return current
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
        current = load_job(job_path)
        stage = current.get("stage", "pipeline")
        qa_key = "final_video" if stage == "render" else stage
        qa_checks[qa_key] = check("FAIL", detail)
        report = save_qa_report(job, qa_checks, qa_metrics)
        if "no real generated images" in str(error).lower():
            stage = "images"
        status = "needs_review" if stage == "render" else "failed"
        stages = current.setdefault("stage_results", {})
        stages[stage] = {"status": status, "detail": detail}
        current.update({
            "status": status,
            "stage": stage,
            "stage_status": status,
            "error": detail,
            "qa_report": report,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "updated_at": time.time(),
        })
        save_json(job_path, current)
        emit("needs_review" if status == "needs_review" else "failed",
             job_id=job["id"], stage=stage, error=detail, qa_report=report)
        raise


if __name__ == "__main__":
    process_job(load_job(sys.argv[1]))
