import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock, Thread

from PIL import Image

from . import database
from .image_generation import ImageProviderChain, generate_image
from .renderer import probe_video, render_video
from .scene_splitter import split_scenes
from .script_generation import generate_script
from .voiceover import generate_voice

logger = logging.getLogger("uvicorn.error")
_assets_lock = Lock()


def _save_assets(job_id: str, **updates) -> None:
    with _assets_lock:
        current = database.get_job(job_id)
        assets = current["assets"] if current else {}
        assets.update(updates)
        database.update_job(job_id, assets=json.dumps(assets))


def _record_stage(job_id: str, name: str, status: str, detail: str | None = None) -> None:
    current = database.get_job(job_id)
    assets = current["assets"] if current else {}
    stage_results = assets.get("stage_results", {})
    stage_results[name] = {"status": status, "detail": detail}
    _save_assets(job_id, stage_results=stage_results)


def _start_heartbeat(job_id: str) -> tuple[Event, Thread]:
    stop = Event()

    def beat() -> None:
        while not stop.wait(20):
            try:
                database.heartbeat_job(job_id)
            except Exception:
                logger.exception("Heartbeat update failed for job=%s", job_id)

    thread = Thread(target=beat, name=f"heartbeat-{job_id[:8]}", daemon=True)
    thread.start()
    return stop, thread


def _finish_stage(timings: dict, stage: str, started: float) -> None:
    elapsed = time.monotonic() - started
    timings[stage] = round(elapsed, 2)
    logger.info("Job stage complete: %s took %.2fs", stage, elapsed)


def process_job(job_id: str) -> None:
    job = database.get_job(job_id)
    if not job:
        return
    root = database.DATA_DIR / job_id
    root.mkdir(parents=True, exist_ok=True)
    log_dir = database.ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_log_handler = logging.FileHandler(log_dir / f"job-{job_id}.log", encoding="utf-8")
    job_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    logger.addHandler(job_log_handler)
    previous_log_level = logger.level
    logger.setLevel(logging.INFO)
    logger.info("Starting job=%s log=%s", job_id, log_dir / f"job-{job_id}.log")
    heartbeat_stop, heartbeat_thread = _start_heartbeat(job_id)
    timings = {}
    job_started = time.monotonic()
    current_stage = "Starting render"
    try:
        database.update_job(job_id, status="running", stage="Writing your script", stage_status="running", progress=0, total=0, error=None, warning=None)
        current_stage = "Writing your script"
        stage_started = time.monotonic()
        script = generate_script(job, root / "script.txt", lambda stage, progress, total: database.update_job(job_id, stage=stage, progress=progress, total=total))
        _record_stage(job_id, "script", "completed")
        _finish_stage(timings, "script", stage_started)

        current_stage = "Planning your scenes"
        stage_started = time.monotonic()
        database.update_job(job_id, stage="Planning your scenes", progress=0, total=1)
        scenes = split_scenes(job, script, root / "scenes.json")
        database.update_job(job_id, stage="Planning your scenes", progress=1, total=1)
        _save_assets(job_id, script_ready=True, scene_count=len(scenes))
        _record_stage(job_id, "scene_planning", "completed")
        _finish_stage(timings, "scene_planning", stage_started)

        current_stage = "Generating images"
        stage_started = time.monotonic()
        database.update_job(job_id, stage="Generating images", progress=0, total=len(scenes))
        image_paths = [None] * len(scenes)
        image_results = [None] * len(scenes)
        image_chain = ImageProviderChain()

        def create_scene_image(index: int, scene: dict):
            image_path = root / "images" / f"scene-{index + 1:03}.jpg"
            return generate_image(job, scene, index, image_path, image_chain)

        completed_images = 0
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="scene-image") as pool:
            futures = {pool.submit(create_scene_image, index, scene): index for index, scene in enumerate(scenes)}
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                image_paths[index] = result.path
                image_results[index] = {
                "scene": index + 1,
                "provider": result.provider,
                "placeholder": result.placeholder,
                "failures": result.failures or [],
                "attribution": result.attribution,
                }
                completed_images += 1
                current_results = [item for item in image_results if item is not None]
                database.update_job(job_id, stage="Generating images", progress=completed_images, total=len(scenes))
                _save_assets(
                    job_id,
                    completed_scenes=completed_images,
                    scene_count=len(scenes),
                    image_results=current_results,
                    placeholder_scenes=[item["scene"] for item in current_results if item["placeholder"]],
                )
        image_failures = []
        for image_result in image_results:
            if image_result and image_result["placeholder"]:
                image_failures.extend(image_result["failures"])
        warning = "; ".join(dict.fromkeys(image_failures)) if image_failures else None
        if warning:
            database.update_job(job_id, warning=warning)
            _record_stage(job_id, "images", "failed", warning)
        else:
            _record_stage(job_id, "images", "completed")
        _finish_stage(timings, "images", stage_started)
        _save_assets(job_id, timing_summary=timings)

        current_stage = "Recording voiceover"
        stage_started = time.monotonic()
        database.update_job(job_id, stage="Recording voiceover", progress=0, total=len(scenes))
        voice_assets = [None] * len(scenes)

        def create_scene_voice(index: int, scene: dict):
            audio_path = root / "audio" / f"scene-{index + 1:03}.mp3"
            audio_duration = generate_voice(job, scene, index, audio_path)
            return {"image_path": image_paths[index], "audio_path": audio_path, "audio_duration": audio_duration}

        completed_voices = 0
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="scene-voice") as pool:
            futures = {pool.submit(create_scene_voice, index, scene): index for index, scene in enumerate(scenes)}
            for future in as_completed(futures):
                index = futures[future]
                voice_assets[index] = future.result()
                completed_voices += 1
                database.update_job(job_id, stage="Recording voiceover", progress=completed_voices, total=len(scenes))
        assets = voice_assets
        _record_stage(job_id, "voice", "completed")
        _finish_stage(timings, "voice", stage_started)
        _save_assets(job_id, timing_summary=timings)

        output = root / ("output-preview.mp4" if job.get("fast_preview") else "output.mp4")
        if not output.exists():
            current_stage = "Rendering final video"
            database.update_job(job_id, stage="Rendering scene clips", progress=0, total=len(scenes))
            render_timings = render_video(
                job,
                scenes,
                assets,
                output,
                lambda stage, done, total: database.update_job(job_id, stage=stage, progress=done, total=total),
            )
            timings.update(render_timings)
        current_stage = "Validating rendered video"
        database.update_job(job_id, stage=current_stage, progress=1, total=1)
        probe = probe_video(output)
        streams = probe.get("streams", [])
        if not any(stream.get("codec_type") == "audio" for stream in streams):
            raise RuntimeError("Rendered video validation failed: output contains no audio stream.")
        real_images = 0
        for image_result, image_path in zip(image_results, image_paths):
            if image_result and not image_result["placeholder"] and image_path:
                with Image.open(image_path) as image:
                    image.verify()
                real_images += 1
        if real_images < 1:
            raise RuntimeError("Rendered video validation failed: fewer than one real generated image was available.")
        _save_assets(
            job_id,
            ffprobe_summary={
                "video_streams": sum(stream.get("codec_type") == "video" for stream in streams),
                "audio_streams": sum(stream.get("codec_type") == "audio" for stream in streams),
                "real_images": real_images,
                "duration": probe.get("format", {}).get("duration"),
            },
        )
        _record_stage(job_id, "validation", "completed")
        timings["total"] = round(time.monotonic() - job_started, 2)
        _save_assets(job_id, timing_summary=timings)
        logger.info("Job timing summary job=%s: %s", job_id, json.dumps(timings, sort_keys=True))
        database.update_job(job_id, status="completed", stage="Ready to watch", stage_status="completed", progress=len(scenes), total=len(scenes), output_file=str(output), error=None)
    except Exception as error:
        logger.exception("Job stage failed job=%s stage=%s", job_id, current_stage)
        timings["total"] = round(time.monotonic() - job_started, 2)
        _save_assets(job_id, timing_summary=timings)
        logger.error("Job timing summary job=%s: %s", job_id, json.dumps(timings, sort_keys=True))
        _record_stage(job_id, current_stage, "failed", str(error))
        database.update_job(job_id, status="failed", stage=f"{current_stage} FAILED", stage_status="failed", error=str(error))
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        logger.removeHandler(job_log_handler)
        job_log_handler.close()
        logger.setLevel(previous_log_level)