import hashlib
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip

from .config import get_mode_config
from .subtitles import make_caption_frames

logger = logging.getLogger("uvicorn.error")


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffprobe() -> str:
    import shutil

    executable = shutil.which("ffprobe")
    if executable:
        return executable
    winget_link = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe"
    if winget_link.is_file():
        return str(winget_link)
    package_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(package_root.glob("Gyan.FFmpeg.Shared*/*/bin/ffprobe.exe"))
    if candidates:
        return str(candidates[-1])
    raise RuntimeError("ffprobe was not found on PATH. Install FFmpeg with ffprobe and restart the app.")


def _run_ffmpeg(command: list[str], label: str) -> subprocess.CompletedProcess:
    logger.info("FFmpeg %s command: %s", label, subprocess.list2cmdline(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        logger.error("FFmpeg %s failed with exit code %s; stderr:\n%s", label, result.returncode, result.stderr)
    return result


def probe_video(path: Path) -> dict:
    command = [
        _ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]
    logger.info("ffprobe command: %s", subprocess.list2cmdline(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        logger.error("ffprobe failed with exit code %s; stderr:\n%s", result.returncode, result.stderr)
        raise RuntimeError(f"ffprobe failed with exit code {result.returncode}: {result.stderr}")
    try:
        details = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        logger.error("ffprobe returned invalid JSON; stdout:\n%s\nstderr:\n%s", result.stdout, result.stderr)
        raise RuntimeError("ffprobe returned invalid JSON.") from error
    logger.info("ffprobe output for %s:\n%s", path, json.dumps(details, indent=2, sort_keys=True))
    return details


@lru_cache(maxsize=1)
def _video_encoder() -> tuple[str, str, list[str]]:
    ffmpeg = _ffmpeg()
    listing_result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if listing_result.returncode:
        logger.error("FFmpeg encoder listing failed with exit code %s; stderr:\n%s", listing_result.returncode, listing_result.stderr)
        raise RuntimeError(f"FFmpeg encoder listing failed: {listing_result.stderr}")
    listing = listing_result.stdout
    for encoder, preset, options in (
        ("h264_nvenc", "p4", ["-cq", "23"]),
        ("h264_qsv", "veryfast", ["-global_quality", "23"]),
        ("h264_amf", "speed", ["-qp_i", "23", "-qp_p", "23"]),
    ):
        if encoder not in listing:
            continue
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1", "-frames:v", "1", "-c:v", encoder, "-f", "null", "-"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return encoder, preset, options
        logger.warning("FFmpeg hardware encoder %s failed; exit code=%s stderr:\n%s", encoder, probe.returncode, probe.stderr.decode(errors="replace"))
    logger.warning("Using FFmpeg software encoder libx264 after hardware encoder checks.")
    return "libx264", "veryfast", ["-crf", "23"]


def _file_digest(*paths: Path, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _render_scene_clip(job: dict, scene: dict, asset: dict, index: int, clip_path: Path, width: int, height: int) -> Path:
    digest = _file_digest(
        Path(asset["image_path"]),
        Path(asset["audio_path"]),
        extra=f"{scene['narration_text']}|{width}x{height}|{job['mode']}|{job.get('fast_preview', False)}|24",
    )
    cached_clip = clip_path.parent / f"scene-{index + 1:03}-{digest[:16]}.mp4"
    if cached_clip.exists() and cached_clip.stat().st_size > 4096:
        return cached_clip

    codec, preset, encoder_options = _video_encoder()
    audio = AudioFileClip(str(asset["audio_path"]))
    scene_clips = []
    overlay_clips = []
    try:
        speed = max(1.0, asset["audio_duration"] / 59.5) if job["mode"] == "shorts" else 1.0
        if speed > 1:
            audio = audio.with_speed_scaled(speed)
        duration = audio.duration
        image = ImageClip(str(asset["image_path"])).with_duration(duration)
        if job.get("fast_preview"):
            moving = image.resized(new_size=(width, height)).with_position((0, 0))
        else:
            zoom_width = int(width * 1.04)
            zoom_height = int(height * 1.04)
            moving = image.resized(new_size=(zoom_width, zoom_height))
            pan_x = zoom_width - width
            pan_y = zoom_height - height
            if index % 2:
                moving = moving.with_position(lambda t: (int(-pan_x * min(t / max(duration, 0.1), 1)), 0))
            else:
                moving = moving.with_position(lambda t: (0, int(-pan_y * min(t / max(duration, 0.1), 1))))
        scene_clips.append(image)
        layers = [moving]
        for frame, start, frame_duration, x, y in make_caption_frames(
            scene["narration_text"], width, height, get_mode_config(job["mode"]).caption_style, duration
        ):
            overlay = ImageClip(np.asarray(frame)).with_start(start).with_duration(frame_duration).with_position((x, y))
            overlay_clips.append(overlay)
            layers.append(overlay)
        video = CompositeVideoClip(layers, size=(width, height)).with_duration(duration).with_audio(audio)
        scene_clips.append(video)
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached_clip.with_name(f"{cached_clip.stem}.partial.mp4")
        video.write_videofile(
            str(temporary),
            fps=24,
            codec=codec,
            audio_codec="aac",
            audio_bitrate="128k",
            preset=preset,
            threads=1,
            ffmpeg_params=[*encoder_options, "-pix_fmt", "yuv420p"],
            logger=None,
        )
        temporary.replace(cached_clip)
        return cached_clip
    except Exception:
        logger.exception(
            "MoviePy FFmpeg scene render failed for scene %d (input_image=%s input_audio=%s output=%s codec=%s fps=24)",
            index + 1,
            asset["image_path"],
            asset["audio_path"],
            temporary,
            codec,
        )
        raise
    finally:
        for clip in scene_clips:
            clip.close()
        for clip in overlay_clips:
            clip.close()
        audio.close()


def _concat_scene_clips(clips: list[Path], destination: Path) -> None:
    manifest = destination.with_name(f"{destination.stem}.concat.txt")
    concat_lines = [f"file '{path.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in clips]
    manifest.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    temporary = destination.with_name(f"{destination.stem}.concat.partial{destination.suffix}")
    command = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-c", "copy", "-movflags", "+faststart", str(temporary),
    ]
    result = _run_ffmpeg(command, "scene concat")
    manifest.unlink(missing_ok=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[-1200:]}")
    temporary.replace(destination)


def _mux_narration(video_path: Path, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.stem}.audio.partial{destination.suffix}")
    command = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-i", str(video_path),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest",
        "-movflags", "+faststart", str(temporary),
    ]
    result = _run_ffmpeg(command, "narration mux")
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg narration mux failed with exit code {result.returncode}: {result.stderr}")
    temporary.replace(destination)


def _mix_music(video_path: Path, music_path: Path, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.stem}.music.partial{destination.suffix}")
    command = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex", "[1:a]volume=0.08[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(temporary),
    ]
    result = _run_ffmpeg(command, "music mix")
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg music mix failed: {result.stderr[-1200:]}")
    temporary.replace(destination)


def render_video(job: dict, scenes: list[dict], assets: list[dict], destination: Path, progress=None) -> dict[str, float]:
    config = get_mode_config(job["mode"])
    if job.get("fast_preview") and job["mode"] == "video":
        width, height = 854, 480
    else:
        width, height = config.width, config.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    clips_dir = destination.parent / "clips"
    started = time.monotonic()
    clip_paths: list[Path | None] = [None] * len(scenes)
    completed = 0
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="scene-render") as pool:
        futures = {
            pool.submit(_render_scene_clip, job, scene, asset, index, clips_dir / f"scene-{index + 1:03}.mp4", width, height): index
            for index, (scene, asset) in enumerate(zip(scenes, assets))
        }
        for future in as_completed(futures):
            index = futures[future]
            clip_paths[index] = future.result()
            completed += 1
            if progress:
                progress("Rendering scene clips", completed, len(scenes))
    clip_seconds = time.monotonic() - started

    final_started = time.monotonic()
    if progress:
        progress("Joining scene clips", 0, 1)
    ordered_clips = [path for path in clip_paths if path is not None]
    _concat_scene_clips(ordered_clips, destination)
    muxed = destination.with_name(f"{destination.stem}.muxed{destination.suffix}")
    _mux_narration(destination, muxed)
    destination.unlink(missing_ok=True)
    muxed.replace(destination)
    music_setting = os.getenv("MUSIC_FILE", "").strip()
    music_path = Path(music_setting) if music_setting else None
    if music_path and not music_path.is_file():
        raise RuntimeError(f"MUSIC_FILE is configured but does not exist or is not a file: {music_path}")
    if music_path:
        mixed = destination.with_name(f"{destination.stem}.mixed{destination.suffix}")
        _mix_music(destination, music_path, mixed)
        destination.unlink(missing_ok=True)
        mixed.replace(destination)
    if progress:
        progress("Joining scene clips", 1, 1)
    final_seconds = time.monotonic() - final_started
    logger.info("FFmpeg encoder selected: %s", _video_encoder()[0])
    logger.info("Render timings: scene_clips=%.2fs concat_and_mux=%.2fs", clip_seconds, final_seconds)
    return {"scene_clips": round(clip_seconds, 2), "final_render": round(final_seconds, 2)}