from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import CONFIG, emit, ffmpeg_path, log, output_directory
from .images import _hash_distance, _perceptual_hash
from .qa import check


def _ffprobe() -> str:
    candidate = os.getenv("FFPROBE_PATH") or shutil.which("ffprobe")
    if candidate:
        return candidate
    if sys.platform == "win32":
        packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        matches = sorted(packages.glob("Gyan.FFmpeg.Shared*/*/bin/ffprobe.exe"))
        if matches:
            return str(matches[-1])
    raise RuntimeError("ffprobe is required for local video validation. Install FFmpeg and add ffprobe to PATH.")


def _run(command: list[str], label: str) -> subprocess.CompletedProcess:
    log("render", "Running FFmpeg command", label=label, command=subprocess.list2cmdline(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3600)
    if result.returncode:
        log("render", "FFmpeg command failed", label=label, code=result.returncode, stderr=result.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {result.stderr}")
    return result


def _subtitle_filter(path: Path, job: dict) -> str:
    escaped = str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    font = "Nirmala UI" if job.get("language") == "Hindi" else "Arial"
    size = CONFIG["video"]["subtitle_font_size_short" if job["mode"] == "shorts" else "subtitle_font_size_long"]
    margin_v = 420 if job["mode"] == "shorts" else 130
    return f"subtitles=filename='{escaped}':force_style='FontName={font},FontSize={size},Bold=1,Outline=4,Shadow=1,Alignment=2,MarginL=80,MarginR=80,MarginV={margin_v}'"


def _scene_clip(job: dict, index: int, scene: dict, image: dict, voice: dict, sfx: dict, effects: dict, destination: Path) -> float:
    duration = float(voice["duration"]) + 0.3
    if duration <= 0:
        raise ValueError(f"Scene {index + 1} has an invalid narration duration.")
    width = CONFIG["video"]["short_width" if job["mode"] == "shorts" else "long_width"]
    height = CONFIG["video"]["short_height" if job["mode"] == "shorts" else "long_height"]
    fps = int(CONFIG["video"]["fps"])
    zoom = float(effects["motion_strength"])
    motion = scene.get("motion", "zoom_in")
    if motion in ("pan_left", "pan_right", "pan_up", "pan_down"):
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
        if motion == "pan_left":
            x = "max(0,(iw-iw/zoom)*(1-on/max(1,%d)))" % int(duration * fps)
        elif motion == "pan_right":
            x = "min(iw-iw/zoom,(iw-iw/zoom)*on/max(1,%d))" % int(duration * fps)
        elif motion == "pan_up":
            y = "max(0,(ih-ih/zoom)*(1-on/max(1,%d)))" % int(duration * fps)
        else:
            y = "min(ih-ih/zoom,(ih-ih/zoom)*on/max(1,%d))" % int(duration * fps)
    else:
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    zoom_filter = f"zoompan=z='min(1+on*{(zoom - 1) / max(1, int(duration * fps)):.8f},{zoom:.4f})':x='{x}':y='{y}':d=1:s={width}x{height}:fps={fps}"
    video_filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        zoom_filter,
    ]
    if effects["color_grade"]:
        video_filters.append("eq=saturation=1.06:contrast=1.035")
    if effects["vignette"]:
        video_filters.append("vignette=PI/5")
    effect_name = "whoosh" if scene.get("sfx") == "whoosh" else "ambience"
    audio_filter = "[1:a]apad=pad_dur=0.3,atrim=duration=%.3f,asetpts=PTS-STARTPTS[voice];[2:a]volume=0.20[fx];[voice][fx]amix=inputs=2:duration=first:normalize=0[a]" % duration
    command = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-i", image["render_path"],
        "-i", voice["path"], "-stream_loop", "-1", "-i", sfx[effect_name],
        "-filter_complex", f"[0:v]{','.join(video_filters)}[v];{audio_filter}",
        "-map", "[v]", "-map", "[a]", "-t", f"{duration:.4f}",
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "24",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-movflags", "+faststart", str(destination),
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=900)
    if result.returncode:
        log("render", "Hardware render failed; retrying scene with libx264", scene=index + 1, command=subprocess.list2cmdline(command), stderr=result.stderr)
        command[command.index("-c:v") + 1] = "libx264"
        preset_index = command.index("-preset") + 1
        command[preset_index] = "veryfast"
        command = [argument for argument in command if argument not in ("-rc", "vbr", "-cq", "24")]
        log("render", "Retrying scene with software encoder", scene=index + 1, command=subprocess.list2cmdline(command))
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1800)
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Scene {index + 1} FFmpeg render failed with exit code {result.returncode}: {result.stderr}")
    log("render", "Rendered scene clip", scene=index + 1, seconds=round(time.monotonic() - started, 2), audio_seconds=round(duration, 2))
    return duration


def _final_render_command(job: dict, clips: list[Path], clip_durations: list[float], music_path: Path, captions: Path, effects: dict, destination: Path) -> list[str]:
    fps = int(CONFIG["video"]["fps"])
    transition = {"fade": "fade", "dissolve": "dissolve", "slide": "slideleft"}.get(effects["transition"], "fade")
    transition_duration = min(0.25, min(clip_durations) / 3) if len(clips) > 1 else 0
    command = [ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    music_index = len(clips)
    command.extend(["-stream_loop", "-1", "-i", str(music_path)])
    graph = []
    for index in range(len(clips)):
        graph.extend([
            f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps},format=yuv420p[v{index}]",
            f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}]",
        ])
    video_label, audio_label = "v0", "a0"
    accumulated_duration = clip_durations[0]
    for index in range(1, len(clips)):
        next_video, next_audio = f"vjoin{index}", f"ajoin{index}"
        offset = max(0, accumulated_duration - transition_duration)
        graph.append(f"[{video_label}][v{index}]xfade=transition={transition}:duration={transition_duration:.3f}:offset={offset:.3f}[{next_video}]")
        graph.append(f"[{audio_label}][a{index}]acrossfade=d={transition_duration:.3f}:c1=tri:c2=tri[{next_audio}]")
        video_label, audio_label = next_video, next_audio
        accumulated_duration += clip_durations[index] - transition_duration
    graph.append(f"[{video_label}]{_subtitle_filter(captions, job)}[captioned]")
    graph.append(f"[{music_index}:a]volume=0.12[music]")
    graph.append(f"[{audio_label}][music]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    command.extend([
        "-filter_complex", ";".join(graph),
        "-map", "[captioned]", "-map", "[a]",
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "24",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-pix_fmt", "yuv420p",
        "-shortest", "-movflags", "+faststart", str(destination),
    ])
    return command


def _make_thumbnail(video: Path, destination: Path) -> None:
    command = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-ss", "00:00:01",
        "-i", str(video), "-frames:v", "1", "-vf", "scale=640:-2", str(destination),
    ]
    _run(command, "thumbnail generation")


def _probe_video(path: Path, expected_duration: float, real_images: int, image_failures: list[str] | None = None, mode: str = "shorts") -> dict:
    command = [_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    log("render", "ffprobe validation command", command=subprocess.list2cmdline(command))
    if result.returncode:
        raise RuntimeError(f"ffprobe validation failed: {result.stderr}")
    log("render", "ffprobe output", output=result.stdout, stderr=result.stderr)
    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    types = {stream.get("codec_type") for stream in streams}
    if not {"video", "audio"}.issubset(types):
        raise RuntimeError(f"Rendered video is missing media streams: found {sorted(types)}.")
    if real_images < 1:
        reason = "; ".join(image_failures or [])
        detail = f" Provider details: {reason}" if reason else ""
        raise RuntimeError(f"Rendered video has no real generated images; every scene used a gradient fallback.{detail}")
    duration = float(probe.get("format", {}).get("duration", 0))
    if duration < expected_duration * 0.75 or duration > expected_duration * 1.25:
        raise RuntimeError(f"Rendered duration {duration:.1f}s does not match requested target {expected_duration:.1f}s ±25%.")
    video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
    expected_width, expected_height = (1080, 1920) if mode == "shorts" else (1920, 1080)
    if video_stream.get("width") != expected_width or video_stream.get("height") != expected_height:
        raise RuntimeError(
            f"Rendered resolution {video_stream.get('width')}x{video_stream.get('height')} does not match "
            f"expected {expected_width}x{expected_height}."
        )
    fps_value = video_stream.get("avg_frame_rate", "0/1")
    numerator, _, denominator = fps_value.partition("/")
    fps = float(numerator) / max(1.0, float(denominator))
    expected_fps = float(CONFIG["video"]["fps"])
    if abs(fps - expected_fps) > 0.1:
        raise RuntimeError(f"Rendered FPS {fps:.2f} does not match configured {expected_fps:.2f}.")
    max_bytes = int(CONFIG["video"]["max_file_size_bytes"])
    if path.stat().st_size < 4096 or path.stat().st_size > max_bytes:
        raise RuntimeError(f"Rendered file size is not sane: {path.stat().st_size} bytes.")
    return {
        "duration": duration,
        "video_streams": sum(stream.get("codec_type") == "video" for stream in streams),
        "audio_streams": sum(stream.get("codec_type") == "audio" for stream in streams),
        "size_bytes": path.stat().st_size,
        "real_images": real_images,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
    }


def _mean_audio_volume(path: Path) -> float:
    result = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(path), "-map", "0:a:0", "-af", "volumedetect",
         "-f", "null", "NUL" if os.name == "nt" else "/dev/null"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    import re

    match = re.search(r"mean_volume:\s*(-?inf|-?[\d.]+)\s*dB", result.stderr, re.IGNORECASE)
    if result.returncode or not match:
        raise RuntimeError(f"Could not measure rendered audio level: {result.stderr.strip()}")
    return float("-inf") if "inf" in match.group(1).lower() else float(match.group(1))


def _sample_frame_similarity(path: Path, duration: float) -> dict:
    times = list(range(0, max(1, int(duration)), 3))
    hashes = []
    with tempfile.TemporaryDirectory(prefix="video-qa-frames-") as folder:
        for index, seconds in enumerate(times):
            frame = Path(folder) / f"frame-{index:04}.jpg"
            result = subprocess.run(
                [ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-ss", str(seconds),
                 "-i", str(path), "-frames:v", "1", str(frame)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            if result.returncode or not frame.is_file():
                raise RuntimeError(f"Could not extract QA frame at {seconds}s: {result.stderr}")
            hashes.append(_perceptual_hash(frame))
    duplicates = sum(
        any(_hash_distance(value, previous) <= 4 for previous in hashes[:index])
        for index, value in enumerate(hashes)
    )
    return {
        "frames_sampled": len(hashes),
        "similar_frames": duplicates,
        "similar_frame_ratio": duplicates / max(1, len(hashes)),
    }


def render(job: dict, script: dict, images: list[dict], voices: list[dict], sfx: dict, music_path: Path, captions: Path, effects: dict) -> dict:
    directory = output_directory(job)
    clips_dir = directory / "scene-clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips, clip_durations = [], []
    for index, (scene, image, voice) in enumerate(zip(script["scenes"], images, voices)):
        clip = clips_dir / f"scene-{index + 1:03}.mp4"
        if not clip.exists() or clip.stat().st_size < 4096:
            clip_duration = _scene_clip(job, index, scene, image, voice, sfx, effects, clip)
        else:
            clip_duration = float(voice["duration"]) + 0.3
        clips.append(clip)
        clip_durations.append(clip_duration)
        emit("stage", job_id=job["id"], stage="render", status="running", progress=index + 1, total=len(script["scenes"]))
    final = directory / "video.mp4"
    render_command = _final_render_command(job, clips, clip_durations, music_path, captions, effects, final)
    try:
        _run(render_command, "transitions, subtitles, audio mix and final render")
    except RuntimeError as error:
        log("render", "Hardware final render failed; retrying with libx264", command=subprocess.list2cmdline(render_command), error=str(error))
        render_command[render_command.index("-c:v") + 1] = "libx264"
        render_command[render_command.index("-preset") + 1] = "veryfast"
        render_command = [argument for argument in render_command if argument not in ("-rc", "vbr", "-cq", "24")]
        log("render", "Retrying final render with software encoder", command=subprocess.list2cmdline(render_command))
        _run(render_command, "software transitions, subtitles, audio mix and final render")
    duration = sum(clip_durations) - max(0, len(clips) - 1) * (min(0.25, min(clip_durations) / 3) if len(clips) > 1 else 0)
    real_images = sum(not image.get("placeholder") for image in images)
    probe = _probe_video(
        final, float(job["duration"]), real_images,
        [image.get("error") for image in images if image.get("error")], job["mode"],
    )
    audio_level = _mean_audio_volume(final)
    if audio_level <= -45:
        raise RuntimeError(f"Rendered audio is silent or too quiet (mean volume {audio_level:.1f} dB).")
    frames = _sample_frame_similarity(final, probe["duration"])
    frame_check = check(
        "WARN" if frames["similar_frame_ratio"] > 0.7 else "PASS",
        "More than 70% of sampled frames are perceptually similar." if frames["similar_frame_ratio"] > 0.7
        else "Frame samples show scene variation.",
        **frames,
    )
    video_check = check(
        "PASS", "ffprobe confirms video/audio streams, requested dimensions and FPS, duration, and nonempty file.",
        duration_seconds=probe["duration"], width=probe["width"], height=probe["height"],
        fps=probe["fps"], size_bytes=probe["size_bytes"], audio_level_db=round(audio_level, 2),
    )
    thumbnail = directory / "thumbnail.jpg"
    _make_thumbnail(final, thumbnail)
    metadata = {
        "title": script["title"],
        "description": script["description"],
        "tags": script["tags"],
        "duration_seconds": probe["duration"],
        "resolution": [1080, 1920] if job["mode"] == "shorts" else [1920, 1080],
        "fps": CONFIG["video"]["fps"],
        "mode": job["mode"],
        "local_only": True,
        "real_images": real_images,
        "warnings": [image["error"] for image in images if image.get("error")],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "video": str(final), "thumbnail": str(thumbnail), "metadata": str(directory / "metadata.json"),
        "qa_checks": {"final_video": video_check, "frame_variation": frame_check},
        "qa_metrics": {"rendered_duration_seconds": probe["duration"], "audio_level_db": round(audio_level, 2), **frames},
        **probe,
    }


if __name__ == "__main__":
    from .common import cli_main, job_directory, load_job

    def _run(job):
        directory = job_directory(job)
        return render(
            job, load_job(directory / "script.json"), load_job(directory / "images.json"),
            load_job(directory / "voice.json"), load_job(directory / "sfx.json"),
            Path(load_job(directory / "music.json")["path"]), Path(load_job(directory / "captions.json")["path"]),
            load_job(directory / "effects.json"),
        )

    cli_main(_run)
