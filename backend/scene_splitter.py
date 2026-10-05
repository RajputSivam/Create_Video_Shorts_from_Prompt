from pathlib import Path
from math import ceil

from .config import get_mode_config
from .providers import generate_json
from .storage import write_text_atomic


def split_scenes(job: dict, script: str, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        import json
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        expected = max(1, round(job["duration"] / 18))
        if job["mode"] == "shorts" or len(cached) <= max(expected + 2, round(expected * 1.25)):
            return cached

    config = get_mode_config(job["mode"])
    if job["mode"] == "shorts":
        count = min(config.target_scene_max, max(config.target_scene_min, round(job["duration"] / 5)))
        scene_duration = f"{config.scene_min}-{config.scene_max} seconds"
    else:
        count = min(ceil(job["duration"] / 15), max(ceil(job["duration"] / 20), round(job["duration"] / 18)))
        scene_duration = "15-20 seconds"
    framing = "Compose for 9:16 vertical framing, with the subject centered and clear of top/bottom UI safe zones." if job["mode"] == "shorts" else "Compose for widescreen 16:9 landscape framing."
    prompt = (
        f"Split this narration into about {count} sequential visual scenes. Each scene duration should be {scene_duration}. {framing}\n"
        "Return only a JSON object with a character_description (a concrete, visually consistent description of the recurring "
        "main character) and a scenes array. Each scene object must have: narration_text (verbatim consecutive narration), "
        "image_prompt (specific visual composition for this scene), duration (seconds). Keep the narration in order, "
        "cover it all, and make each scene duration roughly equal. No markdown.\n\n"
        f"Narration:\n{script}"
    )
    result = generate_json(prompt)
    character = ""
    if isinstance(result, dict):
        character = str(result.get("character_description", "")).strip()
        scenes = result.get("scenes", [])
    else:
        scenes = result
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("Scene generation returned an empty result.")
    normalized = []
    for scene in scenes:
        text = str(scene.get("narration_text", "")).strip()
        image_prompt = str(scene.get("image_prompt", "")).strip()
        if text and image_prompt:
            normalized.append({"narration_text": text, "image_prompt": image_prompt, "duration": float(scene.get("duration", 5)), "character_description": character})
    if not normalized:
        raise RuntimeError("Scene generation did not include usable narration and image prompts.")
    if job["mode"] == "video" and len(normalized) > count:
        grouped = []
        for group_index in range(count):
            start = group_index * len(normalized) // count
            end = (group_index + 1) * len(normalized) // count
            group = normalized[start:end]
            prompts = list(dict.fromkeys(item["image_prompt"] for item in group))
            grouped.append({
                "narration_text": " ".join(item["narration_text"] for item in group),
                "image_prompt": "; ".join(prompts),
                "duration": sum(item["duration"] for item in group),
                "character_description": character,
            })
        normalized = grouped
    import json
    write_text_atomic(cache_path, json.dumps(normalized, ensure_ascii=False, indent=2))
    return normalized