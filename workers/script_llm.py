from __future__ import annotations

import json
import math
import os
import re
import difflib
import time
import urllib.error
import urllib.request

from .common import CONFIG, job_directory, log, save_json

WORDS_PER_MINUTE = 140
HINDI_WORDS_PER_MINUTE = 156
PLACEHOLDER_PHRASES = (
    "early experiments",
    "unexpected setbacks",
    "surprising story behind",
    "here is the",
    "this is the story behind",
)


def _words_per_minute(job: dict) -> int:
    return HINDI_WORDS_PER_MINUTE if job.get("language") == "Hindi" else WORDS_PER_MINUTE


def expected_scene_count(job: dict) -> int:
    if job.get("scene_count") is not None:
        return int(job["scene_count"])
    duration = int(job["duration"])
    if job["mode"] == "shorts":
        return max(8, min(14, math.ceil(duration / 4.5)))
    return max(8, math.ceil(duration / 18))


def _prompt(job: dict, count: int, validation_feedback: str | None = None) -> str:
    language = "Hindi (Devanagari)" if job.get("language") == "Hindi" else "English"
    target_words = max(35, int(job["duration"] * _words_per_minute(job) / 60))
    core_topic = job["prompt"].strip()
    return f"""
You write a scene-by-scene video script. Treat the user input below strictly as topic and ordered instructions; never copy it into narration.
User topic and instructions: {core_topic}
Return only valid JSON with exactly this shape: {{"scenes":[{{"narration":"...","image_prompt":"..."}}]}}
Create {count} scenes (accepted output range {max(1, count - 1)}-{count + 1}) for a {job['duration']}-second {job['mode']} video. Follow the topic's scene points in their original order. Each scene must advance a distinct beat.
Narration must be only in {language}. For Hindi, use Devanagari only, not transliterated Hindi or English. Total narration length must be about {target_words} words, within 25 percent.
Every image_prompt must be ENGLISH ONLY, exactly 25-40 words, and describe only that scene's visible people, action, setting, lighting, and framing. Never put narration, Devanagari, placeholders, generic birds/flowers, text, logos, or watermarks in the image prompt unless specifically required by the topic.
Do not repeat the entire user prompt. Return no markdown or prose outside the JSON.
{f"Fix these validation errors in your next response: {validation_feedback}" if validation_feedback else ""}
"""


def _ollama_generate(job: dict, count: int, validation_feedback: str | None = None) -> dict:
    base = os.getenv("OLLAMA_URL", CONFIG["models"]["ollama_url"]).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", CONFIG["models"]["llm"])
    payload = json.dumps({
        "model": model,
        "prompt": _prompt(job, count, validation_feedback),
        "stream": False,
        "format": "json",
        "keep_alive": 0,
        "options": {"temperature": 0.7, "num_ctx": 8192},
    }).encode()
    request = urllib.request.Request(f"{base}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local Ollama returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Local Ollama is unavailable at {base}; start Ollama and pull {model}. Details: {error}") from error
    try:
        return json.loads(result["response"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("Ollama did not return valid JSON for the script.") from error


def verify_ollama_unloaded(required: bool = False) -> None:
    base = os.getenv("OLLAMA_URL", CONFIG["models"]["ollama_url"]).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", CONFIG["models"]["llm"])
    deadline = time.monotonic() + 30
    while True:
        request = urllib.request.Request(f"{base}/api/ps")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                loaded = json.loads(response.read()).get("models", [])
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Could not verify Ollama model unload: HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            if not required:
                log("script", "Ollama unload check skipped because the local endpoint is unavailable",
                    reason=str(error))
                return
            raise RuntimeError(f"Could not verify Ollama model unload at {base}: {error}") from error

        resident = [
            item.get("name", "")
            for item in loaded
            if item.get("name", "") == model or item.get("name", "").startswith(model + ":")
        ]
        if not resident:
            log("script", "Ollama model unloaded; memory is available for image generation", model=model)
            return
        if time.monotonic() >= deadline:
            unload_payload = json.dumps({"model": model, "prompt": "", "stream": False, "keep_alive": 0}).encode()
            unload_request = urllib.request.Request(
                f"{base}/api/generate",
                data=unload_payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(unload_request, timeout=30):
                    pass
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Could not unload Ollama model {model}: HTTP {error.code}: {body}") from error
            with urllib.request.urlopen(request, timeout=10) as response:
                remaining = json.loads(response.read()).get("models", [])
            still_loaded = [
                item.get("name", "")
                for item in remaining
                if item.get("name", "") == model or item.get("name", "").startswith(model + ":")
            ]
            if still_loaded:
                raise RuntimeError(f"Ollama model {model} is still resident before image generation: {', '.join(still_loaded)}")
            log("script", "Ollama model unloaded after explicit keep_alive=0 request", model=model)
            return
        time.sleep(1)


def _scene_language_ok(value: str, language: str) -> bool:
    if language == "Hindi":
        return bool(re.search(r"[\u0900-\u097f]", value))
    return not bool(re.search(r"[\u0900-\u097f]", value))


def _contains_placeholder_text(value: str) -> bool:
    lowered = value.casefold()
    return any(phrase in lowered for phrase in PLACEHOLDER_PHRASES)


def _save_debug_artifacts(job: dict, script: dict) -> None:
    directory = job_directory(job)
    lines = [f"{index + 1}. {scene['image_prompt']}" for index, scene in enumerate(script["scenes"])]
    (directory / "image_prompts.txt").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "language": job.get("language", "English"),
        "voice": job.get("voice", "female"),
        "mode": job.get("mode", "landscape"),
        "duration": job.get("duration"),
        "scene_count": len(script["scenes"]),
        "provider": script.get("provider", "ollama"),
    }
    save_json(directory / "language_voice.json", payload)


def validate_script(script: dict, job: dict, count: int, allow_enriched: bool = False) -> dict:
    if not isinstance(script, dict) or not isinstance(script.get("scenes"), list):
        raise ValueError("Script must be a JSON object containing a scenes array.")
    allowed_top_level = {"scenes"}
    if allow_enriched:
        allowed_top_level.update({"title", "description", "tags", "character_description", "word_count", "provider", "warning"})
    extra_top_level = set(script) - allowed_top_level
    if extra_top_level:
        raise ValueError(f"Script contains unsupported top-level fields: {', '.join(sorted(extra_top_level))}.")
    scene_count = len(script["scenes"])
    if not max(1, count - 1) <= scene_count <= count + 1:
        raise ValueError(f"Expected {count} scenes (allowing one either way); received {scene_count}.")
    for index, scene in enumerate(script["scenes"]):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {index + 1} must be a JSON object.")
        allowed_scene_fields = {"narration", "image_prompt"}
        if allow_enriched:
            allowed_scene_fields.update({"mood", "motion", "transition", "sfx", "duration_hint"})
        extra_scene_fields = set(scene) - allowed_scene_fields
        if extra_scene_fields:
            raise ValueError(f"Scene {index + 1} contains unsupported fields: {', '.join(sorted(extra_scene_fields))}.")
        for field in ("narration", "image_prompt"):
            if not isinstance(scene.get(field), str) or not scene[field].strip():
                raise ValueError(f"Scene {index + 1} is missing {field}.")
        scene["narration"] = scene["narration"].strip()
        scene["image_prompt"] = scene["image_prompt"].strip()
        if job.get("language") == "Hindi":
            letters = [character for character in scene["narration"] if character.isalpha()]
            devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
            ratio = devanagari / max(1, len(letters))
            if ratio < 0.7:
                raise ValueError(f"Scene {index + 1} narration is not Hindi: Devanagari letter ratio {ratio:.0%}, minimum 70%.")
        elif not _scene_language_ok(scene["narration"], job.get("language", "English")):
            raise ValueError(f"Scene {index + 1} narration language does not match the selected job language: {job.get('language', 'English')}.")
        else:
            letters = [character for character in scene["narration"] if character.isalpha()]
            english_letters = sum(character.isascii() for character in letters)
            if english_letters / max(1, len(letters)) < 0.7:
                raise ValueError(f"Scene {index + 1} narration is not English: Latin letter ratio is below 70%.")
        if re.search(r"[\u0900-\u097f]", scene["image_prompt"]):
            raise ValueError(f"Scene {index + 1} image prompt contains Devanagari text and must remain English only.")
        if any(not character.isascii() and character.isalpha() for character in scene["image_prompt"]):
            raise ValueError(f"Scene {index + 1} image prompt contains non-English letters.")
        if _contains_placeholder_text(scene["narration"]) or _contains_placeholder_text(scene["image_prompt"]):
            raise ValueError(f"Scene {index + 1} contains a generic placeholder phrase and must be regenerated.")
        user_topic = job.get("prompt", "").strip().lower()
        if user_topic:
            normalized = re.sub(r"\s+", " ", scene["narration"].lower())
            prompt_words = re.findall(r"\w+", user_topic)
            narration_words = re.findall(r"\w+", normalized)
            prompt_similarity = difflib.SequenceMatcher(None, " ".join(prompt_words), " ".join(narration_words)).ratio()
            if (len(user_topic) > 25 and user_topic in normalized) or prompt_similarity >= 0.8:
                raise ValueError(f"Scene {index + 1} narration copies the user prompt verbatim instead of telling the story.")
        prompt_word_count = len(scene["image_prompt"].split())
        if not 25 <= prompt_word_count <= 40:
            raise ValueError(f"Scene {index + 1} image_prompt must contain 25-40 words; received {prompt_word_count}.")
        scene.setdefault("mood", "calm")
        scene.setdefault("motion", "zoom_in")
        scene.setdefault("transition", "fade")
        scene.setdefault("sfx", "none")
        scene["duration_hint"] = int(job["duration"] / scene_count)
    words = sum(len(scene["narration"].split()) for scene in script["scenes"])
    target = int(job["duration"] * _words_per_minute(job) / 60)
    minimum, maximum = round(target * 0.75), round(target * 1.25)
    if words < max(20, minimum) or words > maximum:
        raise ValueError(
            f"Narration length is outside the 25% target range: {words} words for a "
            f"{job['duration']} second video (target about {target}, accepted {minimum}-{maximum})."
        )
    script.setdefault("title", "A story made for you")
    script.setdefault("description", f"A {job['duration']}-second {job['mode']} created from your prompt.")
    script.setdefault("tags", [job["mode"], "story"])
    script.setdefault("character_description", "")
    script["word_count"] = words
    return script


def generate_script(job: dict) -> dict:
    count = expected_scene_count(job)
    use_ollama = os.getenv("SCRIPT_PROVIDER", "ollama").lower()
    if use_ollama != "ollama":
        raise ValueError("SCRIPT_PROVIDER must be ollama; cloud LLMs are not enabled in this local pipeline.")

    last_error: Exception | None = None
    feedback = None
    for attempt in range(1, 4):
        try:
            raw = _ollama_generate(job, count, feedback)
            script = validate_script(raw, job, count)
            script["provider"] = "ollama"
            script["warning"] = None
            directory = job_directory(job)
            directory.mkdir(parents=True, exist_ok=True)
            save_json(directory / "script.json", script)
            _save_debug_artifacts(job, script)
            log("script", "Script validated", provider=script["provider"], scenes=count, words=script["word_count"])
            return script
        except Exception as error:
            last_error = error
            feedback = f"{type(error).__name__}: {error}"
            if attempt < 3:
                log("script", "Ollama script validation failed; retrying with corrective feedback",
                    attempt=attempt, reason=feedback)
    raise RuntimeError(f"Script generation failed: {last_error}")


if __name__ == "__main__":
    from .common import cli_main

    cli_main(generate_script)
