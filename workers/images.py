from __future__ import annotations

import hashlib
import importlib.util
import io
import gc
import os
import shutil
import threading
import time
from pathlib import Path

from PIL import Image, ImageStat

from .common import CONFIG, configured_path, job_directory, output_directory, emit, log, save_json

_gpu_lock = threading.Lock()
_STYLE_SUFFIXES = {
    "cinematic": "cinematic illustration, warm soft colors, young Indian girl with long black hair, warm light",
    "realistic": "realistic illustration, warm soft colors, young Indian girl with long black hair, warm light",
    "storybook": "storybook illustration, warm soft colors, young Indian girl with long black hair, warm light",
    "cartoon": "cartoon illustration, bright soft colors, young Indian girl with long black hair, warm light",
    "anime": "anime illustration, rich soft colors, young Indian girl with long black hair, warm light",
}


def _perceptual_hash(path: str | Path) -> int:
    with Image.open(path) as source:
        pixels = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(pixels.getdata())
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(values[row * 9 + column] > values[row * 9 + column + 1])
    return result


def _image_quality_issue(path: str | Path) -> str | None:
    with Image.open(path) as source:
        stats = ImageStat.Stat(source.convert("RGB"))
        mean = sum(stats.mean) / 3
        deviation = sum(stats.stddev) / 3
    if deviation < 2.0:
        return "Image is blank or nearly uniform."
    if mean < 3 or mean > 252:
        return f"Image is nearly black or white (mean brightness {mean:.1f})."
    return None


def _hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_near_duplicate_images(paths: list[str | Path], threshold: int = 4) -> list[tuple[int, int]]:
    hashes = [_perceptual_hash(path) for path in paths]
    return [
        (previous + 1, current + 1)
        for current, value in enumerate(hashes)
        for previous in range(current)
        if _hash_distance(value, hashes[previous]) <= threshold
    ]


def _clip_prompt_matcher():
    from .doctor import _cached_snapshot

    snapshot = (
        _cached_snapshot("openai/clip-vit-base-patch32", "model.safetensors")
        or _cached_snapshot("openai/clip-vit-base-patch32", "pytorch_model.bin")
    )
    if snapshot is None:
        log("images", "CLIP prompt similarity check skipped; no CLIP weights are cached locally")
        return None
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Cached CLIP weights were found, but transformers is not installed for offline similarity checks.")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True).to("cpu").eval()

    def similarity(image_path: str | Path, prompt: str) -> float:
        with Image.open(image_path) as source:
            inputs = processor(text=[prompt], images=[source.convert("RGB")], return_tensors="pt", padding=True)
        with torch.inference_mode():
            output = model(**inputs)
            image_vector = output.image_embeds[0]
            text_vector = output.text_embeds[0]
            score = torch.nn.functional.cosine_similarity(image_vector, text_vector, dim=0)
        return float(score.item())

    return similarity


def _create_pipeline(model: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=os.getenv("HF_HUB_CACHE") or str(configured_path("cache") / "huggingface"),
        local_files_only=True,
    )
    pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    pipeline.enable_model_cpu_offload()
    return pipeline


def _prompt_token_count(tokenizer, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
    token_ids = encoded["input_ids"]
    return len(token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids)


def _bounded_prompt(pipeline, scene_prompt: str, style: str, scene_number: int) -> tuple[str, int]:
    suffix = _STYLE_SUFFIXES.get(style, _STYLE_SUFFIXES["cinematic"])
    words = scene_prompt.split()[:50]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    def compose() -> str:
        return f"{' '.join(words)}, {suffix}"

    prompt = compose()
    token_count = max(_prompt_token_count(tokenizer, prompt) for tokenizer in tokenizers)
    if token_count > 77:
        log("images", "Image prompt exceeds the CLIP 77-token limit",
            scene=scene_number, tokens=token_count, limit=77, words=len(words))
    elif token_count > 70:
        log("images", "Image prompt exceeds the 70-token generation budget",
            scene=scene_number, tokens=token_count, limit=70, words=len(words))
    if token_count > 70:
        while token_count > 70 and len(words) > 40:
            words.pop()
            prompt = compose()
            token_count = max(_prompt_token_count(tokenizer, prompt) for tokenizer in tokenizers)
        if token_count > 70:
            raise ValueError(f"Scene {scene_number} image prompt cannot fit the 40-word minimum under the 70-token budget ({token_count} tokens).")
        log("images", "Image prompt trimmed to fit both CLIP encoders",
            scene=scene_number, tokens=token_count, words=len(words))
    return prompt, token_count


def _is_cuda_oom(error: Exception, torch) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
    return (bool(oom_type) and isinstance(error, oom_type)) or "out of memory" in str(error).lower()


def generate_images(job: dict, script: dict) -> list[dict]:
    try:
        import torch
    except ImportError:
        torch = None

    output = job_directory(job) / "images"
    cache = configured_path("cache") / "images"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    short = job["mode"] == "shorts"
    width, height = (768, 1344) if short else (1344, 768)
    model = os.getenv("SDXL_MODEL", CONFIG["models"]["image"])
    configured_steps = {"fast": 4, "balanced": 6, "high": 8}.get(job.get("quality", "balanced"), CONFIG["models"]["image_steps"])
    steps = int(os.getenv("SDXL_STEPS", configured_steps))
    pipeline = None
    pipeline_error = None
    results = []
    prompt_by_scene: dict[int, str] = {}
    seed_by_scene: dict[int, int] = {}
    seed_base = int(hashlib.sha256(job["id"].encode()).hexdigest()[:8], 16)
    try:
        for index, scene in enumerate(script["scenes"]):
            scene_prompt = scene["image_prompt"]
            suffix = _STYLE_SUFFIXES.get(job.get("style"), _STYLE_SUFFIXES["cinematic"])
            prompt = f"{scene_prompt}, {suffix}"
            digest = hashlib.sha256(f"{prompt}|{job.get('style')}|{seed_base}|{width}x{height}".encode()).hexdigest()
            target = output / f"scene-{index + 1:03}.jpg"
            cached = cache / f"{digest}.jpg"
            if cached.is_file() and cached.stat().st_size > 1024:
                target.write_bytes(cached.read_bytes())
                results.append({"scene": index + 1, "path": str(target), "provider": "local-sdxl-cache", "placeholder": False, "seed": seed_base + index})
                prompt_by_scene[index + 1] = prompt
                seed_by_scene[index + 1] = seed_base + index
                preview = output_directory(job) / "scenes" / f"scene-{index + 1:03}.jpg"
                preview.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, preview)
                results[-1]["preview_url"] = f"/outputs/{job['id']}/scenes/{preview.name}"
                emit("stage", job_id=job["id"], stage="images", status="running", progress=index + 1, total=len(script["scenes"]), scene=results[-1])
                continue
            seed = seed_base + index
            started = time.monotonic()
            try:
                if torch is None:
                    raise RuntimeError("Local SDXL dependencies are not installed. Run scripts/setup.ps1 -InstallML.")
                if pipeline_error:
                    raise RuntimeError(pipeline_error)
                with _gpu_lock:
                    if pipeline is None:
                        try:
                            pipeline = _create_pipeline(model)
                        except Exception as error:
                            pipeline_error = f"Could not load local SDXL model {model}: {type(error).__name__}: {error}"
                            raise RuntimeError(pipeline_error) from error
                    prompt, token_count = _bounded_prompt(pipeline, scene_prompt, job.get("style", "cinematic"), index + 1)
                    log("images", "CLIP prompt length verified", scene=index + 1,
                        tokens=token_count, words=len(scene_prompt.split()))
                    digest = hashlib.sha256(f"{prompt}|{job.get('style')}|{seed_base}|{width}x{height}".encode()).hexdigest()
                    cached = cache / f"{digest}.jpg"
                    if cached.is_file() and cached.stat().st_size > 1024:
                        target.write_bytes(cached.read_bytes())
                        result = {"scene": index + 1, "path": str(target), "provider": "local-sdxl-cache", "placeholder": False}
                    else:
                        generator = torch.Generator(device="cpu").manual_seed(seed)

                        def generate_at_size(image_width: int, image_height: int):
                            return pipeline(
                                prompt=prompt,
                                negative_prompt="text, watermark, logo, deformed hands, extra fingers, duplicate people, blurry",
                                width=image_width,
                                height=image_height,
                                num_inference_steps=max(1, min(8, steps)),
                                guidance_scale=float(CONFIG["models"]["image_guidance"]),
                                generator=generator,
                            ).images[0]

                        try:
                            image = generate_at_size(width, height)
                            image_width, image_height = width, height
                        except Exception as error:
                            if torch is None or not _is_cuda_oom(error, torch):
                                raise
                            log("images", "CUDA out of memory; retrying this scene at reduced resolution",
                                scene=index + 1, initial_resolution=f"{width}x{height}",
                                error=f"{type(error).__name__}: {error}")
                            gc.collect()
                            torch.cuda.empty_cache()
                            image = None
                            for retry_width, retry_height in ((768, 768), (512, 512)):
                                try:
                                    log("images", "Retrying scene after CUDA OOM",
                                        scene=index + 1, resolution=f"{retry_width}x{retry_height}")
                                    generator = torch.Generator(device="cpu").manual_seed(seed)
                                    image = generate_at_size(retry_width, retry_height)
                                    image_width, image_height = retry_width, retry_height
                                    break
                                except Exception as retry_error:
                                    if not _is_cuda_oom(retry_error, torch):
                                        raise
                                    gc.collect()
                                    torch.cuda.empty_cache()
                            if image is None:
                                raise RuntimeError(f"SDXL exhausted GPU memory at {width}x{height}, 768x768 and 512x512.") from error
                        buffer = io.BytesIO()
                        image.convert("RGB").save(buffer, format="JPEG", quality=92)
                        data = buffer.getvalue()
                        cached.write_bytes(data)
                        target.write_bytes(data)
                        del image
                        result = {
                            "scene": index + 1, "path": str(target), "provider": "local-sdxl",
                            "placeholder": False, "seed": seed,
                            "generated_resolution": f"{image_width}x{image_height}",
                        }
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                result.setdefault("scene", index + 1)
                result["path"] = str(target)
            except Exception as error:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                target.unlink(missing_ok=True)
                raise RuntimeError(f"Image generation failed for scene {index + 1}: {type(error).__name__}: {error}") from error
            prompt_by_scene[index + 1] = prompt
            seed_by_scene[index + 1] = int(result.get("seed", seed))
            results.append(result)
            log("images", "Scene image complete", scene=index + 1, provider=result["provider"], seconds=round(time.monotonic() - started, 2))
            preview = output_directory(job) / "scenes" / f"scene-{index + 1:03}.jpg"
            preview.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, preview)
            result["preview_url"] = f"/outputs/{job['id']}/scenes/{preview.name}"
            emit("stage", job_id=job["id"], stage="images", status="running", progress=index + 1, total=len(script["scenes"]), scene=result)
        if len(results) != len(script["scenes"]):
            raise RuntimeError(f"Image count mismatch: expected {len(script['scenes'])}, generated {len(results)}.")
        initially_duplicated = find_near_duplicate_images([item["path"] for item in results])
        log("images", "Initial perceptual duplicate scan complete",
            duplicate_pairs=initially_duplicated, count=len(initially_duplicated))
        hashes: list[tuple[int, int]] = []
        for index, result in enumerate(results):
            problem = _image_quality_issue(result["path"])
            if problem:
                raise RuntimeError(f"Image QA failed for scene {index + 1}: {problem}")
            image_hash = _perceptual_hash(result["path"])
            duplicate = next(
                ((prior_scene, distance) for prior_scene, prior_hash in hashes
                 if (distance := _hash_distance(image_hash, prior_hash)) <= 4),
                None,
            )
            if duplicate:
                if torch is None:
                    raise RuntimeError(f"Duplicate image detected for scenes {duplicate[0]} and {index + 1}; regeneration is unavailable.")
                if pipeline is None:
                    pipeline = _create_pipeline(model)
                retry_seed = seed_base + index + 100003
                prompt = prompt_by_scene[index + 1]
                log("images", "Near-duplicate scene image detected; regenerating with a new seed",
                    scene=index + 1, duplicate_of=duplicate[0], retry_seed=retry_seed)
                with _gpu_lock:
                    regenerated = pipeline(
                        prompt=prompt,
                        negative_prompt="text, watermark, logo, deformed hands, extra fingers, duplicate people, blurry",
                        width=width, height=height,
                        num_inference_steps=max(1, min(8, steps)),
                        guidance_scale=float(CONFIG["models"]["image_guidance"]),
                        generator=torch.Generator(device="cpu").manual_seed(retry_seed),
                    ).images[0]
                regenerated.convert("RGB").save(result["path"], format="JPEG", quality=92)
                result["seed"] = retry_seed
                seed_by_scene[index + 1] = retry_seed
                image_hash = _perceptual_hash(result["path"])
                if any(_hash_distance(image_hash, prior_hash) <= 4 for _, prior_hash in hashes):
                    raise RuntimeError(f"Duplicate image persists after one new-seed regeneration for scene {index + 1}.")
            hashes.append((index + 1, image_hash))
        clip_similarity = _clip_prompt_matcher()
        if clip_similarity is not None:
            for index, result in enumerate(results):
                score = clip_similarity(result["path"], script["scenes"][index]["image_prompt"])
                if score < 0.2:
                    if torch is None:
                        raise RuntimeError(f"Scene {index + 1} image similarity is low ({score:.3f}) and SDXL cannot regenerate it.")
                    if pipeline is None:
                        pipeline = _create_pipeline(model)
                    retry_seed = seed_base + index + 200003
                    log("images", "CLIP prompt match is low; regenerating once with a new seed",
                        scene=index + 1, similarity=round(score, 3), retry_seed=retry_seed)
                    with _gpu_lock:
                        regenerated = pipeline(
                            prompt=prompt_by_scene[index + 1],
                            negative_prompt="text, watermark, logo, deformed hands, extra fingers, duplicate people, blurry",
                            width=width, height=height,
                            num_inference_steps=max(1, min(8, steps)),
                            guidance_scale=float(CONFIG["models"]["image_guidance"]),
                            generator=torch.Generator(device="cpu").manual_seed(retry_seed),
                        ).images[0]
                    regenerated.convert("RGB").save(result["path"], format="JPEG", quality=92)
                    result["seed"] = retry_seed
                    seed_by_scene[index + 1] = retry_seed
                    score = clip_similarity(result["path"], script["scenes"][index]["image_prompt"])
                    if score < 0.2:
                        raise RuntimeError(
                            f"Scene {index + 1} image still poorly matches its English prompt after regeneration "
                            f"(CLIP similarity {score:.3f}; minimum 0.200)."
                        )
                result["clip_similarity"] = round(score, 4)
        for index, result in enumerate(results):
            problem = _image_quality_issue(result["path"])
            if problem:
                raise RuntimeError(f"Image QA failed after regeneration for scene {index + 1}: {problem}")
        remaining_duplicates = find_near_duplicate_images([item["path"] for item in results])
        if remaining_duplicates:
            raise RuntimeError(f"Near-duplicate images remain after regeneration: {remaining_duplicates}.")
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("images", "SDXL pipeline unloaded; garbage collection and CUDA cache cleanup complete")
    directory = job_directory(job)
    save_json(directory / "images.json", results)
    prompts = "\n".join(f"{index + 1}. {scene['image_prompt']}" for index, scene in enumerate(script["scenes"]))
    (directory / "image_prompts.txt").write_text(prompts, encoding="utf-8")
    save_json(directory / "image_prompts.json", [
        {
            "scene": index + 1,
            "image_prompt": scene["image_prompt"],
            "sdxl_prompt": prompt_by_scene[index + 1],
            "seed": seed_by_scene[index + 1],
        }
        for index, scene in enumerate(script["scenes"])
    ])
    return results


if __name__ == "__main__":
    from .common import cli_main, load_job

    cli_main(lambda job: generate_images(job, load_job(job_directory(job) / "script.json")))
