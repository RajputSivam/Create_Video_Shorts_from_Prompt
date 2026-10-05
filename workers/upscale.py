from __future__ import annotations

from PIL import Image

from .common import job_directory, log


def upscale_images(job: dict, image_results: list[dict]) -> list[dict]:
    width, height = (1080, 1920) if job["mode"] == "shorts" else (1920, 1080)
    for item in image_results:
        with Image.open(item["path"]) as image:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            target = job_directory(job) / "images" / f"scene-{item['scene']:03}-final.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            resized.save(target, format="JPEG", quality=91, optimize=True)
        item["render_path"] = str(target)
    log("upscale", "Scene frames upscaled using Lanczos", scenes=len(image_results), resolution=f"{width}x{height}")
    return image_results


if __name__ == "__main__":
    from .common import cli_main, load_job

    def _run(job):
        return upscale_images(job, load_job(job_directory(job) / "images.json"))

    cli_main(_run)
