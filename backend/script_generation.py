import asyncio
from pathlib import Path

from .config import get_mode_config
from .providers import generate_text
from .storage import write_text_atomic


def generate_script(job: dict, cache_path: Path, progress=None) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    config = get_mode_config(job["mode"])
    minutes = job["duration"] / 60
    target_words = round(job["duration"] * 140 / 45) if job["mode"] == "shorts" else round(minutes * config.words_per_minute)
    language = "Hindi" if job["language"] == "Hindi" else "English"
    if job["mode"] == "shorts":
        if progress:
            progress("Writing your script", 0, 1)
        prompt = (
            f"Write a spoken {language} short-form video script about: {job['prompt']}\n"
            f"Aim for {target_words} words for a {job['duration']}-second video. The very first sentence is a vivid, "
            "specific hook that lands within 3 seconds. Keep the pacing quick; skip greetings and long introductions. "
            "End with a satisfying loop-back or a concise call to action. Return only narration, no headings or stage directions."
        )
        script = asyncio.run(generate_text(prompt)).strip()
        if progress:
            progress("Writing your script", 1, 1)
    else:
        chapter_count = max(3, round(minutes * 0.75)) if job.get("fast_preview") else min(10, max(8, round(minutes * 0.75)))
        if progress:
            progress("Writing your outline", 0, chapter_count + 1)
        outline_prompt = (
            f"Create a compelling {chapter_count}-chapter outline for a {language} narrated video about: {job['prompt']}. "
            "Make the chapters build on one another, with a strong opening and a clear payoff. Return only a numbered outline."
        )
        outline = asyncio.run(generate_text(outline_prompt))
        if progress:
            progress("Writing your chapters", 1, chapter_count + 1)
        per_chapter = max(90, round(target_words / chapter_count))
        chapters = []
        for index in range(chapter_count):
            chapter_prompt = (
                f"Write chapter {index + 1} of a {chapter_count}-chapter narrated {language} video about: {job['prompt']}\n"
                f"Full outline:\n{outline}\nAim for about {per_chapter} spoken words. Continue naturally from the previous chapter; "
                "use vivid but narration-friendly language. Return only the narration, no title or stage directions."
            )
            chapters.append(asyncio.run(generate_text(chapter_prompt)).strip())
            if progress:
                progress("Writing your chapters", index + 2, chapter_count + 1)
        script = "\n\n".join(chapters)

    write_text_atomic(cache_path, script)
    return script