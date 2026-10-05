from __future__ import annotations

from .common import job_directory, log


def prepare_effects(job: dict) -> dict:
    level = job.get("effects_level", "cinematic")
    if level not in ("minimal", "cinematic", "dynamic"):
        raise ValueError(f"Unsupported effects level: {level}")
    result = {
        "level": level,
        "motion_strength": {"minimal": 1.015, "cinematic": 1.04, "dynamic": 1.065}[level],
        "transition": "fade" if level == "minimal" else "dissolve" if level == "cinematic" else "slide",
        "color_grade": level != "minimal",
        "vignette": level == "dynamic",
    }
    from .common import save_json

    save_json(job_directory(job) / "effects.json", result)
    log("effects", "Prepared deterministic cinematic movement and transitions", level=level)
    return result


if __name__ == "__main__":
    from .common import cli_main

    cli_main(prepare_effects)
