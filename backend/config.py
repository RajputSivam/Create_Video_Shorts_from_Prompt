from dataclasses import dataclass


@dataclass(frozen=True)
class ModeConfig:
    width: int
    height: int
    min_seconds: int
    max_seconds: int
    scene_min: int
    scene_max: int
    target_scene_min: int
    target_scene_max: int
    zoom: float
    transition: str
    caption_style: str
    words_per_minute: int


MODE_CONFIGS = {
    "video": ModeConfig(1280, 720, 300, 900, 15, 20, 16, 48, 0.025, "cut", "sentence", 150),
    "shorts": ModeConfig(1080, 1920, 15, 60, 3, 6, 6, 10, 0.11, "punch", "word", 140),
}

STYLE_PREFIXES = {
    "cinematic": "cinematic film still, dramatic natural lighting, rich color grading, 35mm lens,",
    "cartoon": "polished 2D animated feature, expressive characters, clean shapes, vibrant palette,",
    "realistic": "photorealistic documentary frame, natural light, authentic details,",
    "storybook": "hand-painted storybook illustration, textured brushwork, whimsical composition,",
}


def get_mode_config(mode: str) -> ModeConfig:
    return MODE_CONFIGS[mode]