import textwrap

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _caption_image(text: str, width: int, height: int, mode: str, active_word: int | None = None) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    size = 54 if mode == "video" else 68
    font = _font(size)
    words = text.split()
    max_chars = 48 if mode == "video" else 22
    lines = textwrap.wrap(text, width=max_chars) or [text]
    line_height = size + 14
    y = max(int(height * 0.55), height - int(height * 0.09) - (len(lines) * line_height))
    if mode == "video":
        padding = 18
        for line in lines:
            bounds = draw.textbbox((0, 0), line, font=font, stroke_width=2)
            x = (width - (bounds[2] - bounds[0])) // 2
            draw.rounded_rectangle((x - padding, y - 8, x + bounds[2] - bounds[0] + padding, y + line_height - 5), radius=8, fill=(10, 18, 24, 165))
            draw.text((x, y), line, font=font, fill="white", stroke_width=2, stroke_fill=(0, 0, 0, 210))
            y += line_height
    else:
        start = max(0, (active_word or 0) - 3)
        visible = words[start:min(len(words), start + 7)]
        if not visible:
            visible = words[:1]
        line = " ".join(visible)
        while len(line) > max_chars and len(visible) > 1:
            visible.pop()
            line = " ".join(visible)
        line_y = height - int(height * 0.09) - line_height
        token_positions = []
        cursor = 0
        widths = []
        for token in visible:
            width_px = draw.textbbox((0, 0), token, font=font)[2]
            widths.append(width_px)
            cursor += width_px + size // 3
        cursor -= size // 3 if visible else 0
        x = (width - cursor) // 2
        index_offset = start
        for local_index, token in enumerate(visible):
            token_width = widths[local_index]
            token_positions.append((x, token))
            x += token_width + size // 3
        padding = 16
        if token_positions:
            draw.rounded_rectangle((token_positions[0][0] - padding, line_y - 9, token_positions[-1][0] + widths[-1] + padding, line_y + line_height + 2), radius=10, fill=(10, 18, 24, 195))
        for local_index, (x, token) in enumerate(token_positions):
            is_active = index_offset + local_index == active_word
            color = (255, 202, 73, 255) if is_active else (255, 255, 255, 255)
            draw.text((x, line_y), token, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0, 220))
    bounds = image.getbbox()
    if bounds:
        return image.crop(bounds), bounds[0], bounds[1]
    return image, 0, 0


def make_caption_frames(text: str, width: int, height: int, mode: str, duration: float) -> list[tuple[Image.Image, float, float, int, int]]:
    if mode == "video":
        image, x, y = _caption_image(text, width, height, mode)
        return [(image, 0, duration, x, y)]
    words = text.split()
    if not words:
        return []
    weights = [max(1, len(word.strip(".,!?;:"))) for word in words]
    total = sum(weights)
    frames = []
    cursor = 0.0
    for index, weight in enumerate(weights):
        word_duration = duration * weight / total
        image, x, y = _caption_image(text, width, height, mode, index)
        frames.append((image, cursor, max(0.08, word_duration), x, y))
        cursor += word_duration
    return frames