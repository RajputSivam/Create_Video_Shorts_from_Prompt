from pathlib import Path


def write_text_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)