"""Annotation renderer (§11.8) — draws the problem onto the document.

This is the visual half of the Smart Rejection Explanation: translucent boxes
over the exact OCR word regions that failed a rule, with labels. The geometry
comes from the OCR word boxes, so when a real OCR engine is swapped in the
boxes land on the real image instead of the rendered text page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import get_settings
from app.documents.ocr import CHAR_WIDTH, LINE_HEIGHT, MARGIN_X, MARGIN_Y

SEVERITY_COLOURS = {
    "error": (220, 38, 38),
    "warning": (217, 119, 6),
    "info": (37, 99, 235),
}


def render(document_path: Path, annotations: list[dict[str, Any]],
           out_path: Path) -> Path | None:
    """Render the page with annotation boxes. Returns None if Pillow is absent."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    try:
        text = document_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.splitlines() or [""]
    width = max(700, MARGIN_X * 2 + max((len(ln) for ln in lines), default=40) * CHAR_WIDTH)
    height = max(400, MARGIN_Y * 2 + len(lines) * LINE_HEIGHT)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    font = _load_font(ImageFont, size=15)
    label_font = _load_font(ImageFont, size=12)

    # Page body.
    for i, line in enumerate(lines):
        draw.text((MARGIN_X, MARGIN_Y + i * LINE_HEIGHT), line,
                  fill=(30, 30, 35), font=font)

    # Annotation boxes.
    for annotation in annotations:
        bbox = annotation.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = (int(v) for v in bbox)
        colour = SEVERITY_COLOURS.get(annotation.get("severity", "error"),
                                      SEVERITY_COLOURS["error"])

        draw.rectangle([x - 3, y - 3, x + w + 3, y + h + 3],
                       fill=colour + (48,), outline=colour + (255,), width=2)

        label = str(annotation.get("label", ""))[:60]
        if not label:
            continue
        text_w = int(draw.textlength(label, font=label_font))
        label_y = max(0, y - 20)
        draw.rectangle([x - 3, label_y, x + text_w + 11, label_y + 18],
                       fill=colour + (235,))
        draw.text((x + 2, label_y + 3), label, fill=(255, 255, 255), font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "PNG")
    return out_path


def _load_font(image_font: Any, size: int) -> Any:
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return image_font.truetype(name, size)
        except OSError:
            continue
    return image_font.load_default()


def annotated_path(doc_id: str) -> Path:
    return Path(get_settings().blob_dir) / "annotated" / f"{doc_id}.png"


def ensure_rendered(doc_id: str, storage_key: str,
                    annotations: list[dict[str, Any]]) -> Path | None:
    """Render on demand and cache the result."""
    out = annotated_path(doc_id)
    if out.exists():
        return out
    if not annotations:
        return None
    return render(Path(storage_key), annotations, out)
