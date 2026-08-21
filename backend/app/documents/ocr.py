"""OCR behind an interface (§19 "every external dependency behind an interface").

MVP ships a text adapter: the synthetic corpus renders documents as text files,
so the *rest* of the pipeline (classification, extraction, rules, rejection,
annotation geometry) is exercised for real. Tesseract and Azure Document
Intelligence adapters slot in behind the same ``OCRAdapter`` protocol without
any caller changing — see TO_BE_DONE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# Rendering geometry used to synthesise word boxes from a text layout, so
# downstream annotation has real coordinates to draw against.
CHAR_WIDTH = 9
LINE_HEIGHT = 22
MARGIN_X = 40
MARGIN_Y = 40


@dataclass
class Word:
    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    line: int


@dataclass
class OCRResult:
    text: str
    words: list[Word] = field(default_factory=list)
    quality: float = 0.0
    page_count: int = 1
    engine: str = "unknown"
    error: str | None = None

    def find_word_boxes(self, needle: str) -> list[tuple[int, int, int, int]]:
        """Bounding boxes for every word matching ``needle`` (for annotation)."""
        target = (needle or "").strip().lower().strip(".,:;")
        if not target:
            return []
        return [
            word.bbox for word in self.words
            if target in word.text.lower() or word.text.lower() in target
        ]

    def line_bbox(self, line_no: int) -> tuple[int, int, int, int] | None:
        words = [w for w in self.words if w.line == line_no]
        if not words:
            return None
        x = min(w.bbox[0] for w in words)
        y = min(w.bbox[1] for w in words)
        right = max(w.bbox[0] + w.bbox[2] for w in words)
        bottom = max(w.bbox[1] + w.bbox[3] for w in words)
        return (x, y, right - x, bottom - y)


class OCRAdapter(Protocol):
    name: str

    def read(self, path: Path) -> OCRResult: ...


class TextOCRAdapter:
    """Reads text-rendered documents and synthesises word geometry."""

    name = "text-adapter"

    SUPPORTED = {".txt", ".md", ".text", ".csv", ".json"}

    def read(self, path: Path) -> OCRResult:
        if path.suffix.lower() not in self.SUPPORTED:
            return OCRResult(
                text="", quality=0.0, engine=self.name,
                error=f"No OCR engine available for '{path.suffix}' files.",
            )
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return OCRResult(text="", quality=0.0, engine=self.name, error=str(exc))

        if not raw.strip():
            return OCRResult(text="", quality=0.0, engine=self.name,
                             error="File is empty or unreadable.")

        words: list[Word] = []
        for line_no, line in enumerate(raw.splitlines()):
            x = MARGIN_X
            for token in line.split(" "):
                if token:
                    words.append(Word(
                        text=token,
                        bbox=(x, MARGIN_Y + line_no * LINE_HEIGHT,
                              len(token) * CHAR_WIDTH, LINE_HEIGHT - 4),
                        confidence=_token_confidence(token),
                        line=line_no,
                    ))
                x += (len(token) + 1) * CHAR_WIDTH

        quality = _weighted_quality(words)
        return OCRResult(
            text=raw,
            words=words,
            quality=quality,
            page_count=max(1, raw.count("\f") + 1),
            engine=self.name,
        )


def _token_confidence(token: str) -> float:
    """Simulated per-word confidence: gibberish scores low, as OCR would."""
    if not token:
        return 0.0
    letters = sum(ch.isalnum() for ch in token)
    ratio = letters / len(token)
    if token.count("~") or token.count("?") > 1:
        return 0.25
    return round(min(0.99, 0.55 + 0.45 * ratio), 3)


def _weighted_quality(words: list[Word]) -> float:
    """Mean word confidence weighted by word length (§11.2)."""
    if not words:
        return 0.0
    total_weight = sum(len(w.text) for w in words)
    if not total_weight:
        return 0.0
    return round(sum(w.confidence * len(w.text) for w in words) / total_weight, 3)


def region_quality_map(result: OCRResult, rows: int = 4, cols: int = 3) -> list[dict]:
    """Grid quality map so ILLEGIBLE rejections can point at *where* (§11.2)."""
    if not result.words:
        return []
    max_x = max(w.bbox[0] + w.bbox[2] for w in result.words) or 1
    max_y = max(w.bbox[1] + w.bbox[3] for w in result.words) or 1

    cells: list[dict] = []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = col * max_x / cols, (col + 1) * max_x / cols
            y0, y1 = row * max_y / rows, (row + 1) * max_y / rows
            inside = [
                w for w in result.words
                if x0 <= w.bbox[0] < x1 and y0 <= w.bbox[1] < y1
            ]
            if inside:
                cells.append({
                    "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    "quality": _weighted_quality(inside),
                    "words": len(inside),
                })
    return cells


_adapter: OCRAdapter = TextOCRAdapter()


def get_adapter() -> OCRAdapter:
    return _adapter


def set_adapter(adapter: OCRAdapter) -> None:
    """Swap in Tesseract/Azure DI without touching the pipeline."""
    global _adapter
    _adapter = adapter
