"""From token labels to character spans to redacted text.

Pure Python, no model and no tokenizer. Each function is a port of the
reference's behavior (`opf/_core/spans.py`, `opf/_core/runtime.py`,
`opf/_api.py`), simplified only where the reference handles cases that cannot
occur here (gaps in token indices).

Span tuples are `(span_class, start, end)` with `end` exclusive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .labels import LabelSpace

REDACTED_LABEL = "redacted"
REDACTED_PLACEHOLDER = "<REDACTED>"

Span = tuple[int, int, int]


@dataclass(frozen=True)
class DetectedSpan:
    """One detected character span, ready to print or serialize."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str


def labels_to_spans(labels: Sequence[int], space: LabelSpace) -> list[Span]:
    """Group per-token class ids into token spans.

    Viterbi output is always well formed, but argmax output need not be, so
    this accepts anything: an `I` or `E` with no open span of its category
    starts one, and anything that interrupts an open span closes it.
    """
    spans: list[Span] = []
    open_class: int | None = None
    open_start = 0

    def close(end: int) -> None:
        nonlocal open_class
        if open_class is not None:
            spans.append((open_class, open_start, end))
        open_class = None

    for index, label in enumerate(labels):
        span_class, tag = space.span_of[label], space.tag_of[label]
        if tag is None:  # background
            close(index)
        elif tag == "S":
            close(index)
            spans.append((span_class, index, index + 1))
        elif tag == "B":
            close(index)
            open_class, open_start = span_class, index
        elif tag == "I":
            if open_class != span_class:
                close(index)
                open_class, open_start = span_class, index
        elif open_class == span_class:  # "E" ending the open span
            close(index + 1)
        else:  # "E" with nothing matching to end
            close(index)
            spans.append((span_class, index, index + 1))

    close(len(labels))
    return spans


def token_spans_to_char_spans(spans: Sequence[Span], char_starts: Sequence[int], char_ends: Sequence[int]) -> list[Span]:
    """Convert token-index spans to character-index spans, dropping empty ones."""
    converted: list[Span] = []
    for span_class, token_start, token_end in spans:
        if not 0 <= token_start < token_end <= len(char_starts):
            continue
        start, end = char_starts[token_start], char_ends[token_end - 1]
        if end > start:
            converted.append((span_class, start, end))
    return converted


def trim_whitespace(spans: Sequence[Span], text: str) -> list[Span]:
    """Strip leading and trailing whitespace from each span; drop blank spans."""
    trimmed: list[Span] = []
    for span_class, start, end in spans:
        if not 0 <= start < end <= len(text):
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((span_class, start, end))
    return trimmed


def discard_overlapping_by_label(spans: Sequence[Span]) -> list[Span]:
    """Within each span class, keep the earliest (then longest) of overlapping spans."""
    by_class: dict[int, list[tuple[int, int]]] = {}
    for span_class, start, end in spans:
        by_class.setdefault(span_class, []).append((start, end))

    kept: list[Span] = []
    for span_class, ranges in by_class.items():
        accepted: list[tuple[int, int]] = []
        for start, end in sorted(ranges, key=lambda r: (r[0], -(r[1] - r[0]))):
            if all(end <= a_start or start >= a_end for a_start, a_end in accepted):
                accepted.append((start, end))
        kept.extend((span_class, start, end) for start, end in accepted)

    kept.sort(key=lambda span: (span[1], span[2], span[0]))
    return kept


def select_non_overlapping(spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
    """Keep a left-to-right non-overlapping subset: earlier first, then longer, then by label."""
    kept: list[DetectedSpan] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), s.label)):
        if span.start < cursor or span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def placeholder_for(label: str) -> str:
    """`private_person` -> `<PRIVATE_PERSON>`."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def collapse_labels(spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
    """The `redacted` output mode: hide which category each span is."""
    return [DetectedSpan(REDACTED_LABEL, s.start, s.end, s.text, REDACTED_PLACEHOLDER) for s in spans]


def redact_text(text: str, spans: Sequence[DetectedSpan]) -> str:
    """Replace each span with its placeholder. Spans must be sorted and non-overlapping."""
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)
