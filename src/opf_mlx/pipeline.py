"""Text in, redaction out: the one place where the parts meet.

    text -> Tokenizer -> token ids -> plan_windows -> Transformer (MLX) -> logits
         -> log-softmax (NumPy) -> Viterbi or argmax -> token spans
         -> character spans -> placeholders

This is a port of the reference's `load_inference_runtime` and `predict_text`
(`opf/_core/runtime.py`) and of `OPF.redact` (`opf/_api.py`). The model only
ever sees token ids and only ever returns logits; everything after that is
NumPy and plain Python.

Nothing here degrades quietly. A checkpoint that does not fit the architecture,
a device whose float32 products are inexact and a model too large for memory
are errors. An input longer than the context window is split, never truncated,
and the split is reported in the result and as a `PrivacyFilterWarning`.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from . import env, precision
from .checkpoint import resolve_checkpoint
from .config import ConfigError, ModelConfig, read_config
from .decoding import ViterbiDecoder, argmax_decode, load_transition_biases
from .labels import resolve_label_space
from .model import Transformer
from .spans import (
    DetectedSpan,
    collapse_labels,
    discard_overlapping_by_label,
    labels_to_spans,
    placeholder_for,
    redact_text,
    select_non_overlapping,
    token_spans_to_char_spans,
    trim_whitespace,
)
from .tokenizer import Tokenizer
from .weights import load_weights, weights_nbytes
from .windows import POLICIES, plan_windows, require_room_for_halo

SCHEMA_VERSION = 1
OUTPUT_MODES: tuple[str, ...] = ("typed", "redacted")
DECODE_MODES: tuple[str, ...] = ("viterbi", "argmax")
DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}
DTYPES: tuple[str, ...] = ("bfloat16", "float32")

CPU_N_CTX = 4096

MISMATCH_WARNING = (
    "Input text did not exactly match tokenizer round-trip decode; spans are based on decoded token text."
)


class PrivacyFilterWarning(UserWarning):
    """Something the caller should know about that is not an error."""


@dataclass(frozen=True)
class RedactionResult:
    """What `PrivacyFilter.redact` returns; the reference's output schema, version 1."""

    schema_version: int
    summary: dict[str, object]
    text: str
    detected_spans: tuple[DetectedSpan, ...]
    redacted_text: str
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "summary": dict(self.summary),
            "text": self.text,
            "detected_spans": [
                {
                    "label": span.label,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "placeholder": span.placeholder,
                }
                for span in self.detected_spans
            ],
            "redacted_text": self.redacted_text,
        }
        if self.warning is not None:
            payload["warning"] = self.warning
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class PrivacyFilter:
    """A loaded Privacy Filter model that redacts text.

    checkpoint: directory with `config.json` and the weights. Default: the
        `OPF_CHECKPOINT` environment variable, else `~/.opf/privacy_filter`,
        downloaded on first use unless `download=False`.
    n_ctx: tokens per forward pass. Default: the checkpoint's `default_n_ctx`
        on the gpu and 4096 on the cpu, as in the reference.
    device: "gpu" or "cpu".
    dtype: "bfloat16" or "float32". Default: as stored in the checkpoint.
    decode_mode: "viterbi" (constrained to well-formed spans) or "argmax".
    viterbi_calibration_path: transition biases. Default:
        `viterbi_calibration.json` in the checkpoint directory if present.
    trim_whitespace: strip whitespace from the ends of detected spans.
    discard_overlapping_predicted_spans: drop spans overlapping an earlier
        span of the same label.
    output_mode: "typed" keeps the categories; "redacted" collapses them.
    window_policy: how an input longer than `n_ctx` is split. "original" is the
        reference's consecutive windows; "halo" gives every kept token `halo`
        tokens of context (default: the model's receptive field).
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        n_ctx: int | None = None,
        device: str = "gpu",
        dtype: str | None = None,
        decode_mode: str = "viterbi",
        viterbi_calibration_path: str | os.PathLike[str] | None = None,
        trim_whitespace: bool = True,
        discard_overlapping_predicted_spans: bool = False,
        output_mode: str = "typed",
        window_policy: str = "original",
        halo: int | None = None,
        download: bool = True,
    ) -> None:
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}; expected one of {OUTPUT_MODES}")
        if decode_mode not in DECODE_MODES:
            raise ValueError(f"Unsupported decode_mode: {decode_mode!r}; expected one of {DECODE_MODES}")
        if device not in DEVICES:
            raise ValueError(f"Unsupported device: {device!r}; expected one of {tuple(DEVICES)}")
        if dtype is not None and dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype!r}; expected one of {DTYPES}")
        if window_policy not in POLICIES:
            raise ValueError(f"unknown window policy {window_policy!r}; expected one of {POLICIES}")
        if n_ctx is not None and n_ctx <= 0:
            raise ValueError(f"n_ctx must be positive (got {n_ctx})")
        if halo is not None and halo < 0:
            raise ValueError(f"halo must not be negative (got {halo})")

        self.checkpoint: Path = resolve_checkpoint(checkpoint, download=download)
        raw_config = read_config(self.checkpoint / "config.json")
        self.config = ModelConfig.from_dict(raw_config)
        self.labels = resolve_label_space(raw_config)
        self.tokenizer = Tokenizer(self.config.encoding)
        if self.config.vocab_size <= self.tokenizer.max_token_id:
            raise ConfigError(
                f"vocab_size {self.config.vocab_size} is too small for the {self.config.encoding} tokenizer, "
                f"whose largest token id is {self.tokenizer.max_token_id}"
            )

        self.device = device
        self.dtype = dtype or self.config.param_dtype
        self.n_ctx = n_ctx if n_ctx is not None else (self.config.default_n_ctx if device == "gpu" else CPU_N_CTX)
        self.decode_mode = decode_mode
        self.trim_whitespace = bool(trim_whitespace)
        self.discard_overlapping_predicted_spans = bool(discard_overlapping_predicted_spans)
        self.output_mode = output_mode
        self.window_policy = window_policy
        self.halo = max(self.config.receptive_field) if halo is None else halo
        if window_policy == "halo":
            require_room_for_halo(self.n_ctx, self.halo)

        note = env.check_memory(
            required=weights_nbytes(self.config, dtype),
            limit=env.gpu_memory_limit() if device == "gpu" else env.total_memory(),
            available=env.available_memory(),
        )
        if note is not None:
            warnings.warn(note, PrivacyFilterWarning, stacklevel=2)

        with mx.stream(DEVICES[device]):
            precision.require_full_precision()
            self.model = Transformer(self.config)
            self.model.load_weights(list(load_weights(self.checkpoint, self.config, dtype=dtype).items()), strict=True)
            mx.eval(self.model.parameters())

        self._decoder = (
            ViterbiDecoder(
                self.labels,
                load_transition_biases(calibration_path=viterbi_calibration_path, checkpoint_dir=self.checkpoint),
            )
            if decode_mode == "viterbi"
            else None
        )

    def redact(self, text: str) -> RedactionResult:
        """Detect private spans in `text` and replace them with placeholders."""
        token_ids = self.tokenizer.encode(text)
        scores, notes = self._score(token_ids)
        labels = self._decoder.decode(scores) if self._decoder is not None else argmax_decode(scores)

        decoded, char_starts, char_ends = self.tokenizer.decode_with_offsets(token_ids)
        decoded_mismatch = decoded != text
        source = decoded if decoded_mismatch else text
        if decoded_mismatch:
            notes.append(MISMATCH_WARNING)

        spans = token_spans_to_char_spans(labels_to_spans(labels, self.labels), char_starts, char_ends)
        if self.trim_whitespace:
            spans = trim_whitespace(spans, source)
        if self.discard_overlapping_predicted_spans:
            spans = discard_overlapping_by_label(spans)
        detected: list[DetectedSpan] = []
        for span_class, start, end in spans:
            if 0 <= start < end <= len(source):
                name = self.labels.span_names[span_class]
                detected.append(DetectedSpan(name, start, end, source[start:end], placeholder_for(name)))
        return self._result(source, select_non_overlapping(detected), decoded_mismatch=decoded_mismatch, notes=notes)

    def log_probs(self, token_ids: Sequence[int]) -> np.ndarray:
        """float32 log-probabilities `[tokens, classes]` for a token sequence, windowed as configured."""
        scores, notes = self._score([int(token) for token in token_ids])
        for note in notes:
            warnings.warn(note, PrivacyFilterWarning, stacklevel=2)
        return scores

    def _score(self, token_ids: list[int]) -> tuple[np.ndarray, list[str]]:
        if not token_ids:
            return np.zeros((0, self.config.num_labels), dtype=np.float32), []
        windows = plan_windows(len(token_ids), self.n_ctx, policy=self.window_policy, halo=self.halo)
        parts: list[np.ndarray] = []
        with mx.stream(DEVICES[self.device]):
            for window in windows:
                logits = self.model(mx.array([token_ids[window.start : window.stop]]))[0]
                kept = logits[window.keep_start - window.start : window.keep_stop - window.start]
                parts.append(log_softmax(np.array(kept.astype(mx.float32))))
        return np.concatenate(parts, axis=0), self._window_notes(len(token_ids), len(windows))

    def _window_notes(self, tokens: int, windows: int) -> list[str]:
        if windows == 1:
            return []
        if self.window_policy == "halo" and self.halo >= max(self.config.receptive_field):
            return []
        if self.window_policy == "halo":
            how = f"{windows} windows with a halo of {self.halo} tokens, less than the model's reach"
        else:
            how = f"{windows} consecutive windows"
        note = (
            f"Input has {tokens} tokens but the context window holds {self.n_ctx} (n_ctx), so it was processed in "
            f"{how}. Nothing was dropped, but tokens near a window boundary see less context than they would in "
            "one pass, which can change predictions there."
        )
        return [note]

    def _result(
        self, text: str, spans: Sequence[DetectedSpan], *, decoded_mismatch: bool, notes: list[str]
    ) -> RedactionResult:
        if self.output_mode == "redacted":
            spans = collapse_labels(spans)
        by_label: dict[str, int] = {}
        for span in spans:
            by_label[span.label] = by_label.get(span.label, 0) + 1
        for note in notes:
            warnings.warn(note, PrivacyFilterWarning, stacklevel=3)
        return RedactionResult(
            schema_version=SCHEMA_VERSION,
            summary={
                "output_mode": self.output_mode,
                "span_count": len(spans),
                "by_label": dict(sorted(by_label.items())),
                "decoded_mismatch": decoded_mismatch,
            },
            text=text,
            detected_spans=tuple(spans),
            redacted_text=redact_text(text, spans),
            warning=" ".join(notes) or None,
        )


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis, in the precision of the input."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
