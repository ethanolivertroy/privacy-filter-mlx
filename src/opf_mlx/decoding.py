"""Turning per-token log-probabilities into one label per token.

The default decoder is a Viterbi search over the BIOES grammar: it maximizes

    start[y_0] + sum_t logp[t, y_t] + sum_t transition[y_(t-1), y_t] + end[y_last]

where disallowed starts, transitions and ends score -1e9. Six optional biases,
read from `viterbi_calibration.json`, shift the allowed transitions. This is a
NumPy port of `ViterbiCRFDecoder` in the reference (`opf/_core/decoding.py`),
in float32 and with the same operation order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .labels import LabelSpace

NEG_INF = -1e9  # the reference's "impossible" score; finite on purpose

BIAS_KEYS: tuple[str, ...] = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)
CALIBRATION_FILENAME = "viterbi_calibration.json"


def argmax_decode(logp: np.ndarray) -> list[int]:
    """Independent per-token argmax. May produce ill-formed label sequences."""
    return np.argmax(logp, axis=1).tolist()


class ViterbiDecoder:
    """Best label sequence that obeys the BIOES grammar."""

    def __init__(self, space: LabelSpace, biases: Mapping[str, float] | None = None) -> None:
        biases = dict.fromkeys(BIAS_KEYS, 0.0) if biases is None else dict(biases)
        if set(biases) != set(BIAS_KEYS):
            raise ValueError(f"biases must contain exactly {sorted(BIAS_KEYS)}")

        n = len(space.class_names)
        self.num_classes = n
        self.start = np.full((n,), NEG_INF, dtype=np.float32)
        self.end = np.full((n,), NEG_INF, dtype=np.float32)
        self.transition = np.full((n, n), NEG_INF, dtype=np.float32)

        starts_span = [tag in ("B", "S") for tag in space.tag_of]
        background = [tag is None for tag in space.tag_of]

        for prev in range(n):
            prev_tag = space.tag_of[prev]
            if starts_span[prev] or background[prev]:
                self.start[prev] = 0.0
            if prev_tag in ("E", "S") or background[prev]:
                self.end[prev] = 0.0

            for nxt in range(n):
                next_tag = space.tag_of[nxt]
                same_category = space.span_of[prev] == space.span_of[nxt]
                if prev_tag in ("B", "I"):  # inside a span: continue it or end it
                    if same_category and next_tag == "I":
                        self.transition[prev, nxt] = biases["transition_bias_inside_to_continue"]
                    elif same_category and next_tag == "E":
                        self.transition[prev, nxt] = biases["transition_bias_inside_to_end"]
                elif background[prev]:  # outside any span
                    if background[nxt]:
                        self.transition[prev, nxt] = biases["transition_bias_background_stay"]
                    elif starts_span[nxt]:
                        self.transition[prev, nxt] = biases["transition_bias_background_to_start"]
                else:  # just finished a span ("E" or "S")
                    if background[nxt]:
                        self.transition[prev, nxt] = biases["transition_bias_end_to_background"]
                    elif starts_span[nxt]:
                        self.transition[prev, nxt] = biases["transition_bias_end_to_start"]

    def decode(self, logp: np.ndarray) -> list[int]:
        """Decode `[seq_len, num_classes]` log-probabilities into class ids."""
        logp = np.asarray(logp, dtype=np.float32)
        if logp.ndim != 2 or logp.shape[1] != self.num_classes:
            raise ValueError(
                f"token log-probabilities must have shape [seq_len, {self.num_classes}], got {tuple(logp.shape)}"
            )
        seq_len = logp.shape[0]
        if seq_len == 0:
            return []

        columns = np.arange(self.num_classes)
        scores = logp[0] + self.start
        backpointers = np.empty((seq_len - 1, self.num_classes), dtype=np.int64)
        for t in range(1, seq_len):
            candidates = scores[:, None] + self.transition  # [previous, next]
            best_previous = candidates.argmax(axis=0)  # ties go to the lowest index
            backpointers[t - 1] = best_previous
            scores = candidates[best_previous, columns] + logp[t]

        if not np.isfinite(scores).any():
            return argmax_decode(logp)

        label = int((scores + self.end).argmax())
        path = [label]
        for t in range(seq_len - 2, -1, -1):
            label = int(backpointers[t, label])
            path.append(label)
        path.reverse()
        return path


def load_transition_biases(
    *,
    calibration_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, float]:
    """Resolve the six Viterbi biases.

    An explicit path wins. Otherwise `viterbi_calibration.json` next to the
    checkpoint is used if it exists, and all-zero biases if it does not.
    """
    path: Path | None = None
    if calibration_path is not None and str(calibration_path).strip():
        path = Path(str(calibration_path).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"viterbi calibration file not found: {path}")
    elif checkpoint_dir is not None:
        candidate = Path(checkpoint_dir) / CALIBRATION_FILENAME
        if candidate.is_file():
            path = candidate
    if path is None:
        return dict.fromkeys(BIAS_KEYS, 0.0)

    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    points = _exact_keys(artifact, {"operating_points"}, f"calibration file {path}")["operating_points"]
    default = _exact_keys(points, {"default"}, "operating_points")["default"]
    raw = _exact_keys(default, {"biases"}, "operating_points.default")["biases"]
    raw = _exact_keys(raw, set(BIAS_KEYS), "operating_points.default.biases")

    biases: dict[str, float] = {}
    for key in BIAS_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"operating_points.default.biases.{key} must be numeric (got {value!r})")
        biases[key] = float(value)
    return biases


def _exact_keys(value: object, expected: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != expected:
        missing, extra = sorted(expected - set(value)), sorted(set(value) - expected)
        raise ValueError(f"{name} must contain exactly {sorted(expected)} (missing={missing}, extra={extra})")
    return value
