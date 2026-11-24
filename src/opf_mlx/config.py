"""Checkpoint configuration: read `config.json` and enforce the model contract.

Nothing here touches MLX. The checks mirror the reference's
`_validate_encoder_artifact_contract` and add refusals for the reference
features this port does not implement, so an unsupported checkpoint fails here
instead of producing wrong numbers later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

MODEL_TYPE = "privacy_filter"
SUPPORTED_CONTRACT_VERSION = 1

REQUIRED_KEYS: tuple[str, ...] = (
    "model_type",
    "encoding",
    "num_hidden_layers",
    "num_experts",
    "experts_per_token",
    "vocab_size",
    "num_labels",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "bidirectional_context",
    "bidirectional_left_context",
    "bidirectional_right_context",
    "initial_context_length",
    "rope_theta",
    "rope_scaling_factor",
    "rope_ntk_alpha",
    "rope_ntk_beta",
    "param_dtype",
)

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}


class ConfigError(ValueError):
    """`config.json` is malformed or describes a model this port does not implement."""


@dataclass(frozen=True)
class ModelConfig:
    """Everything the model, tokenizer and pipeline need from `config.json`."""

    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_labels: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    left_context: int
    right_context: int
    num_experts: int
    experts_per_token: int
    intermediate_size: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float
    initial_context_length: int
    default_n_ctx: int
    encoding: str
    param_dtype: str
    # Not stored in the released config; these are the reference's code defaults.
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    norm_eps: float = 1e-5

    @property
    def queries_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def qkv_size(self) -> int:
        return self.head_dim * (self.num_attention_heads + 2 * self.num_key_value_heads)

    @property
    def receptive_field(self) -> tuple[int, int]:
        """How far left and right a token's logits can see through all layers."""
        return (self.num_hidden_layers * self.left_context, self.num_hidden_layers * self.right_context)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(read_config(path))

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ModelConfig":
        errors: list[str] = []

        missing = sorted(key for key in REQUIRED_KEYS if key not in raw)
        if missing:
            errors.append("missing config keys: " + ", ".join(missing))

        if "model_type" in raw and raw["model_type"] != MODEL_TYPE:
            errors.append(f"model_type must be {MODEL_TYPE!r} (got {raw['model_type']!r})")

        version = raw.get("inference_contract_version", SUPPORTED_CONTRACT_VERSION)
        if version != SUPPORTED_CONTRACT_VERSION or isinstance(version, bool):
            errors.append(
                f"inference_contract_version must be {SUPPORTED_CONTRACT_VERSION} (got {version!r}); "
                "this port implements only that contract"
            )

        encoding = raw.get("encoding")
        if "encoding" in raw and (not isinstance(encoding, str) or not encoding):
            errors.append(f"encoding must be a non-empty string (got {encoding!r})")

        if "bidirectional_context" in raw and raw["bidirectional_context"] is not True:
            errors.append(
                "bidirectional_context must be true; only bidirectional checkpoints are supported "
                f"(got {raw['bidirectional_context']!r})"
            )

        if raw.get("packed_geglu") not in (None, False):
            errors.append("packed_geglu is set; only the contiguous SwiGLU layout is implemented")

        num_layers = _int(raw, "num_hidden_layers", errors)
        hidden = _int(raw, "hidden_size", errors)
        vocab = _int(raw, "vocab_size", errors)
        num_labels = _int(raw, "num_labels", errors)
        head_dim = _int(raw, "head_dim", errors)
        heads = _int(raw, "num_attention_heads", errors)
        kv_heads = _int(raw, "num_key_value_heads", errors)
        left = _int(raw, "bidirectional_left_context", errors, minimum=0)
        right = _int(raw, "bidirectional_right_context", errors, minimum=0)
        window = _int(raw, "sliding_window", errors, minimum=0)
        experts = _int(raw, "num_experts", errors)
        per_token = _int(raw, "experts_per_token", errors)
        intermediate = _int(raw, "intermediate_size", errors)
        context = _int(raw, "initial_context_length", errors)
        theta = _number(raw, "rope_theta", errors)
        scaling = _number(raw, "rope_scaling_factor", errors)
        alpha = _number(raw, "rope_ntk_alpha", errors)
        beta = _number(raw, "rope_ntk_beta", errors)
        limit = _number(raw, "swiglu_limit", errors) if "swiglu_limit" in raw else 7.0

        if None not in (left, right, window) and window != left + right + 1:
            errors.append(
                "sliding_window must equal bidirectional_left_context + bidirectional_right_context + 1 "
                f"(got {window}, expected {left + right + 1})"
            )
        if None not in (heads, kv_heads) and heads % kv_heads != 0:
            errors.append(f"num_attention_heads ({heads}) must be a multiple of num_key_value_heads ({kv_heads})")
        if head_dim is not None and head_dim % 2 != 0:
            errors.append(f"head_dim must be even for rotary embeddings (got {head_dim})")
        if None not in (experts, per_token) and per_token > experts:
            errors.append(f"experts_per_token ({per_token}) must not exceed num_experts ({experts})")

        if "default_n_ctx" in raw:
            default_n_ctx = _int(raw, "default_n_ctx", errors)
        else:
            default_n_ctx = context

        dtype = raw.get("param_dtype")
        param_dtype = _DTYPE_ALIASES.get(dtype.lower()) if isinstance(dtype, str) else None
        if "param_dtype" in raw and param_dtype is None:
            errors.append(f"param_dtype must be one of {sorted(_DTYPE_ALIASES)} (got {dtype!r})")

        if None not in (head_dim, context, theta, scaling, alpha, beta) and scaling > 1.0 and theta > 1.0:
            low, high = rope_blend_range(head_dim, context, theta, alpha, beta)
            if not 0 < low < high < head_dim / 2 - 1:
                errors.append(
                    "rope parameters put the YaRN blend region outside the frequency range "
                    f"(need 0 < low < high < {head_dim / 2 - 1:g}, got low={low:.3f}, high={high:.3f})"
                )

        if errors:
            raise ConfigError("Incompatible checkpoint configuration: " + "; ".join(errors))

        return cls(
            num_hidden_layers=num_layers,
            hidden_size=hidden,
            vocab_size=vocab,
            num_labels=num_labels,
            head_dim=head_dim,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            left_context=left,
            right_context=right,
            num_experts=experts,
            experts_per_token=per_token,
            intermediate_size=intermediate,
            rope_theta=float(theta),
            rope_scaling_factor=float(scaling),
            rope_ntk_alpha=float(alpha),
            rope_ntk_beta=float(beta),
            initial_context_length=context,
            default_n_ctx=default_n_ctx,
            encoding=encoding,
            param_dtype=param_dtype,
            swiglu_limit=float(limit),
        )


def rope_blend_range(head_dim: int, context: int, theta: float, alpha: float, beta: float) -> tuple[float, float]:
    """Frequency-pair indices between which YaRN blends from unscaled to scaled."""
    half = head_dim / 2
    low = half * math.log(context / (beta * 2 * math.pi)) / math.log(theta)
    high = half * math.log(context / (alpha * 2 * math.pi)) / math.log(theta)
    return low, high


def read_config(path: str | Path) -> dict[str, object]:
    """Load `config.json` as a dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"Checkpoint config at {path} must contain a JSON object")
    return payload


def _int(raw: Mapping[str, object], key: str, errors: list[str], *, minimum: int = 1) -> int | None:
    if key not in raw:
        return None
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{key} must be an integer (got {value!r})")
        return None
    if value < minimum:
        errors.append(f"{key} must be >= {minimum} (got {value})")
        return None
    return value


def _number(raw: Mapping[str, object], key: str, errors: list[str]) -> float | None:
    if key not in raw:
        return None
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        errors.append(f"{key} must be a positive number (got {value!r})")
        return None
    return float(value)
