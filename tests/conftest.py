"""Shared fixtures.

`RELEASED_CONFIG` is a literal copy of `original/config.json` from
huggingface.co/openai/privacy-filter at revision 7ffa9a04, so tests can check
the released model's contract without the 2.8 GB checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

RELEASED_CONFIG: dict[str, object] = {
    "model_type": "privacy_filter",
    "inference_contract_version": 1,
    "encoding": "o200k_base",
    "num_hidden_layers": 8,
    "num_experts": 128,
    "experts_per_token": 4,
    "vocab_size": 200064,
    "num_labels": 33,
    "hidden_size": 640,
    "intermediate_size": 640,
    "head_dim": 64,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "sliding_window": 257,
    "bidirectional_context": True,
    "bidirectional_left_context": 128,
    "bidirectional_right_context": 128,
    "initial_context_length": 4096,
    "max_position_embeddings": 131072,
    "default_n_ctx": 128000,
    "rope_theta": 150000,
    "rope_scaling_factor": 32.0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 32.0,
    "param_dtype": "bfloat16",
}

# A small model with the same structure as the released one: grouped-query
# attention (6 query heads on 2 KV heads), a narrow band so that short test
# sequences exceed it, several experts with top-2 routing, and YaRN-blended
# RoPE. The vocabulary must still cover every o200k_base token id.
TINY_CONFIG: dict[str, object] = {
    **RELEASED_CONFIG,
    "num_hidden_layers": 2,
    "num_experts": 6,
    "experts_per_token": 2,
    "hidden_size": 16,
    "intermediate_size": 12,
    "head_dim": 8,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "sliding_window": 6,
    "bidirectional_left_context": 3,
    "bidirectional_right_context": 2,
    "initial_context_length": 64,
    "rope_theta": 100,
    "rope_scaling_factor": 4.0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 2.0,
    "default_n_ctx": 64,
}


@pytest.fixture
def released_config() -> dict[str, object]:
    return dict(RELEASED_CONFIG)


@pytest.fixture
def tiny_config() -> dict[str, object]:
    return dict(TINY_CONFIG)


# Where the installed checkpoint is, read once, before any test changes the environment.
_INSTALLED = (
    Path(os.environ["OPF_CHECKPOINT"]).expanduser()
    if os.environ.get("OPF_CHECKPOINT")
    else Path.home() / ".opf" / "privacy_filter"
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Every test gets an empty HOME and no OPF_CHECKPOINT.

    The code under test resolves and downloads checkpoints. If it is wrong, it
    must be wrong inside a temporary directory, never in the real home.
    """
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)


def real_checkpoint_dir() -> Path | None:
    """The installed checkpoint directory, or None. Tests may read it; nothing writes there."""
    if (_INSTALLED / "config.json").is_file() and (_INSTALLED / "model.safetensors").is_file():
        return _INSTALLED
    return None


@pytest.fixture(scope="session")
def checkpoint_dir() -> Path:
    path = real_checkpoint_dir()
    if path is None:
        pytest.skip("real checkpoint not installed (run `opf-mlx download`)")
    return path
