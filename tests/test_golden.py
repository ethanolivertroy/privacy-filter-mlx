"""opf-mlx against outputs recorded from the original PyTorch implementation.

tests/golden/ is written by reference/make_golden.py, which runs the reference
on the CPU and stores what it computes. Nothing here needs PyTorch.

Two models are compared. A tiny one with random weights is part of the
repository, so its tests always run. The released checkpoint is 2.8 GB and is
read from where it is installed; its tests are skipped when it is absent.

How close is close? With float32 weights the two implementations differ by
float32 round-off: a few 1e-5 on logits of up to 28. In bfloat16 every step
rounds to 8 significant bits, so one last-bit difference early on moves many
later roundings. The reference shows the size of that effect on itself: its own
bfloat16 and float32 logits differ by 0.07 to 0.45 (rms) on the recorded texts.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from opf_mlx import PrivacyFilter, PrivacyFilterWarning
from opf_mlx.config import ModelConfig, read_config
from opf_mlx.model import Transformer, rope_tables
from opf_mlx.tokenizer import Tokenizer
from opf_mlx.weights import load_weights

GOLDEN = Path(__file__).parent / "golden"
DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}
DTYPES = ("bfloat16", "float32")

TINY = np.load(GOLDEN / "tiny.npz")
TINY_LENGTHS = (1, 2, 7, 40, 300)

REAL = np.load(GOLDEN / "real.npz")
CASES = {case["name"]: case for case in json.loads((GOLDEN / "real.json").read_text(encoding="ascii"))["cases"]}
SCORED = [name for name in CASES if f"{name}/float32" in REAL.files]  # cases whose logits were recorded
PLAIN = [name for name, case in CASES.items() if not case["options"]]  # cases run with default options

# The reference's name for an option -> ours.
OPTION_NAMES = {"context_window_length": "n_ctx"}


def float32(array: mx.array) -> np.ndarray:
    return np.array(array.astype(mx.float32))


# --------------------------------------------------------------------------
# The tiny model
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(read_config(GOLDEN / "tiny" / "config.json"))


def tiny_model(config: ModelConfig, dtype: str) -> Transformer:
    model = Transformer(config)
    weights = load_weights(GOLDEN / "tiny", config, dtype=None if dtype == config.param_dtype else dtype)
    model.load_weights(list(weights.items()), strict=True)
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", TINY_LENGTHS)
def test_tiny_float32_logits_match_the_reference(tiny_config, device, length):
    with mx.stream(DEVICES[device]):
        logits = float32(tiny_model(tiny_config, "float32")(mx.array(TINY[f"tokens/{length}"][None])))[0]

    # Measured: at most 3.1e-5, on logits up to 8.
    np.testing.assert_allclose(logits, TINY[f"logits/float32/{length}"], rtol=0, atol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "sublayer, given",
    [
        ("block.0.attn", "embedding"),
        ("block.0.mlp", "block.0.attn"),
        ("block.1.attn", "block.0.mlp"),
        ("block.1.mlp", "block.1.attn"),
    ],
)
def test_each_tiny_bfloat16_sublayer_reproduces_the_reference(tiny_config, device, sublayer, given):
    # Each sublayer gets the reference's own input, so nothing accumulates and
    # a rounding step in the wrong place shows up as many differing entries.
    with mx.stream(DEVICES[device]):
        model = tiny_model(tiny_config, "bfloat16")
        block = model.block[int(sublayer.split(".")[1])]
        x = mx.array(TINY[f"stages/{given}"][None]).astype(mx.bfloat16)
        out = block.attn(x, *rope_tables(tiny_config, 40)) if sublayer.endswith("attn") else block.mlp(x)
        out = float32(out)[0]

    # Measured: every one of the 640 entries identical, on both devices. A
    # few may differ where float32 sums are taken in another order.
    assert (out == TINY[f"stages/{sublayer}"]).mean() >= 0.99


@pytest.mark.parametrize("device", DEVICES)
def test_the_tiny_bfloat16_embedding_is_the_stored_one(tiny_config, device):
    with mx.stream(DEVICES[device]):
        embedded = float32(tiny_model(tiny_config, "bfloat16").embedding(mx.array(TINY["tokens/40"][None])))[0]

    assert np.array_equal(embedded, TINY["stages/embedding"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", TINY_LENGTHS)
def test_tiny_bfloat16_logits_stay_within_rounding_noise_of_the_reference(tiny_config, device, length):
    with mx.stream(DEVICES[device]):
        logits = float32(tiny_model(tiny_config, "bfloat16")(mx.array(TINY[f"tokens/{length}"][None])))[0]

    # Measured: identical in 8 of the 10 runs; otherwise off by at most 0.047
    # (one step of bfloat16 at this size is 0.03).
    assert np.abs(logits - TINY[f"logits/bfloat16/{length}"]).max() <= 0.25


def test_tiny_rotary_tables_match_the_reference(tiny_config):
    cos, sin = rope_tables(tiny_config, 300)

    # One unit in the last place of float32, from the cosine and sine routines.
    np.testing.assert_allclose(np.array(cos), TINY["rope/cos"], rtol=0, atol=2.5e-7)
    np.testing.assert_allclose(np.array(sin), TINY["rope/sin"], rtol=0, atol=2.5e-7)


def test_released_rotary_tables_match_the_reference(released_config):
    config = ModelConfig.from_dict(released_config)
    positions = REAL["rope/positions"]
    cos, sin = rope_tables(config, int(positions.max()) + 1)

    # The base frequencies are computed in double precision and rounded once.
    # PyTorch's float32 `pow` is one unit in the last place away from that for
    # 2 of the 32, and an error in a frequency grows with the position:
    # measured 1.2e-7 per position, 1.3e-3 at position 131071.
    allowed = 2.5e-7 * (1 + positions)[:, None]
    assert (np.abs(np.array(cos)[positions] - REAL["rope/cos"]) <= allowed).all()
    assert (np.abs(np.array(sin)[positions] - REAL["rope/sin"]) <= allowed).all()


# --------------------------------------------------------------------------
# The released checkpoint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_the_tokenizer_produces_the_reference_token_ids(released_config, name):
    tokenizer = Tokenizer(str(released_config["encoding"]))

    assert tokenizer.encode(CASES[name]["text"]) == CASES[name]["tokens"]


@pytest.fixture(scope="module", params=[(device, dtype) for device in DEVICES for dtype in DTYPES], ids="-".join)
def released(request, checkpoint_dir) -> PrivacyFilter:
    """The released model on one device in one precision. One is loaded at a time."""
    device, dtype = request.param
    return PrivacyFilter(checkpoint_dir, device=device, dtype=dtype)


@pytest.mark.parametrize("name", SCORED)
def test_released_logits_match_the_reference(released, name):
    with mx.stream(DEVICES[released.device]):
        logits = float32(released.model(mx.array([CASES[name]["tokens"]])))[0]

    difference = np.abs(logits - REAL[f"{name}/{released.dtype}"])
    if released.dtype == "float32":
        # Measured: at most 3.7e-5 on either device, up to 2430 tokens.
        assert difference.max() <= 5e-4
    else:
        # Measured: 0.09 to 0.33 rms on the gpu, 0.09 to 0.19 on the cpu. The
        # reference's own bfloat16 and float32 logits differ by 0.07 to 0.45
        # rms on the same texts.
        assert np.sqrt(np.mean(difference**2)) <= 0.4


@pytest.mark.parametrize("name", CASES)
def test_released_redaction_matches_the_reference(released, checkpoint_dir, name):
    case = CASES[name]
    options = {OPTION_NAMES.get(key, key): value for key, value in case["options"].items()}
    redactor = released
    if options:
        redactor = PrivacyFilter(checkpoint_dir, device=released.device, dtype=released.dtype, **options)
    expected = case["results"][released.dtype]
    # We say so when the input was split into windows or did not survive
    # tokenization unchanged. The reference reports only the second.
    split = "n_ctx" in options
    warns = split or expected["summary"]["decoded_mismatch"]

    if warns:
        with pytest.warns(PrivacyFilterWarning):
            result = redactor.redact(case["text"]).to_dict()
    else:
        result = redactor.redact(case["text"]).to_dict()

    if split:
        assert "3 consecutive windows" in result.pop("warning")
    assert result == expected


def test_a_token_of_the_released_model_is_not_seen_beyond_its_receptive_field(released):
    # 8 layers of attention over 128 tokens on each side: 1024 tokens either way.
    tokens = CASES["long"]["tokens"]  # 2430 tokens
    changed = list(tokens)
    changed[1200] += 1

    with mx.stream(DEVICES[released.device]):
        before, after = (float32(released.model(mx.array([ids])))[0] for ids in (tokens, changed))
    moved = np.abs(after - before).max(axis=1) > 0

    # Not one bit moves outside 176..2224: attention is masked beyond the band
    # and the experts take one token at a time. Measured inside: with float32
    # weights every token from 176 to 2224 moves (on the cpu all but token
    # 176); bfloat16 rounds the faintest effects away at the edges (29 tokens
    # on the gpu, 49 on the cpu).
    assert not moved[:176].any() and not moved[2225:].any()
    assert moved[400:2000].all()
    assert moved[176:400].mean() >= 0.75 and moved[2000:2225].mean() >= 0.75
    if released.dtype == "float32":
        assert moved[177:2225].all()


def test_precision_does_not_change_any_recorded_result():
    # The cases were chosen to be decided by the model, not by rounding: the
    # reference itself gives the same answer in both precisions.
    for name, case in CASES.items():
        assert case["results"]["bfloat16"] == case["results"]["float32"], name
