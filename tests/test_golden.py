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
SCORED = [name for name in CASES if f"{name}/float32" in REAL.files]
PLAIN = [name for name, case in CASES.items() if not case["options"]]

OPTION_NAMES = {"context_window_length": "n_ctx"}


def float32(array: mx.array) -> np.ndarray:
    return np.array(array.astype(mx.float32))


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
    with mx.stream(DEVICES[device]):
        model = tiny_model(tiny_config, "bfloat16")
        block = model.block[int(sublayer.split(".")[1])]
        x = mx.array(TINY[f"stages/{given}"][None]).astype(mx.bfloat16)
        out = block.attn(x, *rope_tables(tiny_config, 40)) if sublayer.endswith("attn") else block.mlp(x)
        out = float32(out)[0]

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

    assert np.abs(logits - TINY[f"logits/bfloat16/{length}"]).max() <= 0.25


def test_tiny_rotary_tables_match_the_reference(tiny_config):
    cos, sin = rope_tables(tiny_config, 300)

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


@pytest.mark.parametrize("name", CASES)
def test_the_tokenizer_produces_the_reference_token_ids(released_config, name):
    tokenizer = Tokenizer(str(released_config["encoding"]))

    assert tokenizer.encode(CASES[name]["text"]) == CASES[name]["tokens"]


@pytest.fixture(scope="module", params=[(device, dtype) for device in DEVICES for dtype in DTYPES], ids="-".join)
def released(request, checkpoint_dir) -> PrivacyFilter:
    device, dtype = request.param
    return PrivacyFilter(checkpoint_dir, device=device, dtype=dtype)


@pytest.mark.parametrize("name", SCORED)
def test_released_logits_match_the_reference(released, name):
    with mx.stream(DEVICES[released.device]):
        logits = float32(released.model(mx.array([CASES[name]["tokens"]])))[0]

    difference = np.abs(logits - REAL[f"{name}/{released.dtype}"])
    if released.dtype == "float32":
        assert difference.max() <= 5e-4
    else:
        assert np.sqrt(np.mean(difference**2)) <= 0.4


@pytest.mark.parametrize("name", CASES)
def test_released_redaction_matches_the_reference(released, checkpoint_dir, name):
    case = CASES[name]
    options = {OPTION_NAMES.get(key, key): value for key, value in case["options"].items()}
    redactor = released
    if options:
        redactor = PrivacyFilter(checkpoint_dir, device=released.device, dtype=released.dtype, **options)
    expected = case["results"][released.dtype]
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
    tokens = CASES["long"]["tokens"]
    changed = list(tokens)
    changed[1200] += 1

    with mx.stream(DEVICES[released.device]):
        before, after = (float32(released.model(mx.array([ids])))[0] for ids in (tokens, changed))
    moved = np.abs(after - before).max(axis=1) > 0

    assert not moved[:176].any() and not moved[2225:].any()
    assert moved[400:2000].all()
    assert moved[176:400].mean() >= 0.75 and moved[2000:2225].mean() >= 0.75
    if released.dtype == "float32":
        assert moved[177:2225].all()


def test_precision_does_not_change_any_recorded_result():
    for name, case in CASES.items():
        assert case["results"]["bfloat16"] == case["results"]["float32"], name
