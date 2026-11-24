from __future__ import annotations

import json
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opf")

from opf import OPF  # noqa: E402
from opf._core.decoding import ViterbiCRFDecoder  # noqa: E402
from opf._core.sequence_labeling import build_label_info  # noqa: E402
from opf._model.model import Transformer as ReferenceTransformer  # noqa: E402

from conftest import TINY_CONFIG  # noqa: E402
from helpers import rigged_checkpoint, write_calibration, write_checkpoint  # noqa: E402
from opf_mlx import PrivacyFilter, PrivacyFilterWarning  # noqa: E402
from opf_mlx.config import ModelConfig  # noqa: E402
from opf_mlx.decoding import BIAS_KEYS, ViterbiDecoder  # noqa: E402
from opf_mlx.labels import resolve_label_space  # noqa: E402
from opf_mlx.model import Transformer, rope_tables  # noqa: E402
from opf_mlx.pipeline import log_softmax  # noqa: E402
from opf_mlx.tokenizer import Tokenizer  # noqa: E402
from opf_mlx.weights import load_weights  # noqa: E402

DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}


def float32(array: mx.array) -> np.ndarray:
    return np.array(array.astype(mx.float32))


def reference_stages(model: ReferenceTransformer, token_ids: list[int]) -> dict[str, np.ndarray]:
    stages: dict[str, np.ndarray] = {}

    def keep(name: str):
        return lambda module, inputs, output: stages.__setitem__(name, output.detach().float().numpy()[0])

    hooks = [model.embedding.register_forward_hook(keep("embedding"))]
    for n, block in enumerate(model.block):
        hooks.append(block.attn.register_forward_hook(keep(f"block.{n}.attn")))
        hooks.append(block.mlp.register_forward_hook(keep(f"block.{n}.mlp")))
    tokens = torch.tensor([token_ids], dtype=torch.int32)
    with torch.inference_mode():
        stages["logits"] = model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))[0].float().numpy()
    for hook in hooks:
        hook.remove()
    return stages


def sublayer_agreement(model: Transformer, stages: dict[str, np.ndarray], tokens: int) -> dict[str, float]:
    cos, sin = rope_tables(model.config, tokens)
    agreement: dict[str, float] = {}
    previous = stages["embedding"]
    for n, block in enumerate(model.block):
        for part in ("attn", "mlp"):
            x = mx.array(previous[None]).astype(mx.bfloat16)
            out = block.attn(x, cos, sin) if part == "attn" else block.mlp(x)
            previous = stages[f"block.{n}.{part}"]
            agreement[f"block.{n}.{part}"] = float((float32(out)[0] == previous).mean())
    logits = model.unembedding(model.norm(mx.array(previous[None]).astype(mx.bfloat16)))
    agreement["logits"] = float((float32(logits)[0] == stages["logits"]).mean())
    return agreement


@pytest.mark.parametrize("seed", range(24))
def test_viterbi_paths_match_the_reference_decoder(released_config, seed):
    rng = np.random.default_rng(seed)
    space = resolve_label_space(released_config)
    biases = {key: float(rng.normal(scale=2.0)) if seed % 3 else 0.0 for key in BIAS_KEYS}
    scores = log_softmax(rng.normal(scale=3.0, size=(int(rng.integers(1, 80)), 33)).astype(np.float32))

    path = ViterbiDecoder(space, biases).decode(scores)

    reference = ViterbiCRFDecoder(label_info=build_label_info(space.class_names), **biases)
    assert path == reference.decode(torch.from_numpy(scores))


TEXTS = (
    "Alice was born on 1990-01-02.",
    "🧿 and 𝔘 met  Alice \n at noon ",
    "Call Alice" + chr(0xD800) + " now",
)


@pytest.mark.parametrize("seed", range(30))
def test_the_pipeline_matches_the_reference_on_rigged_models(tmp_path, tiny_config, seed):
    rng = np.random.default_rng(seed)
    text = TEXTS[seed % len(TEXTS)]
    space = resolve_label_space(tiny_config)
    tokens = Tokenizer(str(tiny_config["encoding"])).encode(text)
    rigged = {}
    for token in list(dict.fromkeys(tokens))[:15]:
        first, second = rng.choice(len(space.class_names), size=2, replace=False)
        rigged[token] = {space.class_names[first]: 30.0, space.class_names[second]: 30.0 - float(rng.uniform(0.2, 4))}
    checkpoint = rigged_checkpoint(tmp_path / "model", rigged, tiny_config)
    if rng.random() < 0.5:
        biases = {key: float(rng.normal(scale=3.0)) for key in BIAS_KEYS}
        write_calibration(checkpoint / "viterbi_calibration.json", **biases)
    options = {
        "decode_mode": str(rng.choice(["viterbi", "argmax"])),
        "output_mode": str(rng.choice(["typed", "redacted"])),
        "trim_whitespace": bool(rng.random() < 0.5),
        "discard_overlapping_predicted_spans": bool(rng.random() < 0.5),
    }
    n_ctx = int(rng.integers(1, 6)) if rng.random() < 0.4 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrivacyFilterWarning)
        result = PrivacyFilter(checkpoint, n_ctx=n_ctx, **options).redact(text).to_dict()

    reference = OPF(model=str(checkpoint), device="cpu", context_window_length=n_ctx, **options)
    expected = reference.redact(text).to_dict()
    if n_ctx is not None and n_ctx < len(tokens):
        note = result.pop("warning")
        assert "consecutive windows" in note
        assert note.endswith(expected.pop("warning", ""))
    assert result == expected


def as_float32(checkpoint: Path) -> Path:
    directory = checkpoint.with_name(checkpoint.name + "-float32")
    directory.mkdir()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    (directory / "config.json").write_text(json.dumps({**config, "param_dtype": "float32"}), encoding="utf-8")
    (directory / "model.safetensors").symlink_to(checkpoint / "model.safetensors")
    return directory


@pytest.fixture(params=range(6), ids=lambda seed: f"seed{seed}")
def random_model(request, tmp_path) -> tuple[Path, ModelConfig, list[int]]:
    raw = {**TINY_CONFIG, "vocab_size": 64}
    checkpoint = write_checkpoint(tmp_path / "tiny", raw, seed=request.param)
    rng = np.random.default_rng(100 + request.param)
    tokens = [int(token) for token in rng.integers(0, 64, size=int(rng.integers(1, 90)))]
    return checkpoint, ModelConfig.from_dict(raw), tokens


@pytest.mark.parametrize("device", DEVICES)
def test_random_float32_models_match_the_reference(random_model, device):
    checkpoint, config, tokens = random_model
    reference = ReferenceTransformer.from_checkpoint(str(as_float32(checkpoint)), device="cpu")

    with mx.stream(DEVICES[device]):
        model = Transformer(config)
        model.load_weights(list(load_weights(checkpoint, config, dtype="float32").items()), strict=True)
        logits = float32(model(mx.array([tokens])))[0]

    np.testing.assert_allclose(logits, reference_stages(reference, tokens)["logits"], rtol=0, atol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_every_sublayer_of_random_bfloat16_models_reproduces_the_reference(random_model, device):
    checkpoint, config, tokens = random_model
    stages = reference_stages(ReferenceTransformer.from_checkpoint(str(checkpoint), device="cpu"), tokens)

    with mx.stream(DEVICES[device]):
        model = Transformer(config)
        model.load_weights(list(load_weights(checkpoint, config).items()), strict=True)
        agreement = sublayer_agreement(model, stages, len(tokens))

    attention = [share for name, share in agreement.items() if name.endswith("attn")]
    others = [share for name, share in agreement.items() if not name.endswith("attn")]
    assert min(attention) >= 0.95, agreement
    assert min(others) >= 0.99, agreement


@pytest.mark.parametrize("device", DEVICES)
def test_every_sublayer_of_the_released_model_reproduces_the_reference(checkpoint_dir, device):
    released = PrivacyFilter(checkpoint_dir, device=device)
    tokens = released.tokenizer.encode(
        "From: Maria Gonzalez <maria.gonzalez@example.com>\nPlease send the invoice for account 4532 9981 0042 7716 "
        "to 1180 Harbor View Road, Portland, OR 97209, or call +1 (503) 555-0142 before 3 March 2021."
    )
    stages = reference_stages(ReferenceTransformer.from_checkpoint(str(checkpoint_dir), device="cpu"), tokens)

    with mx.stream(DEVICES[device]):
        agreement = sublayer_agreement(released.model, stages, len(tokens))

    attention = [share for name, share in agreement.items() if name.endswith("attn")]
    experts = [share for name, share in agreement.items() if name.endswith("mlp")]
    assert min(attention) >= 0.95, agreement
    assert min(experts) >= 0.999, agreement
    assert agreement["logits"] >= 0.995, agreement


@pytest.mark.parametrize("discard", [False, True])
def test_chained_overlapping_spans_are_resolved_like_the_reference(tmp_path, tiny_config, discard):
    rigged = {
        87: {"B-secret": 30.0},
        4103: {"E-secret": 30.0},
        100: {"B-private_person": 30.0},
        123: {"I-private_person": 30.0},
        64: {"I-private_person": 30.0},
        43120: {"E-private_person": 30.0},
        242: {"S-private_person": 30.0},
    }
    checkpoint = rigged_checkpoint(tmp_path / "model", rigged, tiny_config)

    result = PrivacyFilter(checkpoint, discard_overlapping_predicted_spans=discard).redact("x🧿a𝔘").to_dict()

    reference = OPF(model=str(checkpoint), device="cpu", discard_overlapping_predicted_spans=discard)
    assert result == reference.redact("x🧿a𝔘").to_dict()
