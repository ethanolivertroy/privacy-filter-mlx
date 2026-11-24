"""Record what the original PyTorch implementation computes, for the main test suite.

    uv run --project reference python reference/make_golden.py

Everything written to tests/golden/ comes out of the reference implementation
(`opf`, pinned in reference/pyproject.toml) running on the CPU. opf-mlx is not
imported here, except for the test helper that writes the tiny checkpoint both
implementations then read.

tests/golden/tiny/        a 2-layer model with random weights (about 30 kB)
tests/golden/tiny.npz     its reference logits and the output of every sublayer
tests/golden/real.json    texts, token ids and the reference's redaction results
tests/golden/real.npz     the reference logits of the released checkpoint

The released checkpoint is read from $OPF_CHECKPOINT or ~/.opf/privacy_filter.
Each model is run twice: as stored (bfloat16) and with the same weights cast to
float32, which the reference does when `param_dtype` in config.json says so.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import tiktoken
import torch
from opf import OPF
from opf._model.model import Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from conftest import TINY_CONFIG  # noqa: E402
from helpers import write_checkpoint  # noqa: E402

UPSTREAM_COMMIT = "f7f00ca7fb869683eb732c010299d901457f19c3"
DTYPES = ("bfloat16", "float32")

PARAGRAPH = (
    "From: Maria Gonzalez <maria.gonzalez@example.com>\nTo: accounts team\n"
    "Please update the billing address for account 4532 9981 0042 7716 to 1180 Harbor View Road, Apt 4B, "
    "Portland, OR 97209. You can reach me on +1 (503) 555-0142 after 3 March 2021. The temporary password "
    "is hunter2-Xq9! and the API token is sk-live-9f8e7d6c5b4a3210. Thanks for sorting out the invoice so quickly; "
    "the quarterly report looks fine and the meeting moved to Thursday afternoon.\n\n"
)

# name, text, options for the reference's `OPF(...)`
CASES: list[tuple[str, str, dict[str, object]]] = [
    ("mixed", PARAGRAPH, {}),
    ("plain", "The committee reviewed the proposal and agreed to revisit the budget in the spring.", {}),
    (
        "unicode",
        "Zoë Müller (zoe.mueller@example.org) wohnt in der Hauptstraße 5, 10115 Berlin. 📞 +49 30 1234567 — "
        "田中太郎さんの電話番号は03-1234-5678です。 🧿",
        {},
    ),
    ("special-token", "Contact John Smith <|endoftext|> at john.smith@example.com or 212-555-0198.", {}),
    ("medium", PARAGRAPH * 5, {}),
    ("long", PARAGRAPH * 18, {}),
    ("lone-surrogate", "Call Priya Raman on 415-555-0133 " + chr(0xD800) + " before Friday.", {}),
    ("empty", "", {}),
    ("argmax", PARAGRAPH, {"decode_mode": "argmax"}),
    ("redacted", PARAGRAPH, {"output_mode": "redacted"}),
    ("untrimmed", PARAGRAPH, {"trim_whitespace": False}),
    ("discard-overlapping", PARAGRAPH, {"discard_overlapping_predicted_spans": True}),
    ("three-windows", PARAGRAPH, {"context_window_length": 50}),
]


def logits_of(model: Transformer, token_ids: list[int]) -> np.ndarray:
    """The reference's logits for one unpadded sequence, as float32 [tokens, classes]."""
    tokens = torch.tensor([token_ids], dtype=torch.int32)
    with torch.inference_mode():
        return model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))[0].float().numpy()


def as_float32(checkpoint: Path, scratch: Path) -> Path:
    """A directory that presents `checkpoint` as a float32 model: same files, another `param_dtype`."""
    directory = scratch / f"{checkpoint.name}-float32"
    directory.mkdir()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    (directory / "config.json").write_text(json.dumps({**config, "param_dtype": "float32"}), encoding="utf-8")
    for name in ("model.safetensors", "viterbi_calibration.json"):
        if (checkpoint / name).exists():
            (directory / name).symlink_to(checkpoint / name)
    return directory


def record_tiny(out: Path, scratch: Path) -> None:
    """A tiny random model: its logits for five sequences and every sublayer's output for one."""
    raw = {**TINY_CONFIG, "vocab_size": 64}
    checkpoint = write_checkpoint(out / "tiny", raw, seed=11)
    rng = np.random.default_rng(11)
    sequences = {f"tokens/{length}": rng.integers(0, 64, size=length).astype(np.int32) for length in (1, 2, 7, 40, 300)}
    arrays: dict[str, np.ndarray] = dict(sequences)
    for dtype, directory in (("bfloat16", checkpoint), ("float32", as_float32(checkpoint, scratch))):
        model = Transformer.from_checkpoint(str(directory), device="cpu")
        for key, ids in sequences.items():
            arrays[f"logits/{dtype}/{key.split('/')[1]}"] = logits_of(model, ids.tolist())
        if dtype == "bfloat16":
            stages: dict[str, np.ndarray] = {}

            def keep(name: str):
                return lambda module, inputs, output: stages.__setitem__(name, output.detach().float().numpy()[0])

            model.embedding.register_forward_hook(keep("embedding"))
            for n, block in enumerate(model.block):
                block.attn.register_forward_hook(keep(f"block.{n}.attn"))
                block.mlp.register_forward_hook(keep(f"block.{n}.mlp"))
            logits_of(model, sequences["tokens/40"].tolist())
            arrays.update({f"stages/{name}": value for name, value in stages.items()})
            attention = model.block[0].attn
            arrays["rope/cos"] = attention.rope.cos_cache[:300].numpy()
            arrays["rope/sin"] = attention.rope.sin_cache[:300].numpy()
    np.savez_compressed(out / "tiny.npz", **arrays)


def record_real(out: Path, scratch: Path, checkpoint: Path) -> None:
    directories = {"bfloat16": checkpoint, "float32": as_float32(checkpoint, scratch)}
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    encoding = tiktoken.get_encoding(config["encoding"])
    models = {dtype: Transformer.from_checkpoint(str(directory), device="cpu") for dtype, directory in directories.items()}

    arrays: dict[str, np.ndarray] = {}
    cases = []
    for name, text, options in CASES:
        token_ids = encoding.encode(text, allowed_special="all")
        case: dict[str, object] = {"name": name, "text": text, "options": options, "tokens": token_ids, "results": {}}
        for dtype in DTYPES:
            case["results"][dtype] = OPF(model=str(directories[dtype]), device="cpu", **options).redact(text).to_dict()
            if not options and token_ids:
                arrays[f"{name}/{dtype}"] = logits_of(models[dtype], token_ids)
        cases.append(case)
        spans = {dtype: case["results"][dtype]["summary"]["span_count"] for dtype in DTYPES}
        print(f"  {name:<20} {len(token_ids):>5} tokens  spans {spans}", file=sys.stderr)

    # The rotary tables: the first positions, then a sample all the way to the last one.
    rope = models["float32"].block[0].attn.rope
    last = rope.cos_cache.shape[0] - 1
    positions = np.unique(np.concatenate([np.arange(128), np.arange(0, last, 997), [last]]))
    arrays["rope/positions"] = positions.astype(np.int32)
    arrays["rope/cos"] = rope.cos_cache.numpy()[positions]
    arrays["rope/sin"] = rope.sin_cache.numpy()[positions]
    np.savez_compressed(out / "real.npz", **arrays)
    document = {
        "source": {
            "implementation": f"github.com/openai/privacy-filter@{UPSTREAM_COMMIT}",
            "torch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "checkpoint_bytes": (checkpoint / "model.safetensors").stat().st_size,
        },
        "cases": cases,
    }
    # ASCII-only JSON, so that the lone surrogate survives the round trip through a file.
    (out / "real.json").write_text(json.dumps(document, indent=1, ensure_ascii=True) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="released checkpoint directory (default: $OPF_CHECKPOINT or ~/.opf/privacy_filter)")
    parser.add_argument("--out", default=str(ROOT / "tests" / "golden"), help="where to write (default: tests/golden)")
    arguments = parser.parse_args()
    checkpoint = Path(arguments.checkpoint or os.environ.get("OPF_CHECKPOINT") or Path.home() / ".opf" / "privacy_filter")
    checkpoint = checkpoint.expanduser()
    if not (checkpoint / "model.safetensors").is_file():
        parser.error(f"no checkpoint at {checkpoint}; run `uv run opf-mlx download` first")
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="opf-golden-") as scratch:
        print("tiny model", file=sys.stderr)
        record_tiny(out, Path(scratch))
        print(f"released checkpoint at {checkpoint}", file=sys.stderr)
        record_real(out, Path(scratch), checkpoint)
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size:>10}  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
