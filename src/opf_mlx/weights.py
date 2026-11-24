"""Checkpoint tensors: what must be there, what is there, and loading it.

The original checkpoint is a safetensors file that MLX can read directly, so
there is no conversion step. Before any tensor is loaded, the file header is
compared with the inventory the model needs: every name, shape and dtype.
Anything missing, unexpected or mis-shaped is reported in a single error.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .config import ModelConfig

_SAFETENSORS_DTYPES: dict[str, tuple[str, int]] = {
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "I32": ("int32", 4),
    "I64": ("int64", 8),
    "U8": ("uint8", 1),
    "U16": ("uint16", 2),
    "U32": ("uint32", 4),
    "U64": ("uint64", 8),
    "BOOL": ("bool", 1),
}
_MLX_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}
_ITEM_SIZE = {"bfloat16": 2, "float32": 4}


class WeightsError(ValueError):
    """The checkpoint's tensors are not what the model needs."""


@dataclass(frozen=True)
class TensorSpec:
    """What the model requires of one tensor."""

    shape: tuple[int, ...]
    dtypes: tuple[str, ...]
    default_dtype: str = ""

    def __post_init__(self) -> None:
        if not self.default_dtype:
            object.__setattr__(self, "default_dtype", self.dtypes[0])


@dataclass(frozen=True)
class TensorEntry:
    """What a safetensors header says about one tensor."""

    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: Path


def expected_tensors(config: ModelConfig) -> dict[str, TensorSpec]:
    """Every tensor of the model, keyed by its name in the checkpoint.

    The names are also the attribute paths of `opf_mlx.model.Transformer`, so
    the checkpoint loads without a name map.

    Tensors that live in the residual stream's precision must be stored in
    `param_dtype`. The norm scales and attention sinks are only ever used in
    float32, so bfloat16 or float32 storage is equally lossless for them.
    """
    stream = (config.param_dtype,)
    either = tuple(dict.fromkeys((config.param_dtype, "float32", "bfloat16")))
    d, e = config.hidden_size, config.num_experts

    tensors: dict[str, TensorSpec] = {"embedding.weight": TensorSpec((config.vocab_size, d), stream, config.param_dtype)}
    for n in range(config.num_hidden_layers):
        attn, mlp = f"block.{n}.attn", f"block.{n}.mlp"
        tensors[f"{attn}.norm.scale"] = TensorSpec((d,), either, config.param_dtype)
        tensors[f"{attn}.qkv.weight"] = TensorSpec((config.qkv_size, d), stream, config.param_dtype)
        tensors[f"{attn}.qkv.bias"] = TensorSpec((config.qkv_size,), stream, config.param_dtype)
        tensors[f"{attn}.sinks"] = TensorSpec((config.num_attention_heads,), ("float32", "bfloat16"), "float32")
        tensors[f"{attn}.out.weight"] = TensorSpec((d, config.attention_width), stream, config.param_dtype)
        tensors[f"{attn}.out.bias"] = TensorSpec((d,), stream, config.param_dtype)
        tensors[f"{mlp}.norm.scale"] = TensorSpec((d,), either, config.param_dtype)
        tensors[f"{mlp}.gate.weight"] = TensorSpec((e, d), stream, config.param_dtype)
        tensors[f"{mlp}.gate.bias"] = TensorSpec((e,), stream, config.param_dtype)
        tensors[f"{mlp}.swiglu.weight"] = TensorSpec((e, d, 2 * config.intermediate_size), stream, config.param_dtype)
        tensors[f"{mlp}.swiglu.bias"] = TensorSpec((e, 2 * config.intermediate_size), stream, config.param_dtype)
        tensors[f"{mlp}.out.weight"] = TensorSpec((e, config.intermediate_size, d), stream, config.param_dtype)
        tensors[f"{mlp}.out.bias"] = TensorSpec((e, d), stream, config.param_dtype)
    tensors["norm.scale"] = TensorSpec((d,), either, config.param_dtype)
    tensors["unembedding.weight"] = TensorSpec((config.num_labels, d), stream, config.param_dtype)
    return tensors


def weights_nbytes(config: ModelConfig, dtype: str | None = None) -> int:
    """Memory the loaded weights occupy at `dtype` (default: as stored in the released layout)."""
    total = 0
    for spec in expected_tensors(config).values():
        total += math.prod(spec.shape) * _ITEM_SIZE[dtype or spec.default_dtype]
    return total


def read_safetensors_header(path: str | Path) -> dict[str, TensorEntry]:
    """Parse and sanity-check a safetensors header without reading tensor data.

    Layout: 8 bytes little-endian header length, a JSON header, then raw tensor
    bytes. Each header entry gives a dtype, a shape and a byte range.
    """
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise WeightsError(f"{path}: too short to be a safetensors file")
        (header_size,) = struct.unpack("<Q", prefix)
        if header_size > file_size - 8:
            raise WeightsError(f"{path}: header length {header_size} exceeds the file size; not a safetensors file")
        try:
            header = json.loads(handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeightsError(f"{path}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise WeightsError(f"{path}: header must be a JSON object")

    data_size = file_size - 8 - header_size
    entries: dict[str, TensorEntry] = {}
    described = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            code, shape, (begin, end) = info["dtype"], tuple(info["shape"]), info["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise WeightsError(f"{path}: malformed header entry for {name!r}") from exc
        if code not in _SAFETENSORS_DTYPES:
            raise WeightsError(f"{path}: tensor {name!r} has unknown dtype {code!r}")
        dtype, item_size = _SAFETENSORS_DTYPES[code]
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape) or end - begin != math.prod(shape) * item_size:
            raise WeightsError(
                f"{path}: tensor {name!r} occupies bytes {begin}..{end}, "
                f"which does not match shape {shape} of {dtype}"
            )
        described = max(described, end)
        entries[name] = TensorEntry(dtype=dtype, shape=shape, nbytes=end - begin, file=path)

    if described > data_size:
        raise WeightsError(
            f"{path}: file is truncated; the header describes {described} bytes of tensor data "
            f"but only {data_size} are present"
        )
    return entries


def read_inventory(checkpoint_dir: str | Path) -> dict[str, TensorEntry]:
    """Headers of every `.safetensors` file in a checkpoint directory, merged."""
    files = sorted(Path(checkpoint_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in checkpoint directory {checkpoint_dir}")
    merged: dict[str, TensorEntry] = {}
    for file in files:
        for name, entry in read_safetensors_header(file).items():
            if name in merged:
                raise WeightsError(f"{name} appears in more than one file: {merged[name].file.name}, {file.name}")
            merged[name] = entry
    return merged


def validate_tensors(found: dict[str, TensorEntry], expected: dict[str, TensorSpec]) -> None:
    """Raise one `WeightsError` listing every way `found` differs from `expected`."""
    problems = [f"missing tensor: {name}" for name in expected if name not in found]
    problems += [f"unexpected tensor: {name}" for name in found if name not in expected]
    for name, spec in expected.items():
        entry = found.get(name)
        if entry is None:
            continue
        if entry.shape != spec.shape:
            problems.append(f"{name}: expected shape {spec.shape}, found {entry.shape}")
        if entry.dtype not in spec.dtypes:
            problems.append(f"{name}: expected dtype {' or '.join(spec.dtypes)}, found {entry.dtype}")
    if problems:
        raise WeightsError(
            f"checkpoint does not match the model ({len(problems)} problem(s)):\n  " + "\n  ".join(problems)
        )


def load_weights(checkpoint_dir: str | Path, config: ModelConfig, *, dtype: str | None = None) -> dict[str, mx.array]:
    """Validate a checkpoint directory's tensors and load them.

    With `dtype=None` every tensor keeps its stored dtype. `dtype="float32"`
    upcasts everything, which is exact. Downcasting is refused.
    """
    if dtype is not None and dtype not in (config.param_dtype, "float32"):
        raise ValueError(
            f"cannot load a {config.param_dtype} checkpoint as {dtype}: "
            f"supported are {config.param_dtype} (as stored) and float32 (exact upcast)"
        )
    upcast = dtype == "float32"

    expected = expected_tensors(config)
    found = read_inventory(checkpoint_dir)
    validate_tensors(found, expected)

    loaded: dict[str, mx.array] = {}
    for file in sorted({entry.file for entry in found.values()}):
        loaded.update(mx.load(str(file)))

    weights: dict[str, mx.array] = {}
    for name in expected:
        array, entry = loaded[name], found[name]
        if tuple(array.shape) != entry.shape or array.dtype != _MLX_DTYPES[entry.dtype]:
            raise WeightsError(
                f"{name}: loaded as {tuple(array.shape)} {array.dtype}, header says {entry.shape} {entry.dtype}"
            )
        weights[name] = array.astype(mx.float32) if upcast else array
    return weights
