from __future__ import annotations

import http.server
import json
import math
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import mlx.core as mx
import numpy as np

from conftest import TINY_CONFIG

from opf_mlx.config import ModelConfig
from opf_mlx.decoding import BIAS_KEYS
from opf_mlx.labels import LabelSpace, resolve_label_space
from opf_mlx.weights import expected_tensors

MLX_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def random_weights(config: ModelConfig, seed: int = 0, scale: float = 0.5) -> dict[str, mx.array]:
    rng = np.random.default_rng(seed)
    weights: dict[str, mx.array] = {}
    for name, spec in expected_tensors(config).items():
        if name.endswith("norm.scale"):
            values = 1.0 + 0.1 * rng.standard_normal(spec.shape)
        elif name.endswith("sinks"):
            values = rng.standard_normal(spec.shape)
        else:
            values = scale * rng.standard_normal(spec.shape, dtype=np.float32)
        weights[name] = mx.array(np.asarray(values, dtype=np.float32)).astype(MLX_DTYPES[spec.default_dtype])
    mx.eval(weights)
    return weights


RIGGED_LOGIT = 20.0
_BACKGROUND_BASIS_DIM = 0


def rigged_weights(
    config: ModelConfig,
    space: LabelSpace,
    token_logits: Mapping[int, Mapping[str, float]],
) -> dict[str, mx.array]:
    if len(token_logits) >= config.hidden_size:
        raise ValueError(f"at most {config.hidden_size - 1} tokens can be rigged")
    specs = expected_tensors(config)
    arrays = {
        name: (np.ones if name.endswith("norm.scale") else np.zeros)(spec.shape, dtype=np.float32)
        for name, spec in specs.items()
    }
    rmsnorm_unit_scale = math.sqrt(config.hidden_size)
    embedding, unembedding = arrays["embedding.weight"], arrays["unembedding.weight"]
    embedding[:, _BACKGROUND_BASIS_DIM] = 1.0
    unembedding[space.background, _BACKGROUND_BASIS_DIM] = RIGGED_LOGIT / rmsnorm_unit_scale
    for basis_dim, (token, logits) in enumerate(token_logits.items(), start=1):
        embedding[token] = 0.0
        embedding[token, basis_dim] = 1.0
        for class_name, logit in logits.items():
            unembedding[space.class_names.index(class_name), basis_dim] = logit / rmsnorm_unit_scale
    weights = {name: mx.array(values).astype(MLX_DTYPES[specs[name].default_dtype]) for name, values in arrays.items()}
    mx.eval(weights)
    return weights


def write_checkpoint(
    directory: Path,
    raw_config: dict[str, object],
    weights: dict[str, mx.array] | None = None,
    *,
    seed: int = 0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
    if weights is None:
        weights = random_weights(ModelConfig.from_dict(raw_config), seed=seed)
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory


def rigged_checkpoint(
    directory: Path,
    token_logits: Mapping[int, Mapping[str, float]],
    raw_config: dict[str, object] = TINY_CONFIG,
) -> Path:
    weights = rigged_weights(ModelConfig.from_dict(raw_config), resolve_label_space(raw_config), token_logits)
    return write_checkpoint(directory, raw_config, weights)


def write_calibration(path: Path, **overrides: float) -> Path:
    biases = {**dict.fromkeys(BIAS_KEYS, 0.0), **overrides}
    path.write_text(json.dumps({"operating_points": {"default": {"biases": biases}}}), encoding="utf-8")
    return path


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    honor_range = True
    requests_seen: list[tuple[str, str | None]]

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.requests_seen.append((self.path, self.headers.get("Range")))
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        header = self.headers.get("Range")
        if self.honor_range and header and header.startswith("bytes=") and header.endswith("-"):
            offset = int(header[len("bytes=") : -1])
            if offset >= len(data):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}")
            data = data[offset:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def serve_directory(root: Path, *, honor_range: bool = True) -> Iterator[tuple[str, list[tuple[str, str | None]]]]:
    seen: list[tuple[str, str | None]] = []

    class Handler(_RangeHandler):
        requests_seen = seen

    Handler.honor_range = honor_range
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: Handler(*a, directory=str(root), **k))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
