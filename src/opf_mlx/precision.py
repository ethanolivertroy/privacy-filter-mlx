"""Keep float32 matrix products at float32 precision.

On Apple GPUs with neural accelerators, MLX runs float32 `matmul` and
`gather_mm` in a reduced-precision mode by default. The switch is the
environment variable MLX_ENABLE_TF32, read once, at the first GPU matmul of the
process. Measured on an M5 Max with MLX 0.32.4, the relative error is near 1e-3
instead of 1e-6, as if the inputs had been rounded to half precision. bfloat16
products are not affected.

The reference runs its expert router and expert MLPs in float32, so this
package turns the mode off when it is imported and, before a model is used,
measures that it is in fact off. Nothing here assumes which chip is present:
a GPU without the mode simply passes the measurement.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np

from .model import ExpertLinear

ENV_VAR = "MLX_ENABLE_TF32"

TOLERANCE = 1e-5

_PROBE_SHAPES = ((1, 640, 640), (64, 64, 64), (300, 640, 1280))
_GROUPED_PROBE_SHAPES = ((3, 40, 640, 1280), (5, 40, 640, 640))


class PrecisionError(RuntimeError):
    """float32 matrix products on the active device are not float32-accurate."""


def request_full_precision() -> None:
    """Ask MLX for exact float32 products, unless the switch was set explicitly."""
    os.environ.setdefault(ENV_VAR, "0")


def float32_matmul_error() -> float:
    """Largest error of float32 `matmul` on the current device, relative to the largest exact entry."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for rows, inner, columns in _PROBE_SHAPES:
        a = rng.standard_normal((rows, inner), dtype=np.float32)
        b = rng.standard_normal((inner, columns), dtype=np.float32)
        exact = a.astype(np.float64) @ b.astype(np.float64)
        got = np.array(mx.array(a) @ mx.array(b))
        worst = max(worst, float(np.abs(got - exact).max() / np.abs(exact).max()))
    return worst


def float32_grouped_product_error() -> float:
    """Largest error of the expert layers' float32 product on the current device, measured like `float32_matmul_error`.

    The expert layers multiply each row by the matrix of its own expert with
    `gather_mm`, which runs in other kernels than `matmul`. Measured on the
    same M5 Max: 8e-4 with the reduced-precision mode on, 1e-6 with it off.
    The layer measured is the one the model uses.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for experts, rows, inner, columns in _GROUPED_PROBE_SHAPES:
        layer = ExpertLinear(experts, inner, columns)
        layer.weight = mx.array(rng.standard_normal((experts, inner, columns), dtype=np.float32))
        x = rng.standard_normal((rows, inner), dtype=np.float32)
        expert = np.sort(rng.integers(0, experts, size=rows)).astype(np.uint32)
        weight = np.array(layer.weight).astype(np.float64)
        exact = np.stack([row @ weight[e] for row, e in zip(x.astype(np.float64), expert)])
        got = np.array(layer(mx.array(x), mx.array(expert)))
        worst = max(worst, float(np.abs(got - exact).max() / np.abs(exact).max()))
    return worst


def float32_product_error() -> float:
    """The larger of the two measurements; not a number if either is."""
    return float(np.max([float32_matmul_error(), float32_grouped_product_error()]))


def require_full_precision() -> float:
    """Measure float32 products on the current device; raise `PrecisionError` if they are inaccurate."""
    error = float32_product_error()
    if math.isnan(error) or error > TOLERANCE:
        raise PrecisionError(
            f"float32 matrix products on this device run at reduced precision (relative error {error:.1e}, "
            f"expected below {TOLERANCE:.0e}). MLX enables this by default on GPUs with neural accelerators and "
            f"reads the setting once, at the first GPU matmul of the process. Set {ENV_VAR}=0 in the environment, "
            "or import opf_mlx before any other MLX computation runs, or run on the cpu "
            "(device='cpu', or --device cpu on the command line)."
        )
    return error
