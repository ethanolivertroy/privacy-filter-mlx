import os
import subprocess
import sys

import pytest

from opf_mlx.precision import (
    TOLERANCE,
    float32_grouped_product_error,
    float32_matmul_error,
    require_full_precision,
)


def run_python(code: str, tf32: str | None = None) -> subprocess.CompletedProcess:
    environment = {key: value for key, value in os.environ.items() if key != "MLX_ENABLE_TF32"}
    if tf32 is not None:
        environment["MLX_ENABLE_TF32"] = tf32
    return subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120)


def test_float32_matmul_is_accurate_in_this_process():
    assert float32_matmul_error() < TOLERANCE


def test_the_grouped_product_of_the_expert_layers_is_accurate_in_this_process():
    assert float32_grouped_product_error() < TOLERANCE


def test_requiring_full_precision_returns_the_measured_error():
    assert 0 <= require_full_precision() < TOLERANCE


def test_importing_the_package_turns_reduced_precision_off():
    result = run_python("import opf_mlx, os; print(os.environ['MLX_ENABLE_TF32'])")

    assert result.stdout.strip() == "0", result.stderr


def test_an_explicit_setting_is_left_alone():
    result = run_python("import opf_mlx, os; print(os.environ['MLX_ENABLE_TF32'])", tf32="1")

    assert result.stdout.strip() == "1", result.stderr


FORCED_ON = """
import mlx.core as mx
from opf_mlx.precision import (
    PrecisionError,
    float32_grouped_product_error,
    float32_matmul_error,
    require_full_precision,
)

print(max(float32_matmul_error(), float32_grouped_product_error()))
with mx.stream(mx.cpu):
    print(max(float32_matmul_error(), float32_grouped_product_error()))
try:
    require_full_precision()
    print("accepted")
except PrecisionError as exc:
    print("refused:", exc)
"""


def test_reduced_precision_is_refused_when_forced_on():
    result = run_python(FORCED_ON, tf32="1")
    assert result.returncode == 0, result.stderr
    gpu_error, cpu_error, verdict = result.stdout.strip().split("\n", 2)

    assert float(cpu_error) < TOLERANCE
    if float(gpu_error) < TOLERANCE:
        pytest.skip("this GPU has no reduced-precision float32 products")
    assert verdict.startswith("refused:")
    assert "MLX_ENABLE_TF32=0" in verdict
    assert "cpu" in verdict


def test_an_inaccurate_grouped_product_is_refused_even_when_matmul_is_accurate(monkeypatch):
    from opf_mlx import precision

    monkeypatch.setattr(precision, "float32_grouped_product_error", lambda: 8e-4)

    with pytest.raises(precision.PrecisionError, match="reduced precision.*8.0e-04"):
        precision.require_full_precision()


@pytest.mark.parametrize("measurement", ["float32_matmul_error", "float32_grouped_product_error"])
def test_a_measurement_that_is_not_a_number_is_refused(monkeypatch, measurement):
    from opf_mlx import precision

    monkeypatch.setattr(precision, measurement, lambda: float("nan"))

    with pytest.raises(precision.PrecisionError, match="reduced precision"):
        precision.require_full_precision()
