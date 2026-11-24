"""Hardware and software detection."""

import dataclasses
import os
import platform

import mlx.core as mx
import pytest

import opf_mlx.env as environment
from opf_mlx.env import (
    Environment,
    InsufficientMemoryError,
    check_memory,
    detect,
    format_bytes,
    format_report,
    gpu_memory_limit,
    parse_vm_stat,
    total_memory,
)
from opf_mlx.precision import TOLERANCE

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             1000.
Pages active:                         500000.
Pages inactive:                         2000.
Pages speculative:                       300.
Pages throttled:                           0.
Pages wired down:                     200000.
Pages purgeable:                        4000.
"Translation faults":             1234567890.
Pages occupied by compressor:          70000.
"""


def test_available_memory_counts_free_inactive_and_speculative_pages():
    # (1000 + 2000 + 300) pages * 16384 bytes
    assert parse_vm_stat(VM_STAT) == 54_067_200


def test_unparseable_vm_stat_output_gives_no_estimate():
    assert parse_vm_stat("vm_stat: command not found") is None


def test_detect_reports_this_machine():
    env = detect()

    # Cross-checked against sources the detector does not use.
    assert env.memory_total == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert env.python_version == platform.python_version()
    assert env.mlx_version == mx.__version__
    assert env.macos_version == platform.mac_ver()[0]
    assert env.architecture == platform.machine()
    assert env.chip  # e.g. "Apple M-series"; never assumed, only reported
    assert 0 < env.memory_available <= env.memory_total
    assert 0 < env.gpu_memory_limit <= env.memory_total
    # Measured, not assumed: this process asked for exact float32 products.
    assert 0 <= env.float32_matmul_error < TOLERANCE


def test_the_precision_reported_covers_the_grouped_product_of_the_expert_layers(monkeypatch):
    # The report must not call a device accurate that the model would refuse.
    from opf_mlx import precision

    monkeypatch.setattr(precision, "float32_grouped_product_error", lambda: 8e-4)

    assert detect().float32_matmul_error == 8e-4


def test_total_memory_is_the_installed_memory():
    assert total_memory() == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


# A made-up machine, so the report can be checked character for character.
MACHINE = Environment(
    chip="Example M9",
    model_identifier="Mac99,1",
    architecture="arm64",
    cpu_cores=10,
    performance_cores=6,
    efficiency_cores=4,
    gpu_cores=16,
    memory_total=32 * 2**30,
    memory_available=12 * 2**30,
    gpu_name="Example M9",
    gpu_memory_limit=24 * 2**30,
    gpu_max_buffer=16 * 2**30,
    float32_matmul_error=1e-6,
    macos_version="26.0",
    macos_build="25A1",
    python_version="3.13.0",
    mlx_version="0.32.4",
    numpy_version="2.5.5",
    tiktoken_version="0.14.0",
)


def test_report_lists_every_detected_fact():
    assert format_report(MACHINE) == "\n".join(
        [
            "chip            Example M9 (Mac99,1, arm64)",
            "cpu cores       10 (6 performance + 4 efficiency)",
            "gpu             Example M9, 16 cores",
            "unified memory  32.0 GiB total, 12.0 GiB available",
            "usable by MLX   24.0 GiB (largest single buffer 16.0 GiB)",
            "float32 matmul  full precision on the gpu (relative error 1.0e-06)",
            "macOS           26.0 (25A1)",
            "python          3.13.0",
            "mlx             0.32.4",
            "numpy           2.5.5",
            "tiktoken        0.14.0",
        ]
    )


def test_report_flags_reduced_float32_precision():
    report = format_report(dataclasses.replace(MACHINE, float32_matmul_error=8e-4))

    assert "float32 matmul  REDUCED precision on the gpu (relative error 8.0e-04); set MLX_ENABLE_TF32=0" in report


def test_report_says_unknown_instead_of_guessing():
    report = format_report(
        dataclasses.replace(MACHINE, chip=None, gpu_cores=None, memory_available=None, float32_matmul_error=None)
    )

    assert "chip            unknown (Mac99,1, arm64)" in report
    assert "gpu             Example M9, unknown cores" in report
    assert "unified memory  32.0 GiB total, unknown available" in report
    assert "float32 matmul  unknown" in report


def test_weights_larger_than_the_device_limit_are_refused():
    with pytest.raises(InsufficientMemoryError, match="6.0 GiB.*4.0 GiB"):
        check_memory(required=6 * 2**30, limit=4 * 2**30, available=16 * 2**30)


def test_weights_larger_than_free_memory_warn_but_proceed():
    warning = check_memory(required=6 * 2**30, limit=8 * 2**30, available=5 * 2**30)

    assert "6.0 GiB" in warning and "5.0 GiB" in warning


def test_enough_memory_is_silent():
    assert check_memory(required=3 * 2**30, limit=8 * 2**30, available=5 * 2**30) is None


def test_unknown_available_memory_is_not_treated_as_zero():
    assert check_memory(required=3 * 2**30, limit=8 * 2**30, available=None) is None


@pytest.mark.parametrize(
    "value, text",
    [(0, "0 B"), (1023, "1023 B"), (1536, "1.5 KiB"), (2_798_984_088, "2.6 GiB"), (137_438_953_472, "128.0 GiB")],
)
def test_format_bytes(value, text):
    assert format_bytes(value) == text


def test_an_unknown_limit_is_not_treated_as_zero():
    assert check_memory(required=3 * 2**30, limit=None, available=5 * 2**30) is None


def test_report_can_carry_extra_rows_in_the_same_columns():
    report = format_report(MACHINE, extra_rows=[("checkpoint", "/models/pf (installed)")])

    assert report.splitlines()[-1] == "checkpoint      /models/pf (installed)"


def test_weights_exactly_as_large_as_the_device_limit_are_accepted():
    assert check_memory(required=4 * 2**30, limit=4 * 2**30, available=16 * 2**30) is None


def test_weights_exactly_as_large_as_free_memory_do_not_warn():
    assert check_memory(required=4 * 2**30, limit=8 * 2**30, available=4 * 2**30) is None


def test_the_device_limit_is_a_real_share_of_this_machines_memory():
    assert 0 < gpu_memory_limit() <= total_memory()


def test_a_probe_that_fails_gives_nothing_to_parse():
    # A command that exits with an error may still have printed something.
    assert environment._run(["sh", "-c", "echo 42; exit 3"]) is None
    assert environment._run(["/nonexistent/opf-mlx-probe"]) is None
    assert environment._run(["sh", "-c", "echo 42"]) == "42\n"
