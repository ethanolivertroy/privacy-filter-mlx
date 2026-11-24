"""Detect the machine this is running on. Nothing here is assumed.

Hardware facts come from `sysctl`, free memory from `vm_stat`, the GPU's usable
memory from MLX itself, and float32 precision from a measurement. Every probe
may fail on an unusual system, in which case the field is `None` rather than a
guess.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Sequence

import mlx.core as mx

from . import precision


class InsufficientMemoryError(MemoryError):
    """The model cannot fit in the memory MLX may use on this machine."""


@dataclass(frozen=True)
class Environment:
    chip: str | None
    model_identifier: str | None
    architecture: str
    cpu_cores: int | None
    performance_cores: int | None
    efficiency_cores: int | None
    gpu_cores: int | None
    memory_total: int | None
    memory_available: int | None
    gpu_name: str | None
    gpu_memory_limit: int | None
    gpu_max_buffer: int | None
    float32_matmul_error: float | None
    macos_version: str
    macos_build: str | None
    python_version: str
    mlx_version: str
    numpy_version: str
    tiktoken_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect() -> Environment:
    """Probe hardware, operating system and library versions."""
    import numpy
    import tiktoken

    device = mx.device_info()
    return Environment(
        chip=_sysctl("machdep.cpu.brand_string"),
        model_identifier=_sysctl("hw.model"),
        architecture=platform.machine(),
        cpu_cores=_sysctl_int("hw.physicalcpu"),
        performance_cores=_sysctl_int("hw.perflevel0.physicalcpu"),
        efficiency_cores=_sysctl_int("hw.perflevel1.physicalcpu"),
        gpu_cores=_gpu_cores(),
        memory_total=_sysctl_int("hw.memsize"),
        memory_available=available_memory(),
        gpu_name=device.get("device_name"),
        gpu_memory_limit=device.get("max_recommended_working_set_size"),
        gpu_max_buffer=device.get("max_buffer_length"),
        float32_matmul_error=_float32_matmul_error(),
        macos_version=platform.mac_ver()[0],
        macos_build=_sysctl("kern.osversion"),
        python_version=platform.python_version(),
        mlx_version=mx.__version__,
        numpy_version=numpy.__version__,
        tiktoken_version=tiktoken.__version__,
    )


def available_memory() -> int | None:
    """Bytes of memory macOS could hand out without compressing or swapping."""
    output = _run(["vm_stat"])
    return parse_vm_stat(output) if output is not None else None


def gpu_memory_limit() -> int | None:
    """The most memory MLX recommends using on the GPU."""
    return mx.device_info().get("max_recommended_working_set_size")


def total_memory() -> int | None:
    """Installed unified memory in bytes."""
    return _sysctl_int("hw.memsize")


def parse_vm_stat(output: str) -> int | None:
    """Free + inactive + speculative pages, in bytes (the usual macOS estimate)."""
    page = re.search(r"page size of (\d+) bytes", output)
    counts = {
        key: re.search(rf"^Pages {key}:\s+(\d+)\.", output, flags=re.MULTILINE)
        for key in ("free", "inactive", "speculative")
    }
    if page is None or any(match is None for match in counts.values()):
        return None
    return sum(int(match.group(1)) for match in counts.values()) * int(page.group(1))


def check_memory(*, required: int, limit: int | None, available: int | None) -> str | None:
    """Decide whether `required` bytes of weights can be loaded.

    `limit` is the most the chosen device may use at all; exceeding it raises.
    Exceeding what is currently free returns a warning. Otherwise `None`.
    """
    if limit is not None and required > limit:
        raise InsufficientMemoryError(
            f"The model needs about {format_bytes(required)} but MLX can use at most "
            f"{format_bytes(limit)} on this machine."
        )
    if available is not None and required > available:
        return (
            f"The model needs about {format_bytes(required)} but only {format_bytes(available)} "
            "of memory is free; macOS will compress or swap, and inference will be slow."
        )
    return None


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    size = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
    raise AssertionError("unreachable")


def format_report(env: Environment, extra_rows: Sequence[tuple[str, str]] = ()) -> str:
    """Human-readable environment summary, with `extra_rows` appended in the same columns."""

    def size(value: int | None) -> str:
        return format_bytes(value) if value is not None else "unknown"

    def text(value: object) -> str:
        return str(value) if value is not None else "unknown"

    cores = text(env.cpu_cores)
    if env.performance_cores is not None and env.efficiency_cores is not None:
        cores += f" ({env.performance_cores} performance + {env.efficiency_cores} efficiency)"
    rows = [
        ("chip", f"{text(env.chip)} ({text(env.model_identifier)}, {env.architecture})"),
        ("cpu cores", cores),
        ("gpu", f"{text(env.gpu_name)}, {text(env.gpu_cores)} cores"),
        ("unified memory", f"{size(env.memory_total)} total, {size(env.memory_available)} available"),
        ("usable by MLX", f"{size(env.gpu_memory_limit)} (largest single buffer {size(env.gpu_max_buffer)})"),
        ("float32 matmul", _precision_text(env.float32_matmul_error)),
        ("macOS", f"{env.macos_version} ({text(env.macos_build)})"),
        ("python", env.python_version),
        ("mlx", env.mlx_version),
        ("numpy", env.numpy_version),
        ("tiktoken", env.tiktoken_version),
        *extra_rows,
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in rows)


def _precision_text(error: float | None) -> str:
    if error is None:
        return "unknown"
    if error <= precision.TOLERANCE:
        return f"full precision on the gpu (relative error {error:.1e})"
    return f"REDUCED precision on the gpu (relative error {error:.1e}); set {precision.ENV_VAR}=0"


def _float32_matmul_error() -> float | None:
    try:
        return precision.float32_product_error()
    except RuntimeError:
        return None


def _run(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _sysctl(key: str) -> str | None:
    output = _run(["sysctl", "-n", key])
    return output.strip() or None if output is not None else None


def _sysctl_int(key: str) -> int | None:
    value = _sysctl(key)
    return int(value) if value is not None and value.isdigit() else None


def _gpu_cores() -> int | None:
    output = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20.0)
    if output is None:
        return None
    try:
        for display in json.loads(output).get("SPDisplaysDataType", []):
            cores = display.get("sppci_cores")
            if cores is not None:
                return int(cores)
    except (ValueError, TypeError, AttributeError):
        return None
    return None
