import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from helpers import rigged_checkpoint

SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark.py"


class BenchmarkRow(NamedTuple):
    tokens: int
    windows: int
    elapsed_ms: float
    tokens_per_s: float
    peak_gib: float


def run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True, timeout=300)


def table_rows(output: str) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for line in output.splitlines():
        if line.strip()[:1].isdigit():
            tokens, windows, elapsed_ms, tokens_per_s, peak_gib = line.split()
            rows.append(
                BenchmarkRow(
                    tokens=int(tokens),
                    windows=int(windows),
                    elapsed_ms=float(elapsed_ms),
                    tokens_per_s=float(tokens_per_s.replace(",", "")),
                    peak_gib=float(peak_gib),
                )
            )
    return rows


def test_the_benchmark_times_redaction_at_each_requested_size(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "model", {})

    done = run("--checkpoint", str(checkpoint), "--sizes", "8,150", "--repeats", "1")

    assert done.returncode == 0, done.stderr
    assert "on the gpu in bfloat16" in done.stdout
    small, large = table_rows(done.stdout)
    assert 6 <= small.tokens <= 8 and small.windows == 1
    assert 148 <= large.tokens <= 150 and large.windows == 3
    for row in (small, large):
        assert row.elapsed_ms > 0 and row.tokens_per_s > 0 and row.peak_gib >= 0


def test_the_benchmark_can_use_a_text_of_your_own(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "model", {})
    notes = tmp_path / "notes.txt"
    notes.write_text("Alice met Bob in Paris.\n", encoding="utf-8")

    done = run("--checkpoint", str(checkpoint), "--device", "cpu", "--sizes", "40", "--repeats", "1", "--text-file", str(notes))

    assert done.returncode == 0, done.stderr
    assert "on the cpu in bfloat16" in done.stdout
    assert "notes.txt" in done.stdout
    (row,) = table_rows(done.stdout)
    assert 38 <= row.tokens <= 40 and row.windows == 1
