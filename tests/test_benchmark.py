"""scripts/benchmark.py: it runs, and its table says what was measured."""

import subprocess
import sys
from pathlib import Path

from helpers import rigged_checkpoint

SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark.py"


def run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True, timeout=300)


def table_rows(output: str) -> list[list[str]]:
    """The lines of the table: the ones that start with a number."""
    return [line.split() for line in output.splitlines() if line.strip()[:1].isdigit()]


def test_the_benchmark_times_redaction_at_each_requested_size(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "model", {})  # a tiny model; its window on the gpu is 64 tokens

    done = run("--checkpoint", str(checkpoint), "--sizes", "8,150", "--repeats", "1")

    assert done.returncode == 0, done.stderr
    assert "on the gpu in bfloat16" in done.stdout
    small, large = table_rows(done.stdout)
    # tokens, windows, best ms, tokens/s, peak GiB
    assert 6 <= int(small[0]) <= 8 and small[1] == "1"
    assert 148 <= int(large[0]) <= 150 and large[1] == "3"
    for row in (small, large):
        assert float(row[2]) > 0 and float(row[3].replace(",", "")) > 0 and float(row[4]) >= 0


def test_the_benchmark_can_use_a_text_of_your_own(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "model", {})
    notes = tmp_path / "notes.txt"
    notes.write_text("Alice met Bob in Paris.\n", encoding="utf-8")  # 7 tokens; repeated to reach the size

    done = run("--checkpoint", str(checkpoint), "--device", "cpu", "--sizes", "40", "--repeats", "1", "--text-file", str(notes))

    assert done.returncode == 0, done.stderr
    assert "on the cpu in bfloat16" in done.stdout
    assert "notes.txt" in done.stdout
    (row,) = table_rows(done.stdout)
    assert 38 <= int(row[0]) <= 40 and row[1] == "1"
