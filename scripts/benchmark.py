"""Time opf-mlx on this machine.

    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --device cpu
    uv run python scripts/benchmark.py --sizes 512,4096 --text-file notes.txt

Times `PrivacyFilter.redact`, text in and result out, on the beginning of one
long text cut to each requested number of tokens. By default that text is the
Python language reference that ships with Python (`pydoc_data.topics`, about
530,000 characters of prose and code): it is varied, and every installation
has it. Varied matters. The tokens of a repetitive text are routed to few
experts, which runs up to a quarter faster on the GPU than ordinary text does.

"peak GiB" is the most memory MLX held at once, weights included.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable

import mlx.core as mx

from opf_mlx import PrivacyFilter, PrivacyFilterWarning
from opf_mlx.env import Environment, detect, format_bytes

DEFAULT_SIZES = {"gpu": (128, 2048, 8192, 32768, 128000), "cpu": (128, 2048, 4096)}


def longest_prefix(encode: Callable[[str], list[int]], text: str, n_tokens: int) -> str:
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(encode(text[:middle])) <= n_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def best_time(redactor: PrivacyFilter, text: str, repeats: int) -> float:
    best = float("inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrivacyFilterWarning)
        for _ in range(repeats):
            started = time.perf_counter()
            redactor.redact(text)
            best = min(best, time.perf_counter() - started)
    return best


def describe(machine: Environment) -> str:
    memory = format_bytes(machine.memory_total) if machine.memory_total is not None else "unknown memory"
    return (
        f"{machine.chip or 'unknown chip'}, {memory}, macOS {machine.macos_version}, "
        f"Python {machine.python_version}, MLX {machine.mlx_version}"
    )


def token_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(","))
    if min(counts) < 1:
        raise ValueError(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Time opf-mlx redaction on this machine.")
    parser.add_argument("--checkpoint", help="Checkpoint directory (default: where opf-mlx looks).")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="Device to run on (default: gpu).")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), help="Weight precision (default: as stored).")
    parser.add_argument(
        "--sizes",
        type=token_counts,
        help="Input lengths in tokens, separated by commas "
        f"(default: {','.join(map(str, DEFAULT_SIZES['gpu']))} on the gpu, {','.join(map(str, DEFAULT_SIZES['cpu']))} on the cpu).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per size; the fastest one counts (default: 3).")
    parser.add_argument("--text-file", type=Path, help="Cut the inputs from this text instead (repeated if it is short).")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.text_file is None:
        import pydoc_data.topics

        corpus, source = "\n".join(pydoc_data.topics.topics.values()), "the Python language reference shipped with Python"
    else:
        corpus, source = args.text_file.read_text(encoding="utf-8"), args.text_file.name
        if not corpus.strip():
            parser.error(f"{args.text_file} holds no text")

    started = time.perf_counter()
    redactor = PrivacyFilter(args.checkpoint, device=args.device, dtype=args.dtype)
    loaded = time.perf_counter() - started
    encode = redactor.tokenizer.encode
    tokens_per_copy = len(encode(corpus))

    print(f"machine   {describe(detect())}")
    print(f"model     {redactor.checkpoint} on the {redactor.device} in {redactor.dtype}, loaded in {loaded:.2f} s")
    print(f"window    {redactor.n_ctx} tokens per forward pass; longer inputs are split")
    print(f"text      {source} ({len(corpus):,} characters, {tokens_per_copy:,} tokens)")
    print(f"timing    PrivacyFilter.redact, text in and result out; fastest of {args.repeats}")
    print(f"\n{'tokens':>10}{'windows':>9}{'best ms':>11}{'tokens/s':>12}{'peak GiB':>11}")
    for size in args.sizes or DEFAULT_SIZES[args.device]:
        text = longest_prefix(encode, corpus * (size // tokens_per_copy + 1), size)
        tokens = len(encode(text))
        mx.reset_peak_memory()
        seconds = best_time(redactor, text, args.repeats)
        print(
            f"{tokens:>10}{-(-tokens // redactor.n_ctx):>9}{seconds * 1000:>11.1f}"
            f"{tokens / seconds:>12,.0f}{mx.get_peak_memory() / 2**30:>11.2f}"
        )


if __name__ == "__main__":
    main()
