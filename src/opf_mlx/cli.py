"""The `opf-mlx` command line.

    opf-mlx "some text"       redact one text (the `redact` subcommand is implied)
    opf-mlx -f notes.txt      redact files, one example per file
    ... | opf-mlx             redact standard input, one example per line
    opf-mlx                   in a terminal: an interactive prompt
    opf-mlx serve             answer detection requests on a local socket
    opf-mlx env               this machine's hardware, memory and library versions
    opf-mlx download          fetch the pretrained checkpoint
    opf-mlx verify            check an installed checkpoint against the pinned release

`redact` takes the flags of the reference's `opf` command and prints the same
formats (`opf/__main__.py`). It adds --dtype, --window-policy, --halo and
--no-download, and its --device is `gpu` or `cpu`.

`serve` takes the model and decode flags of `redact` and answers requests
instead of reading input (`opf_mlx.serve`).

This module only parses arguments, reads input and prints. Everything about
the model is in `opf_mlx.pipeline`.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from . import checkpoint, env, serve
from .config import ModelConfig
from .labels import BACKGROUND
from .pipeline import DECODE_MODES, DEVICES, DTYPES, OUTPUT_MODES, PrivacyFilter, PrivacyFilterWarning
from .precision import PrecisionError
from .spans import DetectedSpan
from .weights import expected_tensors, read_inventory, validate_tensors
from .windows import POLICIES

PROG = "opf-mlx"
ROOT_HELP = f"""usage: {PROG} [redact] [options] [text]
       {PROG} serve | env | download | verify [options]

OpenAI Privacy Filter in MLX: redact private information from text, locally on Apple Silicon.

Subcommands:
  redact    Redact text (default, implied).
  serve     Answer detection requests on a local socket: text in, spans out.
  env       Report this machine's hardware, memory and library versions.
  download  Fetch the pretrained checkpoint.
  verify    Check an installed checkpoint against the pinned release.

Each subcommand has its own flags; see `{PROG} redact --help`."""

BANNER = "OPF redaction. Type '/exit' (or 'quit') to stop."
PROMPT = "text> "
EXIT_COMMANDS = frozenset({"/exit", "exit", "quit", ":q"})

# The reference's ANSI 256-colour palette for categories.
PALETTE = (39, 45, 82, 208, 201, 51, 99, 220, 118, 214)
RESET = "\x1b[0m"

# What a user can cause: reported as one line and exit status 1. Anything else is a bug and keeps its traceback.
USER_ERRORS = (checkpoint.CheckpointError, PrecisionError, env.InsufficientMemoryError, OSError, ValueError)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = _redact
    if arguments and arguments[0] in COMMANDS:
        command, arguments = COMMANDS[arguments[0]], arguments[1:]
    elif any(argument in ("-h", "--help") for argument in arguments):
        print(ROOT_HELP)
        return 0

    with warnings.catch_warnings():
        warnings.simplefilter("always", PrivacyFilterWarning)
        warnings.showwarning = _show_warning
        try:
            return command(arguments)
        except USER_ERRORS as error:
            print(f"error: {error}", file=sys.stderr)
            return 1


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    print(f"warning: {message}", file=sys.stderr)


# -------------------------------------------------------------------- redact


def parse_redact_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Redact text to remove private information.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    source = parser.add_argument_group("Input / Source")
    runtime = parser.add_argument_group("Model / Runtime")
    decode = parser.add_argument_group("Decode")
    output = parser.add_argument_group("Output")

    source.add_argument("text", nargs="?", help="Text to redact.")
    source.add_argument(
        "-f",
        "--text-file",
        action="append",
        default=None,
        help="Text file; each file is one input example (repeat for several files).",
    )

    _add_runtime_arguments(runtime)
    _add_decode_arguments(decode)

    output.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default="typed",
        help="typed: keep model categories.\nredacted: collapse all spans into one generic redacted label.",
    )
    output.add_argument(
        "--format",
        choices=("text", "json"),
        default=None,
        help="Print redacted text or the structured JSON result (default: json at the prompt, text otherwise).",
    )
    output.add_argument("--json-indent", type=int, default=2, help="Indentation of printed JSON (default: 2).")
    output.add_argument(
        "--no-print-color-coded-text",
        dest="print_color_coded_text",
        action="store_false",
        help="Do not print the ANSI colour-coded text section after JSON output.",
    )

    arguments = parser.parse_args(argv)
    if arguments.json_indent < 0:
        parser.error("--json-indent must be >= 0")
    return arguments


def _add_runtime_arguments(runtime: argparse._ArgumentGroup) -> None:
    """The flags that choose and place the model; shared by `redact` and `serve`."""
    runtime.add_argument(
        "--checkpoint",
        default=None,
        help=f"Checkpoint directory. Default: {checkpoint.ENV_VAR}, else ~/.opf/privacy_filter (downloaded if absent).",
    )
    runtime.add_argument("--no-download", action="store_true", help="Fail instead of downloading a missing checkpoint.")
    runtime.add_argument("--device", choices=tuple(DEVICES), default="gpu", help="Device to run on (default: gpu).")
    runtime.add_argument("--dtype", choices=DTYPES, default=None, help="Weight precision (default: as stored).")
    runtime.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Tokens per forward pass. Default: the checkpoint's default_n_ctx on the gpu, 4096 on the cpu.",
    )
    runtime.add_argument(
        "--window-policy",
        choices=POLICIES,
        default="original",
        help=(
            "How an input longer than --n-ctx is split.\n"
            "original: consecutive windows, as in the reference.\n"
            "halo: each window is surrounded by --halo tokens of context."
        ),
    )
    runtime.add_argument("--halo", type=int, default=None, help="Context tokens per side for --window-policy halo.")


def _add_decode_arguments(decode: argparse._ArgumentGroup) -> None:
    """The flags that turn logits into spans; shared by `redact` and `serve`."""
    decode.add_argument(
        "--decode-mode",
        choices=DECODE_MODES,
        default="viterbi",
        help="Decode token labels with constrained Viterbi or independent argmax.",
    )
    decode.add_argument(
        "--discard-overlapping-predicted-spans",
        action="store_true",
        help="Discard overlapping predicted spans per label.",
    )
    decode.add_argument(
        "--trim-whitespace",
        dest="trim_whitespace",
        action="store_true",
        default=True,
        help="Trim whitespace from predicted character spans (default).",
    )
    decode.add_argument(
        "--no-trim-whitespace",
        dest="trim_whitespace",
        action="store_false",
        help="Do not trim whitespace from predicted character spans.",
    )
    decode.add_argument(
        "--viterbi-calibration-path",
        default=None,
        help=(
            "Viterbi calibration JSON. Default: <checkpoint>/viterbi_calibration.json if present,\n"
            "else all-zero transition biases."
        ),
    )


def build_filter(arguments: argparse.Namespace) -> PrivacyFilter:
    location = checkpoint.resolve_checkpoint(
        arguments.checkpoint,
        download=not arguments.no_download,
        source=checkpoint.RELEASED,
        progress=checkpoint.stderr_progress(),
    )
    return PrivacyFilter(
        location,
        n_ctx=arguments.n_ctx,
        device=arguments.device,
        dtype=arguments.dtype,
        decode_mode=arguments.decode_mode,
        viterbi_calibration_path=arguments.viterbi_calibration_path,
        trim_whitespace=arguments.trim_whitespace,
        discard_overlapping_predicted_spans=arguments.discard_overlapping_predicted_spans,
        output_mode=arguments.output_mode,
        window_policy=arguments.window_policy,
        halo=arguments.halo,
    )


def _redact(argv: Sequence[str]) -> int:
    arguments = parse_redact_arguments(argv)
    at_prompt = not arguments.text and not arguments.text_file and sys.stdin.isatty()
    as_json = (arguments.format or ("json" if at_prompt else "text")) == "json"
    privacy_filter = build_filter(arguments)

    colours: dict[str, int] | None = None
    if as_json:
        if at_prompt:
            print(
                f"session: checkpoint={privacy_filter.checkpoint} device={privacy_filter.device} "
                f"encoding={privacy_filter.config.encoding} n_ctx={privacy_filter.n_ctx} "
                f"output_mode={privacy_filter.output_mode} dtype={privacy_filter.dtype}",
                file=sys.stderr,
            )
        if privacy_filter.output_mode != "redacted":
            colours = label_colours(privacy_filter.labels.span_names)

    for text in _inputs(arguments):
        started = time.perf_counter()
        result = privacy_filter.redact(text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not as_json:
            print(result.redacted_text, flush=True)
            continue
        print(summary_line(result.summary, latency_ms), file=sys.stderr)
        print(json.dumps(result.to_dict(), indent=arguments.json_indent, ensure_ascii=False))
        if arguments.print_color_coded_text and colours is not None:
            print(colour_legend(colours))
            print("color coded text:")
            print(colour_coded_text(result.text, result.detected_spans, colours))
        sys.stdout.flush()
    return 0


def _inputs(arguments: argparse.Namespace) -> Iterator[str]:
    """The texts to redact: the argument, else files, else piped lines, else the prompt."""
    if arguments.text:
        yield arguments.text
    elif arguments.text_file:
        for name in arguments.text_file:
            text = Path(name).expanduser().read_text(encoding="utf-8")
            if text:
                yield text
    elif not sys.stdin.isatty():
        for raw in sys.stdin:
            line = raw.rstrip("\r\n")
            if line.strip():
                yield line
    else:
        yield from _prompt()


def _prompt() -> Iterator[str]:
    print(BANNER)
    try:
        import readline  # noqa: F401  (gives input() line editing and history)
    except ImportError:
        print("warning: line editing is unavailable (no readline module); using plain input", file=sys.stderr)
    while True:
        try:
            line = input(PROMPT)
        except EOFError:
            print()
            return
        if line.strip().lower() in EXIT_COMMANDS:
            return
        if line:
            yield line


def summary_line(summary: Mapping[str, object], latency_ms: float) -> str:
    by_label = ", ".join(f"{label}:{count}" for label, count in summary["by_label"].items()) or "-"
    return (
        f"summary: output_mode={summary['output_mode']} spans={summary['span_count']} by_label={by_label} "
        f"latency_ms={latency_ms:.1f} decoded_mismatch={'yes' if summary['decoded_mismatch'] else 'no'}"
    )


def label_colours(span_names: Sequence[str]) -> dict[str, int]:
    """One palette colour per category, in label-space order, cycling if there are more categories than colours."""
    colours: dict[str, int] = {}
    for name in span_names:
        if name != BACKGROUND and name not in colours:
            colours[name] = PALETTE[len(colours) % len(PALETTE)]
    return colours


def colour_legend(colours: Mapping[str, int]) -> str:
    return "color legend: " + " | ".join(_paint(label, label, colours) for label in colours)


def colour_coded_text(text: str, spans: Sequence[DetectedSpan], colours: Mapping[str, int]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in spans:  # sorted and non-overlapping, as the pipeline returns them
        pieces.append(text[cursor : span.start])
        pieces.append(_paint(text[span.start : span.end], span.label, colours))
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _paint(text: str, label: str, colours: Mapping[str, int]) -> str:
    return f"\x1b[38;5;{colours[label]}m{text}{RESET}" if label in colours else text


# --------------------------------------------------------------------- serve


def parse_serve_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} serve",
        description=(
            "Answer detection requests on a local socket: text in, spans out.\n"
            "Listens on a Unix socket only the user can open, or on a loopback port. Never on the network."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    listen = parser.add_argument_group("Listen")
    where = listen.add_mutually_exclusive_group()
    where.add_argument("--socket", default=None, help="Unix socket to listen on (default: ~/.opf/opf-mlx.sock).")
    where.add_argument("--port", type=int, default=None, help="Listen on a loopback TCP port instead; 0 picks a free one.")
    listen.add_argument("--host", default=None, help="Loopback address for --port (default: 127.0.0.1).")
    listen.add_argument(
        "--idle-exit", type=float, default=None, metavar="SECONDS", help="Stop after this long without a request (default: never)."
    )
    listen.add_argument(
        "--max-request-bytes",
        type=int,
        default=serve.DEFAULT_MAX_REQUEST_BYTES,
        help=f"Refuse larger request bodies (default: {serve.DEFAULT_MAX_REQUEST_BYTES}).",
    )
    _add_runtime_arguments(parser.add_argument_group("Model / Runtime"))
    _add_decode_arguments(parser.add_argument_group("Decode"))

    arguments = parser.parse_args(argv)
    if arguments.host is not None and arguments.port is None:
        parser.error("--host needs --port")
    if arguments.idle_exit is not None and arguments.idle_exit <= 0:
        parser.error("--idle-exit must be positive")
    if arguments.max_request_bytes <= 0:
        parser.error("--max-request-bytes must be positive")
    arguments.output_mode = "typed"  # a caller that masks needs to know what each span is
    return arguments


def detector_for(privacy_filter: PrivacyFilter) -> serve.Detector:
    """The model as the service sees it: one text in, its spans out."""

    def detect(text: str) -> serve.Detection:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrivacyFilterWarning)  # it travels in the result instead
            result = privacy_filter.redact(text)
        if result.summary["decoded_mismatch"]:
            # Spans would index the decoded text, not the text the caller holds.
            raise ValueError("the text does not survive tokenization")
        return serve.Detection([(span.start, span.end, span.label) for span in result.detected_spans], result.warning)

    return detect


def _serve(argv: Sequence[str]) -> int:
    arguments = parse_serve_arguments(argv)
    # Claim the socket before loading the model: a second server gives up at once.
    claim = None if arguments.port is not None else serve.SocketClaim(arguments.socket or serve.default_socket_path())
    try:
        privacy_filter = build_filter(arguments)
        server = serve.DetectionServer(
            detector_for(privacy_filter),
            claim=claim,
            host=arguments.host or "127.0.0.1",
            port=arguments.port,
            info={
                "labels": [name for name in privacy_filter.labels.span_names if name != BACKGROUND],
                "device": privacy_filter.device,
                "dtype": privacy_filter.dtype,
                "n_ctx": privacy_filter.n_ctx,
            },
            max_request_bytes=arguments.max_request_bytes,
            idle_exit=arguments.idle_exit,
        )
    except BaseException:
        if claim is not None:
            claim.release()
        raise

    with server:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: server.stop())
        print(f"listening on {server.address}", flush=True)  # the model is loaded: listening means ready
        server.serve()
    return 0


# ---------------------------------------------------- env, download, verify


def _env(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} env", description="Report this machine's hardware, memory and library versions."
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    arguments = parser.parse_args(argv)

    machine = env.detect()
    location = checkpoint.checkpoint_location()
    installed = _is_installed(location)
    if arguments.json:
        print(json.dumps({**machine.to_dict(), "checkpoint": {"path": str(location), "installed": installed}}, indent=2))
    else:
        state = "installed" if installed else f"not installed; run `{PROG} download`"
        print(env.format_report(machine, extra_rows=[("checkpoint", f"{location} ({state})")]))
    return 0


def _is_installed(location: Path) -> bool:
    try:
        checkpoint.resolve_checkpoint(location, download=False)
    except checkpoint.CheckpointError:
        return False
    return True


def _checkpoint_parser(command: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"{PROG} {command}", description=description)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Checkpoint directory. Default: {checkpoint.ENV_VAR}, else ~/.opf/privacy_filter.",
    )
    return parser


def _download(argv: Sequence[str]) -> int:
    parser = _checkpoint_parser("download", "Fetch the pretrained checkpoint (pinned revision, verified).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files in the directory that are not the pinned release. Without this they are left alone.",
    )
    arguments = parser.parse_args(argv)
    source = checkpoint.RELEASED
    target = checkpoint.checkpoint_location(arguments.checkpoint)
    print(
        f"Downloading {source.repo} ({source.subdir}/, revision {source.revision[:12]}) from {source.base_url} "
        f"into {target}",
        file=sys.stderr,
    )
    checkpoint.download_checkpoint(
        target, source=source, progress=checkpoint.stderr_progress(), overwrite=arguments.force
    )
    print(target)
    return 0


def _verify(argv: Sequence[str]) -> int:
    arguments = _checkpoint_parser(
        "verify", "Check a checkpoint's files against the pinned release and its tensors against the architecture."
    ).parse_args(argv)
    target = checkpoint.checkpoint_location(arguments.checkpoint)

    matches = checkpoint.verify_checkpoint(target, source=checkpoint.RELEASED)
    rows = [
        (name, "ok" if same else "differs from the pinned release" if (target / name).is_file() else "missing")
        for name, same in matches.items()
    ]
    try:
        expected = expected_tensors(ModelConfig.from_file(target / "config.json"))
        validate_tensors(read_inventory(target), expected)
        rows.append(("tensors", f"ok ({len(expected)} tensors match the architecture)"))
        tensors_fit = True
    except (OSError, ValueError) as error:
        rows.append(("tensors", str(error).replace("\n", "\n    ")))
        tensors_fit = False

    width = max(len(name) for name, _ in rows)
    print(target)
    for name, status in rows:
        print(f"  {name:<{width}}  {status}")
    return 0 if tensors_fit and all(matches.values()) else 1


COMMANDS = {"redact": _redact, "serve": _serve, "env": _env, "download": _download, "verify": _verify}
