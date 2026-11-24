"""The `opf-mlx` command line, on rigged checkpoints (see test_pipeline.py).

Flags and output mirror the reference's `opf` command (`opf/__main__.py`), so
most expectations here are the reference's formats written out literally.
"""

import io
import json
import os
import pty
import re
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import RIGGED_LOGIT, rigged_checkpoint, serve_directory, write_calibration

from opf_mlx import checkpoint as checkpoint_module
from opf_mlx import cli
from opf_mlx.checkpoint import RemoteFile, Source, sha256_file
from opf_mlx.cli import main

SENTENCE = "Alice was born on 1990-01-02."
REDACTED = "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
ALICE, WAS, BORN = 100151, 673, 12275
PERSON_AND_DATE = {
    ALICE: {"S-private_person": RIGGED_LOGIT},
    3204: {"B-private_date": RIGGED_LOGIT},
    15: {"I-private_date": RIGGED_LOGIT},
    12: {"I-private_date": RIGGED_LOGIT},
    2290: {"I-private_date": RIGGED_LOGIT},
    3286: {"E-private_date": RIGGED_LOGIT},
}
OPEN_SPAN = {
    WAS: {"B-private_person": RIGGED_LOGIT},
    BORN: {"O": RIGGED_LOGIT, "E-private_person": RIGGED_LOGIT - 4},
}

RESULT = {
    "schema_version": 1,
    "summary": {
        "output_mode": "typed",
        "span_count": 2,
        "by_label": {"private_date": 1, "private_person": 1},
        "decoded_mismatch": False,
    },
    "text": SENTENCE,
    "detected_spans": [
        {"label": "private_person", "start": 0, "end": 5, "text": "Alice", "placeholder": "<PRIVATE_PERSON>"},
        {"label": "private_date", "start": 18, "end": 28, "text": "1990-01-02", "placeholder": "<PRIVATE_DATE>"},
    ],
    "redacted_text": REDACTED,
}
# The reference's palette, one colour per category in label-space order.
LEGEND = (
    "color legend: \x1b[38;5;39maccount_number\x1b[0m | \x1b[38;5;45mprivate_address\x1b[0m | "
    "\x1b[38;5;82mprivate_date\x1b[0m | \x1b[38;5;208mprivate_email\x1b[0m | \x1b[38;5;201mprivate_person\x1b[0m | "
    "\x1b[38;5;51mprivate_phone\x1b[0m | \x1b[38;5;99mprivate_url\x1b[0m | \x1b[38;5;220msecret\x1b[0m"
)
COLOURED = "\x1b[38;5;201mAlice\x1b[0m was born on \x1b[38;5;82m1990-01-02\x1b[0m."
JSON_OUTPUT = json.dumps(RESULT, indent=2) + "\n" + LEGEND + "\ncolor coded text:\n" + COLOURED + "\n"
SUMMARY = (
    r"summary: output_mode=typed spans=2 by_label=private_date:1, private_person:1 "
    r"latency_ms=\d+\.\d decoded_mismatch=no\n"
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    return str(rigged_checkpoint(tmp_path_factory.mktemp("person-and-date"), PERSON_AND_DATE))


@pytest.fixture(scope="module")
def open_span_checkpoint(tmp_path_factory) -> str:
    return str(rigged_checkpoint(tmp_path_factory.mktemp("open-span"), OPEN_SPAN))


# -------------------------------------------------------------------- redact


def test_text_argument_prints_the_redacted_text(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, SENTENCE]) == 0

    out, err = capsys.readouterr()
    assert out == REDACTED + "\n"
    assert err == ""


def test_redact_can_be_named_explicitly(checkpoint, capsys):
    assert main(["redact", "--checkpoint", checkpoint, SENTENCE]) == 0

    assert capsys.readouterr().out == REDACTED + "\n"


def test_checkpoint_comes_from_the_environment_when_no_flag_is_given(checkpoint, capsys, monkeypatch):
    monkeypatch.setenv("OPF_CHECKPOINT", checkpoint)

    assert main([SENTENCE]) == 0

    assert capsys.readouterr().out == REDACTED + "\n"


def test_json_format_prints_the_result_a_legend_and_coloured_text(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, "--format", "json", SENTENCE]) == 0

    out, err = capsys.readouterr()
    assert out == JSON_OUTPUT
    assert re.fullmatch(SUMMARY, err), err


def test_the_coloured_section_can_be_left_out(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, "--format", "json", "--no-print-color-coded-text", SENTENCE]) == 0

    assert json.loads(capsys.readouterr().out) == RESULT


def test_json_indent_is_configurable(checkpoint, capsys):
    arguments = ["--checkpoint", checkpoint, "--format", "json", "--no-print-color-coded-text", "--json-indent", "0"]

    assert main([*arguments, SENTENCE]) == 0

    assert capsys.readouterr().out == json.dumps(RESULT, indent=0) + "\n"


def test_a_negative_json_indent_is_a_usage_error(checkpoint, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--checkpoint", checkpoint, "--json-indent", "-1", SENTENCE])

    assert stop.value.code == 2
    assert "--json-indent must be >= 0" in capsys.readouterr().err


def test_redacted_output_mode_hides_categories_and_has_no_legend(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, "--output-mode", "redacted", "--format", "json", SENTENCE]) == 0

    out, err = capsys.readouterr()
    assert json.loads(out)["redacted_text"] == "<REDACTED> was born on <REDACTED>."
    assert "summary: output_mode=redacted spans=2 by_label=redacted:2 " in err


def test_a_text_without_spans_summarizes_as_a_dash(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, "--format", "json", "--no-print-color-coded-text", "It was."]) == 0

    assert "summary: output_mode=typed spans=0 by_label=- " in capsys.readouterr().err


def test_each_text_file_is_one_example(checkpoint, capsys, tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("Alice was born.\nAlice left.\n", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Alice left.", encoding="utf-8")

    assert main(["--checkpoint", checkpoint, "-f", str(first), "--text-file", str(empty), "-f", str(second)]) == 0

    # The first file is one text of two lines; the empty file is skipped.
    assert capsys.readouterr().out == "<PRIVATE_PERSON> was born.\n<PRIVATE_PERSON> left.\n\n<PRIVATE_PERSON> left.\n"


def test_each_line_of_standard_input_is_one_example(checkpoint, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Alice was born.\r\n\n   \nAlice left.\n"))

    assert main(["--checkpoint", checkpoint]) == 0

    assert capsys.readouterr().out == "<PRIVATE_PERSON> was born.\n<PRIVATE_PERSON> left.\n"


def test_decode_mode_flag(open_span_checkpoint, capsys):
    assert main(["--checkpoint", open_span_checkpoint, SENTENCE]) == 0
    assert capsys.readouterr().out == "Alice <PRIVATE_PERSON> on 1990-01-02.\n"

    assert main(["--checkpoint", open_span_checkpoint, "--decode-mode", "argmax", SENTENCE]) == 0
    assert capsys.readouterr().out == "Alice <PRIVATE_PERSON> born on 1990-01-02.\n"


def test_viterbi_calibration_path_flag(open_span_checkpoint, capsys, tmp_path):
    calibration = write_calibration(tmp_path / "calibration.json", transition_bias_inside_to_end=-30.0)

    assert main(["--checkpoint", open_span_checkpoint, "--viterbi-calibration-path", str(calibration), SENTENCE]) == 0

    assert capsys.readouterr().out == SENTENCE + "\n"


def test_whitespace_trimming_flags(open_span_checkpoint, capsys):
    assert main(["--checkpoint", open_span_checkpoint, "--no-trim-whitespace", SENTENCE]) == 0
    assert capsys.readouterr().out == "Alice<PRIVATE_PERSON> on 1990-01-02.\n"

    assert main(["--checkpoint", open_span_checkpoint, "--trim-whitespace", SENTENCE]) == 0
    assert capsys.readouterr().out == "Alice <PRIVATE_PERSON> on 1990-01-02.\n"


def test_a_split_input_is_reported_on_standard_error(checkpoint, capsys):
    assert main(["--checkpoint", checkpoint, "--n-ctx", "5", SENTENCE]) == 0

    out, err = capsys.readouterr()
    assert out == REDACTED + "\n"
    assert err == (
        "warning: Input has 12 tokens but the context window holds 5 (n_ctx), so it was processed in 3 consecutive "
        "windows. Nothing was dropped, but tokens near a window boundary see less context than they would in one "
        "pass, which can change predictions there.\n"
    )


def test_every_runtime_flag_reaches_the_filter(checkpoint, tmp_path):
    calibration = write_calibration(tmp_path / "calibration.json")
    arguments = cli.parse_redact_arguments(
        [
            "--checkpoint", checkpoint, "--device", "cpu", "--dtype", "float32", "--n-ctx", "7",
            "--decode-mode", "argmax", "--viterbi-calibration-path", str(calibration), "--no-trim-whitespace",
            "--discard-overlapping-predicted-spans", "--output-mode", "redacted", "--window-policy", "halo",
            "--halo", "3", "--no-download",
        ]  # fmt: skip
    )

    filt = cli.build_filter(arguments)

    assert (filt.device, filt.dtype, filt.n_ctx, filt.decode_mode) == ("cpu", "float32", 7, "argmax")
    assert (filt.trim_whitespace, filt.discard_overlapping_predicted_spans) == (False, True)
    assert (filt.output_mode, filt.window_policy, filt.halo) == ("redacted", "halo", 3)
    assert str(filt.checkpoint) == checkpoint


def test_defaults_are_the_references(checkpoint):
    filt = cli.build_filter(cli.parse_redact_arguments(["--checkpoint", checkpoint]))

    assert (filt.device, filt.dtype, filt.n_ctx, filt.decode_mode) == ("gpu", "bfloat16", 64, "viterbi")
    assert (filt.trim_whitespace, filt.discard_overlapping_predicted_spans) == (True, False)
    assert (filt.output_mode, filt.window_policy) == ("typed", "original")


def test_a_missing_checkpoint_is_a_one_line_error(capsys, tmp_path):
    missing = tmp_path / "nowhere"

    assert main(["--checkpoint", str(missing), SENTENCE]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"error: Checkpoint directory not found: {missing}\n"


def test_an_unreadable_text_file_is_a_one_line_error(checkpoint, capsys, tmp_path):
    assert main(["--checkpoint", checkpoint, "-f", str(tmp_path / "missing.txt")]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and "missing.txt" in err and "Traceback" not in err


def test_an_unknown_flag_is_a_usage_error(checkpoint, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--checkpoint", checkpoint, "--device", "cuda", SENTENCE])

    assert stop.value.code == 2
    assert "invalid choice: 'cuda'" in capsys.readouterr().err


def test_help_lists_the_subcommands(capsys):
    assert main(["--help"]) == 0

    out = capsys.readouterr().out
    for command in ("redact", "serve", "env", "download", "verify"):
        assert f"\n  {command} " in out


def test_redact_help_lists_the_flags(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["redact", "--help"])

    assert stop.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--checkpoint", "--n-ctx", "--device", "--dtype", "--window-policy", "--decode-mode", "--format"):
        assert flag in out


# ---------------------------------------------------------- process-level


def run_module(*arguments: str, **options) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "opf_mlx", *arguments], capture_output=True, text=True, timeout=120, **options
    )


def test_the_package_runs_as_a_module(checkpoint):
    done = run_module("--checkpoint", checkpoint, SENTENCE)

    assert (done.returncode, done.stdout, done.stderr) == (0, REDACTED + "\n", "")


def test_the_console_script_is_installed(checkpoint):
    script = Path(sys.executable).parent / "opf-mlx"
    if not script.exists():
        pytest.skip("the opf-mlx console script is not installed in this environment")

    done = subprocess.run([str(script), "--checkpoint", checkpoint, SENTENCE], capture_output=True, text=True, timeout=120)

    assert (done.returncode, done.stdout, done.stderr) == (0, REDACTED + "\n", "")


def test_piped_input_is_read_line_by_line(checkpoint):
    done = run_module("--checkpoint", checkpoint, input="Alice was born.\nAlice left.\n")

    assert (done.returncode, done.stdout) == (0, "<PRIVATE_PERSON> was born.\n<PRIVATE_PERSON> left.\n")


def test_errors_exit_with_status_1(tmp_path):
    done = run_module("--checkpoint", str(tmp_path / "nowhere"), SENTENCE)

    assert done.returncode == 1
    assert done.stderr == f"error: Checkpoint directory not found: {tmp_path / 'nowhere'}\n"


def test_a_terminal_gets_a_prompt_and_json(checkpoint):
    # Standard input is a pseudo-terminal, as it is when a person runs the command.
    terminal, child_side = pty.openpty()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "opf_mlx", "--checkpoint", checkpoint],
            stdin=child_side,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(child_side)
        os.write(terminal, (SENTENCE + "\n\n/exit\n").encode())
        try:
            out, err = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            pytest.fail("the prompt did not exit on /exit")
    finally:
        os.close(terminal)

    assert process.returncode == 0
    # A banner, a prompt per line read (the empty line is ignored), and JSON by default.
    assert out == "OPF redaction. Type '/exit' (or 'quit') to stop.\ntext> " + JSON_OUTPUT + "text> text> "
    session, summary = err.split("\n", 1)
    assert session == (
        f"session: checkpoint={checkpoint} device=gpu encoding=o200k_base n_ctx=64 output_mode=typed dtype=bfloat16"
    )
    assert re.fullmatch(SUMMARY, summary), summary


# ----------------------------------------------------------------------- env


def test_env_reports_the_machine_and_the_checkpoint(checkpoint, capsys, monkeypatch):
    monkeypatch.setenv("OPF_CHECKPOINT", checkpoint)

    assert main(["env"]) == 0

    out = capsys.readouterr().out
    labels = [line.split("  ")[0] for line in out.splitlines() if line]
    assert labels == [
        "chip", "cpu cores", "gpu", "unified memory", "usable by MLX", "float32 matmul",
        "macOS", "python", "mlx", "numpy", "tiktoken", "checkpoint",
    ]  # fmt: skip
    assert "float32 matmul  full precision on the gpu" in out
    assert f"checkpoint      {checkpoint} (installed)" in out


def test_env_says_when_no_checkpoint_is_installed(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("OPF_CHECKPOINT", str(tmp_path / "nowhere"))

    assert main(["env"]) == 0

    assert f"checkpoint      {tmp_path / 'nowhere'} (not installed; run `opf-mlx download`)" in capsys.readouterr().out


def test_env_as_json(checkpoint, capsys, monkeypatch):
    monkeypatch.setenv("OPF_CHECKPOINT", checkpoint)

    assert main(["env", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["memory_total"] == os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert report["checkpoint"] == {"path": checkpoint, "installed": True}
    assert 0 <= report["float32_matmul_error"] < 1e-5


# --------------------------------------------------------- download, verify


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A tiny checkpoint served the way the hub serves the real one, pinned as the released source."""
    root = tmp_path / "hub"
    files = rigged_checkpoint(root / "acme" / "tiny" / "resolve" / "rev1" / "original", PERSON_AND_DATE)
    write_calibration(files / "viterbi_calibration.json")
    pinned = {path.name: RemoteFile(path.stat().st_size, sha256_file(path)) for path in sorted(files.iterdir())}
    with serve_directory(root) as (url, _):
        monkeypatch.setattr(checkpoint_module, "RELEASED", Source(url, "acme/tiny", "rev1", "original", pinned))
        yield files


def test_download_fetches_the_pinned_release(release, capsys, tmp_path):
    target = tmp_path / "installed"

    assert main(["download", "--checkpoint", str(target)]) == 0

    out, err = capsys.readouterr()
    assert out == f"{target}\n"
    assert "Downloading acme/tiny (original/, revision rev1)" in err
    for name in ("config.json", "model.safetensors", "viterbi_calibration.json"):
        assert (target / name).read_bytes() == (release / name).read_bytes()


def test_download_does_not_replace_a_different_checkpoint(release, capsys, tmp_path):
    target = tmp_path / "installed"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"someone else's weights")

    assert main(["download", "--checkpoint", str(target)]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines()[-1].startswith(f"error: {target} already holds model.safetensors, and it is not the pinned")
    assert (target / "model.safetensors").read_bytes() == b"someone else's weights"


def test_download_replaces_files_when_forced(release, capsys, tmp_path):
    target = tmp_path / "installed"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"someone else's weights")

    assert main(["download", "--checkpoint", str(target), "--force"]) == 0

    assert (target / "model.safetensors").read_bytes() == (release / "model.safetensors").read_bytes()


def test_a_downloaded_checkpoint_verifies_and_runs(release, capsys, tmp_path):
    target = tmp_path / "installed"
    main(["download", "--checkpoint", str(target)])
    capsys.readouterr()

    assert main(["verify", "--checkpoint", str(target)]) == 0
    assert capsys.readouterr().out == (
        f"{target}\n"
        "  config.json               ok\n"
        "  model.safetensors         ok\n"
        "  viterbi_calibration.json  ok\n"
        "  tensors                   ok (29 tensors match the architecture)\n"
    )

    assert main(["--checkpoint", str(target), SENTENCE]) == 0
    assert capsys.readouterr().out == REDACTED + "\n"


def test_verify_fails_on_a_changed_or_missing_file(release, capsys, tmp_path):
    target = tmp_path / "installed"
    main(["download", "--checkpoint", str(target)])
    (target / "config.json").write_text((target / "config.json").read_text() + " ")
    (target / "viterbi_calibration.json").unlink()
    capsys.readouterr()

    assert main(["verify", "--checkpoint", str(target)]) == 1

    assert capsys.readouterr().out == (
        f"{target}\n"
        "  config.json               differs from the pinned release\n"
        "  model.safetensors         ok\n"
        "  viterbi_calibration.json  missing\n"
        "  tensors                   ok (29 tensors match the architecture)\n"
    )


def test_verify_reports_tensors_that_do_not_fit_the_architecture(release, capsys, tmp_path):
    target = tmp_path / "installed"
    main(["download", "--checkpoint", str(target)])
    config = json.loads((target / "config.json").read_text())
    (target / "config.json").write_text(json.dumps({**config, "num_hidden_layers": 3}))
    capsys.readouterr()

    assert main(["verify", "--checkpoint", str(target)]) == 1

    out = capsys.readouterr().out
    # A third layer has 13 tensors, and the file has none of them.
    assert "  tensors                   checkpoint does not match the model (13 problem(s)):\n" in out
    assert "\n      missing tensor: block.2.attn.norm.scale\n" in out


def test_intact_files_that_do_not_fit_together_fail_verification(release, capsys, monkeypatch, tmp_path):
    target = tmp_path / "installed"
    main(["download", "--checkpoint", str(target)])
    # Pin a release whose config asks for three layers while its weights hold two.
    config = target / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "num_hidden_layers": 3}))
    pinned = dict(checkpoint_module.RELEASED.files)
    pinned["config.json"] = RemoteFile(config.stat().st_size, sha256_file(config))
    monkeypatch.setattr(checkpoint_module, "RELEASED", Source("http://unused", "acme/tiny", "rev2", "original", pinned))
    capsys.readouterr()

    assert main(["verify", "--checkpoint", str(target)]) == 1

    out = capsys.readouterr().out
    assert "  config.json               ok\n" in out
    assert "  tensors                   checkpoint does not match the model (13 problem(s)):\n" in out


def test_redacting_downloads_the_default_checkpoint_on_first_use(release, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    assert main([SENTENCE]) == 0

    out, err = capsys.readouterr()
    assert out == REDACTED + "\n"
    assert f"No checkpoint at {tmp_path / 'home' / '.opf' / 'privacy_filter'}. Downloading acme/tiny" in err
    assert (tmp_path / "home" / ".opf" / "privacy_filter" / "model.safetensors").is_file()


def test_downloading_can_be_forbidden(release, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    assert main(["--no-download", SENTENCE]) == 1

    assert "download is disabled" in capsys.readouterr().err
    assert not (tmp_path / "home" / ".opf").exists()
