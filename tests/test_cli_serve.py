"""`opf-mlx serve`: the command that puts a real model behind `opf_mlx.serve`.

The checkpoints are rigged (see test_pipeline.py), so what the model finds in
a text can be read off its tokens.
"""

import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from helpers import RIGGED_LOGIT, rigged_checkpoint

from opf_mlx import cli, serve
from opf_mlx.cli import main

SENTENCE = "Alice was born on 1990-01-02."
ALICE = 100151
PERSON_AND_DATE = {
    ALICE: {"S-private_person": RIGGED_LOGIT},
    3204: {"B-private_date": RIGGED_LOGIT},
    15: {"I-private_date": RIGGED_LOGIT},
    12: {"I-private_date": RIGGED_LOGIT},
    2290: {"I-private_date": RIGGED_LOGIT},
    3286: {"E-private_date": RIGGED_LOGIT},
}
LABELS = [
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
]  # fmt: skip


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    return str(rigged_checkpoint(tmp_path_factory.mktemp("person-and-date"), PERSON_AND_DATE))


@pytest.fixture
def short_dir():
    """A Unix socket path is limited to about 100 bytes; pytest's temporary paths are longer."""
    path = Path(tempfile.mkdtemp(prefix="opf-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class Server:
    """`python -m opf_mlx serve ...` as a child process."""

    def __init__(self, *arguments: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "opf_mlx", "serve", *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )  # fmt: skip
        self.address = ""

    def wait_until_listening(self) -> str:
        line = self.process.stdout.readline()  # the one line it prints; an early exit gives ""
        assert line.startswith("listening on "), (line, self.process.stderr.read())
        self.address = line[len("listening on ") :].strip()
        return self.address

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        if self.address.startswith("unix:"):
            connection = http.client.HTTPConnection("localhost", timeout=30)
            connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.sock.connect(self.address[len("unix:") :])
        else:
            host, port = self.address[len("http://") :].split(":")
            connection = http.client.HTTPConnection(host, int(port), timeout=30)
        payload = None if body is None else json.dumps(body)
        connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status, data = response.status, json.loads(response.read())
        connection.close()
        return status, data

    def finish(self, signum: int | None = None) -> tuple[int, str, str]:
        if signum is not None:
            self.process.send_signal(signum)
        try:
            out, err = self.process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            raise
        return self.process.returncode, out, err

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc: object) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


# ------------------------------------------------------------- in process


def test_the_detector_reports_spans_of_the_text_it_was_given(checkpoint):
    arguments = cli.parse_serve_arguments(["--checkpoint", checkpoint])
    detect = cli.detector_for(cli.build_filter(arguments))

    assert detect(SENTENCE) == serve.Detection([(0, 5, "private_person"), (18, 28, "private_date")], None)
    assert detect("") == serve.Detection([], None)


def test_an_input_split_into_windows_is_reported_in_the_result_not_printed(checkpoint, capsys):
    arguments = cli.parse_serve_arguments(["--checkpoint", checkpoint, "--n-ctx", "4"])
    detect = cli.detector_for(cli.build_filter(arguments))

    detection = detect(SENTENCE)  # more than 4 tokens; a warning raised here would fail the test

    assert [label for _, _, label in detection.spans][:1] == ["private_person"]
    assert detection.warning.startswith("Input has 12 tokens but the context window holds 4 (n_ctx), so it was processed in 3 consecutive windows.")
    assert capsys.readouterr() == ("", "")


def test_serve_always_keeps_the_categories_and_takes_the_runtime_flags(checkpoint):
    arguments = cli.parse_serve_arguments(
        ["--checkpoint", checkpoint, "--device", "cpu", "--dtype", "float32", "--decode-mode", "argmax", "--no-download"]
    )
    filt = cli.build_filter(arguments)

    assert (filt.output_mode, filt.device, filt.dtype, filt.decode_mode) == ("typed", "cpu", "float32", "argmax")


def test_the_default_socket_is_in_the_users_opf_directory():
    assert serve.default_socket_path() == Path.home() / ".opf" / "opf-mlx.sock"
    assert cli.parse_serve_arguments([]).socket is None  # resolved when serving, after HOME is known


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--socket", "/tmp/x.sock", "--port", "0"], "not allowed with argument"),
        (["--host", "127.0.0.1"], "--host needs --port"),
        (["--idle-exit", "0"], "--idle-exit must be positive"),
        (["--max-request-bytes", "0"], "--max-request-bytes must be positive"),
        (["--output-mode", "redacted"], "unrecognized arguments"),
    ],
)
def test_flags_that_do_not_fit_together_are_usage_errors(arguments, message, capsys):
    with pytest.raises(SystemExit) as stop:
        main(["serve", *arguments])

    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_serve_help_lists_the_flags(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["serve", "--help"])

    assert stop.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--socket", "--port", "--host", "--idle-exit", "--max-request-bytes", "--checkpoint", "--device"):
        assert flag in out


def test_a_host_that_is_not_loopback_is_a_one_line_error(checkpoint, capsys):
    assert main(["serve", "--checkpoint", checkpoint, "--host", "0.0.0.0", "--port", "0"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == "error: serve never listens on the network: only a loopback address is accepted (got '0.0.0.0')\n"


def test_a_missing_checkpoint_is_a_one_line_error_and_leaves_no_socket(short_dir, capsys):
    path = short_dir / "s.sock"

    assert main(["serve", "--checkpoint", str(short_dir / "nowhere"), "--socket", str(path)]) == 1

    assert capsys.readouterr().err == f"error: Checkpoint directory not found: {short_dir / 'nowhere'}\n"
    assert not path.exists()
    serve.SocketClaim(path).release()  # and the path is free again


# -------------------------------------------------------- process-level


def test_serving_on_a_unix_socket_until_told_to_stop(checkpoint, short_dir):
    path = short_dir / "s.sock"
    with Server("--checkpoint", checkpoint, "--socket", str(path)) as server:
        assert server.wait_until_listening() == f"unix:{path}"
        health = server.request("GET", "/health")
        found = server.request("POST", "/v1/detect", {"texts": [SENTENCE, "nothing"], "offsets": "utf16"})
        code, out, err = server.finish(signal.SIGTERM)

    assert health == (200, {"status": "ok", "api": 1, "labels": LABELS, "device": "gpu", "dtype": "bfloat16", "n_ctx": 64})
    assert found == (
        200,
        {
            "results": [
                {
                    "spans": [
                        {"start": 0, "end": 5, "label": "private_person"},
                        {"start": 18, "end": 28, "label": "private_date"},
                    ]
                },
                {"spans": []},
            ]
        },
    )
    assert (code, out, err) == (0, "", "")
    assert not path.exists()


def test_serving_on_a_loopback_port_and_stopping_on_interrupt(checkpoint):
    with Server("--checkpoint", checkpoint, "--port", "0") as server:
        assert server.wait_until_listening().startswith("http://127.0.0.1:")
        status, body = server.request("POST", "/v1/detect", {"texts": ["Alice"]})
        code, _, err = server.finish(signal.SIGINT)

    assert (status, body) == (200, {"results": [{"spans": [{"start": 0, "end": 5, "label": "private_person"}]}]})
    assert (code, err) == (0, "")


def test_an_idle_server_stops_by_itself(checkpoint, short_dir):
    path = short_dir / "s.sock"
    with Server("--checkpoint", checkpoint, "--socket", str(path), "--idle-exit", "0.5") as server:
        server.wait_until_listening()
        code, _, err = server.finish()

    assert (code, err) == (0, "")
    assert not path.exists()


def test_a_second_server_on_the_same_socket_gives_up_at_once(checkpoint, short_dir):
    path = short_dir / "s.sock"
    with Server("--checkpoint", checkpoint, "--socket", str(path)) as first:
        first.wait_until_listening()
        second = subprocess.run(
            [sys.executable, "-m", "opf_mlx", "serve", "--checkpoint", str(short_dir / "not-even-read"), "--socket", str(path)],
            capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        still_up = first.request("GET", "/health")[0]
        first.finish(signal.SIGTERM)

    assert (second.returncode, second.stdout) == (1, "")
    assert second.stderr == f"error: another opf-mlx serve already owns {path}\n"
    assert still_up == 200


def test_the_socket_path_can_come_from_the_home_directory(checkpoint, short_dir):
    # HOME is where the default lives; a short one keeps the path within the system's limit.
    environment = {**os.environ, "HOME": str(short_dir)}
    process = subprocess.Popen(
        [sys.executable, "-m", "opf_mlx", "serve", "--checkpoint", checkpoint],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
    )  # fmt: skip
    try:
        line = process.stdout.readline()
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()

    assert line == f"listening on unix:{short_dir / '.opf' / 'opf-mlx.sock'}\n"
    assert not (short_dir / ".opf" / "opf-mlx.sock").exists()
