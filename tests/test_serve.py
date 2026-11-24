"""The local detection service (`opf_mlx.serve`), driven with fake detectors.

The service only moves text in and spans out, so nothing here loads a model.
The command that wires it to a real one is tested in test_cli.py.
"""

import ast
import http.client
import json
import re
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from opf_mlx import serve
from opf_mlx.serve import Detection, DetectionServer, ServeError, SocketClaim

GRIN = "\U0001f600"  # one code point, two UTF-16 code units
JSON = {"Content-Type": "application/json"}


def find_words(*words: str, label: str = "secret"):
    """A detector that flags every occurrence of the given words."""
    pattern = re.compile("|".join(re.escape(word) for word in words))

    def detect(text: str) -> Detection:
        return Detection([(match.start(), match.end(), label) for match in pattern.finditer(text)])

    return detect


@pytest.fixture
def short_dir():
    """A directory with a short path: a Unix socket path is limited to about 100 bytes."""
    path = Path(tempfile.mkdtemp(prefix="opf-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str) -> None:
        super().__init__("localhost", timeout=10)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def connect(address: str) -> http.client.HTTPConnection:
    if address.startswith("unix:"):
        return UnixConnection(address[len("unix:") :])
    host, port = address[len("http://") :].split(":")
    return http.client.HTTPConnection(host, int(port), timeout=10)


def call(address: str, method: str, path: str, body=None, headers=None) -> tuple[int, dict]:
    connection = connect(address)
    try:
        payload = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        connection.request(method, path, body=payload, headers=JSON if headers is None else headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def detect(address: str, texts, **extra) -> tuple[int, dict]:
    return call(address, "POST", "/v1/detect", {"texts": texts, **extra})


@contextmanager
def running(detector, **options):
    options.setdefault("port", 0)
    server = DetectionServer(detector, poll_interval=0.01, **options)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop()
        thread.join(5)
        server.close()


# ------------------------------------------------------------------ detecting


def test_each_text_gets_its_spans_in_order():
    with running(find_words("hunter2", "Alice")) as server:
        status, body = detect(server.address, ["pw hunter2", "nothing here", "Alice and Alice"])

    assert status == 200
    assert body == {
        "results": [
            {"spans": [{"start": 3, "end": 10, "label": "secret"}]},
            {"spans": []},
            {"spans": [{"start": 0, "end": 5, "label": "secret"}, {"start": 10, "end": 15, "label": "secret"}]},
        ]
    }


def test_no_texts_is_a_valid_request():
    with running(find_words("x")) as server:
        assert detect(server.address, []) == (200, {"results": []})


def test_offsets_count_code_points_unless_utf16_is_asked_for():
    text = f"{GRIN} key hunter2 {GRIN}{GRIN} hunter2"
    with running(find_words("hunter2")) as server:
        _, by_code_point = detect(server.address, [text])
        _, explicit = detect(server.address, [text], offsets="codepoint")
        _, by_utf16 = detect(server.address, [text], offsets="utf16")

    assert by_code_point == explicit
    assert [(s["start"], s["end"]) for s in by_code_point["results"][0]["spans"]] == [(6, 13), (17, 24)]
    spans = [(s["start"], s["end"]) for s in by_utf16["results"][0]["spans"]]
    assert spans == [(7, 14), (20, 27)]
    as_utf16 = text.encode("utf-16-le")
    assert [as_utf16[2 * a : 2 * b].decode("utf-16-le") for a, b in spans] == ["hunter2", "hunter2"]


def test_a_lone_surrogate_reaches_the_detector_as_a_replacement_character():
    # JavaScript strings can hold half a surrogate pair; JSON carries it as an escape.
    seen: list[str] = []

    def detector(text: str) -> Detection:
        seen.append(text)
        return find_words("hunter2")(text)

    with running(detector) as server:
        status, body = call(server.address, "POST", "/v1/detect", b'{"texts": ["a\\ud800 hunter2"], "offsets": "utf16"}')

    assert seen == ["a� hunter2"]
    assert (status, body) == (200, {"results": [{"spans": [{"start": 3, "end": 10, "label": "secret"}]}]})


def test_a_detector_warning_is_passed_on():
    with running(lambda text: Detection([], warning="split into 2 windows")) as server:
        assert detect(server.address, ["x"]) == (200, {"results": [{"spans": [], "warning": "split into 2 windows"}]})


def test_health_reports_the_api_version_and_what_the_caller_said_about_the_model():
    with running(find_words("x"), info={"labels": ["secret"], "dtype": "bfloat16"}) as server:
        status, body = call(server.address, "GET", "/health")

    assert (status, body) == (200, {"status": "ok", "api": 1, "labels": ["secret"], "dtype": "bfloat16"})


# ------------------------------------------------------------- bad requests


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"{not json", "request body is not valid JSON"),
        (b"\xff\xfe", "request body is not valid JSON"),
        (b"[]", "request body must be a JSON object"),
        (b"{}", '"texts" must be a list of strings'),
        (b'{"texts": "one"}', '"texts" must be a list of strings'),
        (b'{"texts": ["ok", 3]}', '"texts" must be a list of strings'),
        (b'{"texts": [], "offsets": "bytes"}', '"offsets" must be "codepoint" or "utf16"'),
        (b'{"texts": [], "text": "typo"}', 'unknown field: "text"'),
    ],
)
def test_a_malformed_request_is_refused_with_a_reason(body, message):
    with running(find_words("x")) as server:
        assert call(server.address, "POST", "/v1/detect", body) == (400, {"error": message})


def test_the_body_must_be_declared_as_json():
    with running(find_words("x")) as server:
        status, body = call(server.address, "POST", "/v1/detect", b'{"texts": []}', headers={"Content-Type": "text/plain"})

    assert (status, body) == (415, {"error": "Content-Type must be application/json"})


def test_a_request_without_a_length_is_refused():
    with running(find_words("x")) as server:
        connection = connect(server.address)
        connection.putrequest("POST", "/v1/detect")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        response = connection.getresponse()
        status, body = response.status, json.loads(response.read())
        connection.close()

    assert (status, body) == (411, {"error": "Content-Length is required"})


def test_a_request_larger_than_the_limit_is_refused_and_never_reaches_the_detector():
    seen: list[str] = []
    with running(lambda text: seen.append(text) or Detection([]), max_request_bytes=64) as server:
        status, body = detect(server.address, ["x" * 100])

    assert (status, body) == (413, {"error": "request body is larger than the limit of 64 bytes"})
    assert seen == []


def test_unknown_paths_and_methods_are_refused():
    with running(find_words("x")) as server:
        assert call(server.address, "GET", "/nowhere") == (404, {"error": "not found"})
        assert call(server.address, "POST", "/health", {}) == (405, {"error": "method not allowed"})
        assert call(server.address, "GET", "/v1/detect") == (405, {"error": "method not allowed"})
        assert call(server.address, "DELETE", "/v1/detect") == (405, {"error": "method not allowed"})


# -------------------------------------------------------- detector failures


def test_a_failing_detector_is_a_500_that_does_not_repeat_the_text_and_the_service_carries_on():
    def detector(text: str) -> Detection:
        if "boom" in text:
            raise RuntimeError(f"cannot process {text!r}")
        return find_words("hunter2")(text)

    with running(detector) as server:
        failed = detect(server.address, ["fine", "boom hunter2"])
        after = detect(server.address, ["hunter2"])

    assert failed == (500, {"error": "detection failed for text 2 of 2 (RuntimeError)"})
    assert after == (200, {"results": [{"spans": [{"start": 0, "end": 7, "label": "secret"}]}]})


@pytest.mark.parametrize("span", [(2, 2, "secret"), (-1, 2, "secret"), (0, 99, "secret"), (0, 1, 7), ("0", 1, "secret")])
def test_a_span_that_does_not_fit_the_text_is_a_500(span):
    with running(lambda text: Detection([span])) as server:
        assert detect(server.address, ["abc"]) == (500, {"error": "detection failed for text 1 of 1 (invalid span)"})


def test_nothing_about_a_request_is_printed(capfd):
    def detector(text: str) -> Detection:
        if "boom" in text:
            raise RuntimeError(text)
        return Detection([])

    with running(detector) as server:
        detect(server.address, ["hunter2 is the password"])
        detect(server.address, ["boom hunter2"])
        call(server.address, "POST", "/v1/detect", b'{"texts": ["hunter2"')
        call(server.address, "GET", "/hunter2")

    assert capfd.readouterr() == ("", "")


# -------------------------------------------------------------- unix socket


def test_a_unix_socket_is_private_to_the_user_and_removed_on_close(short_dir):
    path = short_dir / "s.sock"
    with running(find_words("hunter2"), socket_path=path, port=None) as server:
        assert server.address == f"unix:{path}"
        assert stat.S_ISSOCK(path.stat().st_mode)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert detect(server.address, ["hunter2"])[0] == 200

    assert not path.exists()


def test_a_socket_left_by_a_dead_server_is_replaced(short_dir):
    path = short_dir / "s.sock"
    leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leftover.bind(str(path))
    leftover.close()  # the file stays, and nothing listens on it

    with running(find_words("x"), socket_path=path, port=None) as server:
        assert call(server.address, "GET", "/health")[0] == 200


def test_a_second_server_on_the_same_socket_is_refused_and_the_first_keeps_working(short_dir):
    path = short_dir / "s.sock"
    with running(find_words("x"), socket_path=path, port=None) as server:
        with pytest.raises(ServeError, match=f"another opf-mlx serve already owns {re.escape(str(path))}"):
            DetectionServer(find_words("x"), socket_path=path)

        assert call(server.address, "GET", "/health")[0] == 200


def test_a_claim_is_taken_before_the_slow_part_and_handed_to_the_server(short_dir):
    path = short_dir / "s.sock"
    with SocketClaim(path) as claim:
        with pytest.raises(ServeError, match="already owns"):
            SocketClaim(path)
        with running(find_words("x"), claim=claim, port=None) as server:
            assert call(server.address, "GET", "/health")[0] == 200

    assert not path.exists()
    SocketClaim(path).release()  # free again


def test_a_path_that_is_not_a_socket_is_left_alone(short_dir):
    path = short_dir / "notes.txt"
    path.write_text("keep me")

    with pytest.raises(ServeError, match="exists and is not a socket"):
        DetectionServer(find_words("x"), socket_path=path)

    assert path.read_text() == "keep me"


def test_a_socket_path_too_long_for_the_system_is_refused_with_its_limit(short_dir):
    path = short_dir / ("x" * 120)

    with pytest.raises(ServeError, match=r"socket path is \d+ bytes; the system allows at most \d+"):
        DetectionServer(find_words("x"), socket_path=path)


def test_the_directory_of_the_socket_is_created(short_dir):
    path = short_dir / "new" / "s.sock"
    with running(find_words("x"), socket_path=path, port=None) as server:
        assert call(server.address, "GET", "/health")[0] == 200


# ----------------------------------------------------------------- tcp, idle


def test_tcp_listens_on_loopback_only():
    with pytest.raises(ServeError, match="only a loopback address"):
        DetectionServer(find_words("x"), host="0.0.0.0", port=0)
    with pytest.raises(ServeError, match="only a loopback address"):
        DetectionServer(find_words("x"), host="example.com", port=0)

    with running(find_words("x"), host="127.0.0.1", port=0) as server:
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+", server.address)


def test_exactly_one_transport_is_chosen(short_dir):
    with pytest.raises(ValueError, match="either a socket path or a port"):
        DetectionServer(find_words("x"))
    with pytest.raises(ValueError, match="either a socket path or a port"):
        DetectionServer(find_words("x"), socket_path=short_dir / "s.sock", port=0)


def test_serving_ends_after_the_idle_time_and_a_request_restarts_the_clock():
    server = DetectionServer(find_words("x"), port=0, idle_exit=0.3, poll_interval=0.01)
    thread = threading.Thread(target=server.serve, daemon=True)
    started = time.monotonic()
    thread.start()
    time.sleep(0.2)
    assert call(server.address, "GET", "/health")[0] == 200  # still up; the clock restarts here
    thread.join(5)
    elapsed = time.monotonic() - started
    server.close()

    assert not thread.is_alive()
    assert 0.5 <= elapsed < 3


def test_a_client_that_stops_talking_does_not_hold_the_service():
    with running(find_words("x"), request_timeout=0.2) as server:
        host, port = server.address[len("http://") :].split(":")
        stalled = socket.create_connection((host, int(port)))
        stalled.sendall(b"POST /v1/detect HTTP/1.0\r\nContent-Length: 50\r\n")  # and then nothing
        started = time.monotonic()
        status, _ = call(server.address, "GET", "/health")
        stalled.close()

    assert status == 200
    assert time.monotonic() - started < 3


def test_serving_is_separate_from_the_model():
    # Only the standard library: no MLX, no tokenizer, and nothing else from this package.
    tree = ast.parse(Path(serve.__file__).read_text())
    imported = {
        (node.module or "." * node.level) if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert {name.split(".")[0] for name in imported} <= sys.stdlib_module_names
