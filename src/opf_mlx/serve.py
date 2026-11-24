"""A local detection service: text in, spans out.

    GET  /health     -> {"status": "ok", "api": 1, ...what the caller said about the model}
    POST /v1/detect  {"texts": ["...", ...], "offsets": "codepoint" | "utf16"}
                     -> {"results": [{"spans": [{"start", "end", "label"}, ...]}, ...]}

This module is transport only. It is handed a `Detector`, a function from one
text to its spans, and knows nothing about the model behind it; the command
line builds that function (`opf_mlx.cli`). It uses the standard library alone.

What it promises:

- It listens on a Unix socket that only the user can open, or on a loopback
  TCP port. It never listens on the network.
- It handles one request at a time, in the thread that called `serve`, so the
  model is never entered twice at once.
- It never prints, logs or repeats the text it is sent. An error names a
  position ("text 2 of 5") and a type, nothing more.
- A request is answered in full or refused with a reason. Too large is 413,
  not a truncated result.

Offsets count code points unless the caller asks for UTF-16 code units, which
is what a JavaScript string indexes by.
"""

from __future__ import annotations

import fcntl
import http.server
import ipaddress
import json
import os
import re
import socket
import socketserver
import stat
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import BinaryIO, Callable, Mapping, Sequence

API_VERSION = 1
OFFSETS: tuple[str, ...] = ("codepoint", "utf16")
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 30.0

# `sockaddr_un.sun_path` holds this many bytes, one of them the terminator.
MAX_SOCKET_PATH_BYTES = (104 if sys.platform == "darwin" else 108) - 1

# How much of a refused body is read and thrown away so the reply arrives
# before the connection closes. Past this the client may see a reset instead.
_DRAIN_LIMIT = 1024 * 1024
_SURROGATES = re.compile("[\ud800-\udfff]")


class ServeError(ValueError):
    """The service cannot start as asked: a problem with the address, not with the code."""


def default_socket_path() -> Path:
    """`~/.opf/opf-mlx.sock`, beside the default checkpoint."""
    return Path.home() / ".opf" / "opf-mlx.sock"


@dataclass(frozen=True)
class Detection:
    """What a detector found in one text.

    spans: `(start, end, label)` in code points of the text the detector was
        given, `end` exclusive.
    warning: something the caller should know that is not an error, such as an
        input that was processed in several windows.
    """

    spans: Sequence[tuple[int, int, str]]
    warning: str | None = None


Detector = Callable[[str], Detection]


class SocketClaim:
    """The right to listen on a Unix socket path.

    Taken before the model loads, so that of two servers started together the
    second gives up at once instead of loading 3 GB first. A lock file beside
    the socket says who owns the path; a socket file whose owner is gone is
    removed, since nothing can be listening on it.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().absolute()
        length = len(os.fsencode(self.path))
        if length > MAX_SOCKET_PATH_BYTES:
            raise ServeError(
                f"socket path is {length} bytes; the system allows at most {MAX_SOCKET_PATH_BYTES}: {self.path}"
            )
        if self._kind() == "other":
            raise ServeError(f"{self.path} exists and is not a socket")

        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock: int | None = os.open(f"{self.path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lock)
            self._lock = None
            raise ServeError(f"another opf-mlx serve already owns {self.path}") from None
        if self._kind() == "socket":
            os.unlink(self.path)  # left behind by a server that is gone

    def _kind(self) -> str | None:
        try:
            mode = os.lstat(self.path).st_mode
        except FileNotFoundError:
            return None
        return "socket" if stat.S_ISSOCK(mode) else "other"

    def release(self) -> None:
        """Remove the socket and give up the path. Safe to call twice."""
        if self._lock is None:
            return
        if self._kind() == "socket":
            os.unlink(self.path)
        os.close(self._lock)
        self._lock = None

    def __enter__(self) -> "SocketClaim":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class DetectionServer:
    """Serve a detector over HTTP on a Unix socket or a loopback port.

    detect: the detector. Called with one text at a time, from the thread that
        runs `serve`.
    socket_path / claim: listen on a Unix socket; `claim` is a `SocketClaim`
        taken earlier for the same purpose.
    port: listen on TCP instead, on `host`, which must be a loopback address.
        Port 0 picks a free one; `address` then says which.
    info: extra fields for `/health`, such as the model's labels.
    max_request_bytes: larger request bodies are refused with 413.
    idle_exit: `serve` returns after this many seconds without a request.
    request_timeout: seconds a client may take to send its request.
    """

    def __init__(
        self,
        detect: Detector,
        *,
        socket_path: str | os.PathLike[str] | None = None,
        claim: SocketClaim | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        info: Mapping[str, object] | None = None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        idle_exit: float | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        poll_interval: float = 0.25,
    ) -> None:
        if sum(option is not None for option in (socket_path, claim, port)) != 1:
            raise ValueError("choose either a socket path or a port to listen on")
        if max_request_bytes <= 0:
            raise ValueError(f"max_request_bytes must be positive (got {max_request_bytes})")
        if idle_exit is not None and idle_exit <= 0:
            raise ValueError(f"idle_exit must be positive (got {idle_exit})")
        if request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive (got {request_timeout})")

        self._detect = detect
        self._info = dict(info or {})
        self.max_request_bytes = max_request_bytes
        self._idle_exit = idle_exit
        self._stopping = False
        self._last_activity = time.monotonic()
        handler = type("Handler", (_Handler,), {"service": self, "timeout": request_timeout})

        self._claim: SocketClaim | None = None
        if port is not None:
            address = _loopback(host)
            family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            server_class = type("Server", (_TCPServer,), {"address_family": family})
            self._server: socketserver.BaseServer = server_class((str(address), port), handler)
            bound = self._server.server_address[1]
            self.address = f"http://[{address}]:{bound}" if address.version == 6 else f"http://{address}:{bound}"
        else:
            self._claim = claim if claim is not None else SocketClaim(socket_path)
            previous_umask = os.umask(0o177)  # the socket is created readable and writable by the user only
            try:
                self._server = _UnixServer(str(self._claim.path), handler)
            except OSError:
                self._claim.release()
                raise
            finally:
                os.umask(previous_umask)
            self.address = f"unix:{self._claim.path}"
        self._server.timeout = poll_interval

    def serve(self) -> None:
        """Answer requests until `stop` is called or the service has been idle for `idle_exit` seconds."""
        self._last_activity = time.monotonic()
        while not self._stopping:
            self._server.handle_request()  # returns after one request, or after `poll_interval` without one
            if self._idle_exit is not None and time.monotonic() - self._last_activity >= self._idle_exit:
                return

    def stop(self) -> None:
        """Make `serve` return. May be called from a signal handler or another thread."""
        self._stopping = True

    def close(self) -> None:
        """Stop listening and remove the socket. Safe to call twice."""
        self._server.server_close()
        if self._claim is not None:
            self._claim.release()

    def __enter__(self) -> "DetectionServer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------- what the handler calls

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def health(self) -> dict[str, object]:
        return {"status": "ok", "api": API_VERSION, **self._info}

    def detect_request(self, headers: Mapping[str, str], body: BinaryIO) -> tuple[int, dict[str, object]]:
        """Answer one `POST /v1/detect`: an HTTP status and a JSON body."""
        declared = headers.get("Content-Length")
        if declared is None:
            return 411, {"error": "Content-Length is required"}
        try:
            length = int(declared)
        except ValueError:
            length = -1
        if length < 0:
            return 400, {"error": "Content-Length is not a length"}
        if length > self.max_request_bytes:
            _drain(body, length)
            return 413, {"error": f"request body is larger than the limit of {self.max_request_bytes} bytes"}
        if (headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            _drain(body, length)
            return 415, {"error": "Content-Type must be application/json"}

        raw = body.read(length)
        if len(raw) != length:
            return 400, {"error": "request body is shorter than its Content-Length"}
        try:
            request = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            return 400, {"error": "request body is not valid JSON"}
        if not isinstance(request, dict):
            return 400, {"error": "request body must be a JSON object"}
        unknown = sorted(set(request) - {"texts", "offsets"})
        if unknown:
            return 400, {"error": f'unknown field: "{unknown[0]}"'}
        texts = request.get("texts")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            return 400, {"error": '"texts" must be a list of strings'}
        offsets = request.get("offsets", "codepoint")
        if offsets not in OFFSETS:
            return 400, {"error": '"offsets" must be "codepoint" or "utf16"'}

        results: list[dict[str, object]] = []
        for index, text in enumerate(texts):
            try:
                results.append(self._detect_one(text, offsets))
            except _InvalidSpan:
                reason = "invalid span"
            except Exception as error:  # the detector is someone else's code; say only what kind of failure
                reason = type(error).__name__
            else:
                continue
            return 500, {"error": f"detection failed for text {index + 1} of {len(texts)} ({reason})"}
        return 200, {"results": results}

    def _detect_one(self, text: str, offsets: str) -> dict[str, object]:
        # Half a surrogate pair cannot be tokenized. It becomes U+FFFD here, one
        # code point for one, so every offset still fits the caller's string.
        clean = _SURROGATES.sub("�", text)
        detection = self._detect(clean)
        spans: list[tuple[int, int, str]] = []
        for span in detection.spans:
            try:
                start, end, label = span
            except (TypeError, ValueError):
                raise _InvalidSpan from None
            if type(start) is not int or type(end) is not int or not isinstance(label, str):
                raise _InvalidSpan
            if not 0 <= start < end <= len(clean):
                raise _InvalidSpan
            spans.append((start, end, label))
        if offsets == "utf16":
            spans = _to_utf16(clean, spans)
        result: dict[str, object] = {"spans": [{"start": start, "end": end, "label": label} for start, end, label in spans]}
        if detection.warning:
            result["warning"] = detection.warning
        return result


class _InvalidSpan(Exception):
    pass


def _to_utf16(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Convert code point offsets into UTF-16 code unit offsets."""
    if text.isascii():
        return spans
    units: dict[int, int] = {}
    position = count = 0
    for point in sorted({offset for start, end, _ in spans for offset in (start, end)}):
        count += len(text[position:point].encode("utf-16-le")) // 2
        position = point
        units[point] = count
    return [(units[start], units[end], label) for start, end, label in spans]


def _drain(body: BinaryIO, length: int) -> None:
    remaining = min(length, _DRAIN_LIMIT)
    while remaining > 0:
        chunk = body.read(min(remaining, 65536))
        if not chunk:
            return
        remaining -= len(chunk)


def _loopback(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        address = ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise ServeError(f"serve never listens on the network: only a loopback address is accepted (got {host!r})")
    return address


class _Handler(http.server.BaseHTTPRequestHandler):
    """One request per connection. Every reply is JSON, and none repeats what was sent."""

    service: DetectionServer
    server_version = "opf-mlx"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass  # a request line or an error from the base class could carry text; print neither

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # The base class would answer in HTML and quote the request line.
        self._reply(code, {"error": HTTPStatus(code).phrase.lower()})

    def _reply(self, status: int, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.service.touch()
        if self.path == "/health":
            self._reply(200, self.service.health())
        elif self.path == "/v1/detect":
            self._not_allowed()
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        self.service.touch()
        if self.path == "/v1/detect":
            self._reply(*self.service.detect_request(self.headers, self.rfile))
        elif self.path == "/health":
            self._not_allowed()
        else:
            self._reply(404, {"error": "not found"})
        self.service.touch()  # a long detection is not idle time

    def _not_allowed(self) -> None:
        self._reply(405, {"error": "method not allowed"})

    do_PUT = do_PATCH = do_DELETE = _not_allowed


class _Quiet:
    request_queue_size = 64

    def handle_error(self, request: object, client_address: object) -> None:
        error = sys.exception()
        if isinstance(error, (ConnectionError, TimeoutError)):
            return  # the client went away; nothing to report
        print(f"opf-mlx serve: internal error ({type(error).__name__})", file=sys.stderr)


class _UnixServer(_Quiet, socketserver.UnixStreamServer):
    pass


class _TCPServer(_Quiet, socketserver.TCPServer):
    allow_reuse_address = True
