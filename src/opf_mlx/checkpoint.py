"""Where the checkpoint lives and how it gets there.

Resolution order is the reference's: an explicit path, then `OPF_CHECKPOINT`,
then `~/.opf/privacy_filter`, which is downloaded on first use. The layout is
also the reference's (the files of the repository's `original/` directory), so
this port and the PyTorch reference can share one copy.

The download is pinned: a fixed repository revision, and a size and SHA-256 for
every file. A file is only moved into place after its hash matches.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

ENV_VAR = "OPF_CHECKPOINT"

Progress = Callable[[str, int, int], None]


class CheckpointError(RuntimeError):
    """The checkpoint could not be found, fetched or verified."""


@dataclass(frozen=True)
class RemoteFile:
    size: int
    sha256: str


@dataclass(frozen=True)
class Source:
    """A set of files at a pinned revision of a Hugging Face model repository."""

    base_url: str
    repo: str
    revision: str
    subdir: str
    files: Mapping[str, RemoteFile]

    def url(self, name: str) -> str:
        return f"{self.base_url}/{self.repo}/resolve/{self.revision}/{self.subdir}/{name}"


RELEASED = Source(
    base_url="https://huggingface.co",
    repo="openai/privacy-filter",
    revision="7ffa9a043d54d1be65afb281eddf0ffbe629385b",
    subdir="original",
    files={
        "config.json": RemoteFile(707, "048a20604a3622de208d30df57cd5424bb583639b9ba20ddd7da593d3f89a248"),
        "model.safetensors": RemoteFile(
            2_798_984_088, "9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c"
        ),
        "viterbi_calibration.json": RemoteFile(
            372, "bbc8611ef08a55ed72d64856cbbbb9a91db8dfa881f0a92e2afbad6e4bbc775a"
        ),
    },
)


def default_checkpoint_dir() -> Path:
    return Path.home() / ".opf" / "privacy_filter"


def checkpoint_location(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Where the checkpoint is expected: an explicit path, else `OPF_CHECKPOINT`, else the default."""
    chosen = explicit if explicit is not None else os.environ.get(ENV_VAR) or None
    return Path(chosen).expanduser() if chosen is not None else default_checkpoint_dir()


def stderr_progress() -> Progress:
    """Progress reporter that prints at most one line per second to stderr."""
    last = {"time": 0.0}

    def report(name: str, done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and now - last["time"] < 1.0:
            return
        last["time"] = now
        print(f"  {name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", file=sys.stderr, flush=True)

    return report


def resolve_checkpoint(
    explicit: str | os.PathLike[str] | None = None,
    *,
    default_dir: Path | None = None,
    download: bool = True,
    source: Source = RELEASED,
    progress: Progress | None = None,
) -> Path:
    """Return the checkpoint directory to use, downloading the default one if needed."""
    chosen = explicit if explicit is not None else os.environ.get(ENV_VAR) or None
    if chosen is not None:
        path = Path(chosen).expanduser()
        if not path.is_dir():
            raise CheckpointError(f"Checkpoint directory not found: {path}")
        _require_complete(path)
        return path

    path = default_dir if default_dir is not None else default_checkpoint_dir()
    if path.exists():
        _require_complete(path)
        return path
    if not download:
        raise CheckpointError(
            f"No checkpoint at {path} and download is disabled. Run `opf-mlx download`, "
            f"pass --checkpoint, or set {ENV_VAR}."
        )
    print(
        f"No checkpoint at {path}. Downloading {source.repo} ({source.subdir}/, revision "
        f"{source.revision[:12]}) from {source.base_url}.",
        file=sys.stderr,
        flush=True,
    )
    return download_checkpoint(path, source=source, progress=progress)


def download_checkpoint(
    target: str | os.PathLike[str],
    *,
    source: Source = RELEASED,
    progress: Progress | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch every file of `source` into `target`.

    Files already present with the pinned hash are kept. A file that is present
    but different is somebody's data: unless `overwrite` is set, the download
    is refused before anything is fetched. (The reference never replaces an
    existing file either.)
    """
    target = Path(target).expanduser()
    wanted = {name: remote for name, remote in source.files.items() if not _matches(target / name, remote)}
    foreign = sorted(name for name in wanted if (target / name).exists())
    if foreign and not overwrite:
        raise CheckpointError(
            f"{target} already holds {', '.join(foreign)}, and {'it is' if len(foreign) == 1 else 'they are'} "
            f"not the pinned release of {source.repo}. Nothing was downloaded. Choose another directory, or ask "
            "to overwrite (--force on the command line, overwrite=True in Python)."
        )
    target.mkdir(parents=True, exist_ok=True)
    for name, remote in wanted.items():
        _fetch(source.url(name), target / name, remote, progress)
    return target


def verify_checkpoint(target: str | os.PathLike[str], *, source: Source = RELEASED) -> dict[str, bool]:
    """Whether each file of `source` is present in `target` with the pinned hash."""
    target = Path(target).expanduser()
    return {name: _matches(target / name, remote) for name, remote in source.files.items()}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _matches(path: Path, remote: RemoteFile) -> bool:
    return path.is_file() and path.stat().st_size == remote.size and sha256_file(path) == remote.sha256


def _require_complete(path: Path) -> None:
    hint = f"Run `opf-mlx download --checkpoint {path}` to fetch it, pass --checkpoint, or set {ENV_VAR}."
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint path is not a directory: {path}. {hint}")
    if not (path / "config.json").is_file():
        raise CheckpointError(f"Checkpoint at {path} is incomplete: missing config.json. {hint}")
    if not any(file.is_file() for file in path.glob("*.safetensors")):
        raise CheckpointError(f"Checkpoint at {path} is incomplete: no .safetensors file. {hint}")


def _fetch(url: str, destination: Path, remote: RemoteFile, progress: Progress | None) -> None:
    name = destination.name
    part = destination.with_name(name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset >= remote.size:
        part.unlink()
        offset = 0

    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = offset > 0 and response.status == 206
            written = offset if resumed else 0
            with part.open("ab" if resumed else "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(name, written, remote.size)
    except urllib.error.HTTPError as exc:
        raise CheckpointError(f"Could not download {name} from {url}: HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise CheckpointError(f"Could not download {name} from {url}: {exc}") from exc

    if written != remote.size:
        if written > remote.size:
            part.unlink()
        raise CheckpointError(f"{name}: size mismatch after download (expected {remote.size} bytes, got {written})")
    digest = sha256_file(part)
    if digest != remote.sha256:
        part.unlink()
        raise CheckpointError(f"{name}: sha256 mismatch (expected {remote.sha256}, got {digest}); file discarded")
    os.replace(part, destination)
