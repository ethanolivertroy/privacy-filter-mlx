"""Finding, downloading and verifying a checkpoint directory.

Downloads run against a local HTTP server that lays files out the way the
Hugging Face endpoint does, so the real transfer code is exercised without
touching the network.
"""

import hashlib
import os
from pathlib import Path

import pytest

from helpers import serve_directory
from opf_mlx.checkpoint import (
    RELEASED,
    CheckpointError,
    RemoteFile,
    Source,
    checkpoint_location,
    default_checkpoint_dir,
    download_checkpoint,
    resolve_checkpoint,
    verify_checkpoint,
)

CONFIG = b'{"model_type": "privacy_filter"}'
WEIGHTS = bytes(range(256)) * 4000  # 1,024,000 bytes
CALIBRATION = b'{"operating_points": {}}'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def remote(tmp_path) -> Path:
    """A directory laid out like `<repo>/resolve/<revision>/original/` on the hub."""
    root = tmp_path / "remote"
    files = root / "acme" / "tiny" / "resolve" / "rev1" / "original"
    files.mkdir(parents=True)
    (files / "config.json").write_bytes(CONFIG)
    (files / "model.safetensors").write_bytes(WEIGHTS)
    (files / "viterbi_calibration.json").write_bytes(CALIBRATION)
    return root


def source(base_url: str, **overrides: RemoteFile) -> Source:
    files = {
        "config.json": RemoteFile(len(CONFIG), sha(CONFIG)),
        "model.safetensors": RemoteFile(len(WEIGHTS), sha(WEIGHTS)),
        "viterbi_calibration.json": RemoteFile(len(CALIBRATION), sha(CALIBRATION)),
    }
    files.update(overrides)
    return Source(base_url=base_url, repo="acme/tiny", revision="rev1", subdir="original", files=files)


def test_download_fetches_every_file_into_the_target(tmp_path, remote):
    target = tmp_path / "ckpt"
    with serve_directory(remote) as (url, _):
        download_checkpoint(target, source=source(url), progress=None)

    assert (target / "config.json").read_bytes() == CONFIG
    assert (target / "model.safetensors").read_bytes() == WEIGHTS
    assert (target / "viterbi_calibration.json").read_bytes() == CALIBRATION
    assert sorted(p.name for p in target.iterdir()) == ["config.json", "model.safetensors", "viterbi_calibration.json"]


def test_download_requests_the_pinned_revision(tmp_path, remote):
    with serve_directory(remote) as (url, seen):
        download_checkpoint(tmp_path / "ckpt", source=source(url), progress=None)

    assert ("/acme/tiny/resolve/rev1/original/model.safetensors", None) in seen


def test_content_that_does_not_match_the_pinned_hash_is_never_installed(tmp_path, remote):
    target = tmp_path / "ckpt"
    wrong = RemoteFile(len(WEIGHTS), sha(b"something else"))
    with serve_directory(remote) as (url, _):
        with pytest.raises(CheckpointError, match="sha256"):
            download_checkpoint(target, source=source(url, **{"model.safetensors": wrong}), progress=None)

    assert not (target / "model.safetensors").exists()
    # Nor are the bad bytes kept for a later attempt to build on.
    assert not (target / "model.safetensors.part").exists()


def test_short_download_is_an_error(tmp_path, remote):
    target = tmp_path / "ckpt"
    longer = RemoteFile(len(WEIGHTS) + 5, sha(WEIGHTS))
    with serve_directory(remote) as (url, _):
        with pytest.raises(CheckpointError, match="size"):
            download_checkpoint(target, source=source(url, **{"model.safetensors": longer}), progress=None)

    assert not (target / "model.safetensors").exists()
    # What did arrive is kept, so that the next attempt can resume from it.
    assert (target / "model.safetensors.part").read_bytes() == WEIGHTS


def test_a_download_longer_than_expected_is_an_error_and_is_not_kept(tmp_path, remote):
    target = tmp_path / "ckpt"
    shorter = RemoteFile(len(WEIGHTS) - 5, sha(WEIGHTS[:-5]))
    with serve_directory(remote) as (url, _):
        with pytest.raises(CheckpointError, match="size"):
            download_checkpoint(target, source=source(url, **{"model.safetensors": shorter}), progress=None)

    assert not (target / "model.safetensors").exists()
    # Too many bytes cannot be the beginning of the file: there is nothing to resume from.
    assert not (target / "model.safetensors.part").exists()


def test_a_leftover_partial_file_as_long_as_the_whole_file_is_discarded(tmp_path, remote):
    # Such a leftover cannot be the beginning of the file, and a server refuses
    # to send "the rest" after it (416). The download starts over instead.
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "model.safetensors.part").write_bytes(bytes(len(WEIGHTS)))

    with serve_directory(remote) as (url, seen):
        download_checkpoint(target, source=source(url), progress=None)

    assert (target / "model.safetensors").read_bytes() == WEIGHTS
    assert ("/acme/tiny/resolve/rev1/original/model.safetensors", None) in seen


def test_interrupted_download_resumes_from_the_partial_file(tmp_path, remote):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "model.safetensors.part").write_bytes(WEIGHTS[:300_000])

    with serve_directory(remote) as (url, seen):
        download_checkpoint(target, source=source(url), progress=None)

    assert (target / "model.safetensors").read_bytes() == WEIGHTS
    assert ("/acme/tiny/resolve/rev1/original/model.safetensors", "bytes=300000-") in seen
    assert not (target / "model.safetensors.part").exists()


def test_resume_restarts_cleanly_when_the_server_ignores_range(tmp_path, remote):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "model.safetensors.part").write_bytes(WEIGHTS[:300_000])

    with serve_directory(remote, honor_range=False) as (url, _):
        download_checkpoint(target, source=source(url), progress=None)

    assert (target / "model.safetensors").read_bytes() == WEIGHTS


def test_files_already_present_and_correct_are_not_downloaded_again(tmp_path, remote):
    target = tmp_path / "ckpt"
    with serve_directory(remote) as (url, seen):
        download_checkpoint(target, source=source(url), progress=None)
        first = len(seen)
        download_checkpoint(target, source=source(url), progress=None)

    assert len(seen) == first


def test_missing_remote_file_is_an_error(tmp_path, remote):
    (remote / "acme" / "tiny" / "resolve" / "rev1" / "original" / "viterbi_calibration.json").unlink()
    with serve_directory(remote) as (url, _):
        with pytest.raises(CheckpointError, match="viterbi_calibration.json"):
            download_checkpoint(tmp_path / "ckpt", source=source(url), progress=None)


def test_verify_reports_each_file(tmp_path, remote):
    target = tmp_path / "ckpt"
    with serve_directory(remote) as (url, _):
        src = source(url)
        download_checkpoint(target, source=src, progress=None)
    (target / "config.json").write_bytes(b"{}")

    report = verify_checkpoint(target, source=src)

    assert report == {"config.json": False, "model.safetensors": True, "viterbi_calibration.json": True}


def test_verify_compares_content_not_only_size(tmp_path, remote):
    target = tmp_path / "ckpt"
    with serve_directory(remote) as (url, _):
        src = source(url)
        download_checkpoint(target, source=src, progress=None)
    (target / "config.json").write_bytes(b"?" * len(CONFIG))

    assert verify_checkpoint(target, source=src)["config.json"] is False


def test_released_source_is_pinned_to_a_revision_and_hashes():
    assert RELEASED.repo == "openai/privacy-filter"
    assert RELEASED.revision == "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
    assert RELEASED.files["model.safetensors"] == RemoteFile(
        2_798_984_088, "9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c"
    )
    assert RELEASED.url("model.safetensors") == (
        "https://huggingface.co/openai/privacy-filter/resolve/"
        "7ffa9a043d54d1be65afb281eddf0ffbe629385b/original/model.safetensors"
    )


# --- resolution order ------------------------------------------------------


def make_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(CONFIG)
    (path / "model.safetensors").write_bytes(b"x")
    return path


def test_explicit_path_wins_over_environment_and_default(tmp_path, monkeypatch):
    explicit, env, default = (make_checkpoint(tmp_path / name) for name in ("explicit", "env", "default"))
    monkeypatch.setenv("OPF_CHECKPOINT", str(env))

    assert resolve_checkpoint(explicit, default_dir=default) == explicit


def test_environment_variable_wins_over_default(tmp_path, monkeypatch):
    env, default = (make_checkpoint(tmp_path / name) for name in ("env", "default"))
    monkeypatch.setenv("OPF_CHECKPOINT", str(env))

    assert resolve_checkpoint(None, default_dir=default) == env


def test_default_directory_is_used_when_nothing_else_is_given(tmp_path, monkeypatch):
    default = make_checkpoint(tmp_path / "default")
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    assert resolve_checkpoint(None, default_dir=default) == default


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors"])
def test_incomplete_checkpoint_directory_is_rejected(tmp_path, monkeypatch, missing):
    path = make_checkpoint(tmp_path / "ckpt")
    (path / missing).unlink()
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    with pytest.raises(CheckpointError, match=str(path)):
        resolve_checkpoint(path)
    with pytest.raises(CheckpointError, match=str(path)):
        resolve_checkpoint(None, default_dir=path)


def test_explicit_path_that_does_not_exist_is_an_error_not_a_download(tmp_path, monkeypatch):
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    with pytest.raises(CheckpointError, match="not found"):
        resolve_checkpoint(tmp_path / "nowhere")


def test_absent_default_directory_is_downloaded(tmp_path, monkeypatch, remote):
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)
    default = tmp_path / "home" / ".opf" / "privacy_filter"

    with serve_directory(remote) as (url, _):
        resolved = resolve_checkpoint(None, default_dir=default, source=source(url), progress=None)

    assert resolved == default
    assert (default / "model.safetensors").read_bytes() == WEIGHTS


def test_download_can_be_forbidden(tmp_path, monkeypatch):
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)

    with pytest.raises(CheckpointError, match="download"):
        resolve_checkpoint(None, default_dir=tmp_path / "absent", download=False)


def test_location_is_the_explicit_path_then_the_environment_then_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPF_CHECKPOINT", raising=False)
    assert checkpoint_location() == tmp_path / "home" / ".opf" / "privacy_filter"

    monkeypatch.setenv("OPF_CHECKPOINT", str(tmp_path / "from-env"))
    assert checkpoint_location() == tmp_path / "from-env"
    assert checkpoint_location(tmp_path / "explicit") == tmp_path / "explicit"


def test_location_expands_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert checkpoint_location("~/models/pf") == tmp_path / "home" / "models" / "pf"


def test_an_explicit_path_may_start_at_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    installed = make_checkpoint(tmp_path / "home" / "models" / "pf")

    assert resolve_checkpoint("~/models/pf") == installed


def test_the_suite_cannot_reach_the_real_home_directory(tmp_path_factory):
    # conftest.py gives every test a temporary HOME and no OPF_CHECKPOINT. Without
    # that, a bug in a download path writes into the checkpoint a person installed.
    assert default_checkpoint_dir().is_relative_to(tmp_path_factory.getbasetemp())
    assert "OPF_CHECKPOINT" not in os.environ


def test_an_existing_file_that_is_not_the_release_is_left_alone(tmp_path, remote):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"someone else's weights")

    with serve_directory(remote) as (url, seen):
        with pytest.raises(CheckpointError, match="model.safetensors.*not the pinned release.*overwrite"):
            download_checkpoint(target, source=source(url), progress=None)

    assert (target / "model.safetensors").read_bytes() == b"someone else's weights"
    # Refused before anything was fetched: the other files were not installed either.
    assert seen == []
    assert sorted(path.name for path in target.iterdir()) == ["model.safetensors"]


def test_an_existing_file_of_the_right_size_but_other_content_is_left_alone(tmp_path, remote):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "config.json").write_bytes(b"?" * len(CONFIG))

    with serve_directory(remote) as (url, seen):
        with pytest.raises(CheckpointError, match="config.json.*not the pinned release"):
            download_checkpoint(target, source=source(url), progress=None)

    assert (target / "config.json").read_bytes() == b"?" * len(CONFIG)
    assert seen == []


def test_replacing_existing_files_must_be_asked_for(tmp_path, remote):
    target = tmp_path / "ckpt"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"someone else's weights")

    with serve_directory(remote) as (url, _):
        download_checkpoint(target, source=source(url), progress=None, overwrite=True)

    assert (target / "model.safetensors").read_bytes() == WEIGHTS
