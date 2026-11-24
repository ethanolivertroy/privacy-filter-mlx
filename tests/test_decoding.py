"""Constrained Viterbi decoding."""

import itertools
import json

import numpy as np
import pytest

from opf_mlx.decoding import ViterbiDecoder, argmax_decode, load_transition_biases
from opf_mlx.labels import resolve_label_space

O = 0
B_PERSON, I_PERSON, E_PERSON, S_PERSON = 17, 18, 19, 20
BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


@pytest.fixture
def space(released_config):
    return resolve_label_space(released_config)


def logp_with(n_tokens: int, n_classes: int, entries: dict[tuple[int, int], float], fill: float = -20.0) -> np.ndarray:
    logp = np.full((n_tokens, n_classes), fill, dtype=np.float32)
    for (t, c), value in entries.items():
        logp[t, c] = value
    return logp


def test_viterbi_rejects_a_sequence_that_starts_inside_a_span(space):
    logp = logp_with(
        2,
        33,
        {(0, O): -2.0, (0, I_PERSON): -0.1, (0, B_PERSON): -1.0, (0, S_PERSON): -3.0, (1, O): -0.1, (1, E_PERSON): -1.5},
    )

    # Per-token argmax picks the ill-formed [I, O]. Valid paths score
    # [O, O] = -2.1, [B, E] = -2.5, [S, O] = -3.1.
    assert argmax_decode(logp) == [I_PERSON, O]
    assert ViterbiDecoder(space).decode(logp) == [O, O]


def test_viterbi_does_not_leave_a_span_open_at_the_end(space):
    logp = logp_with(2, 33, {(0, B_PERSON): -0.1, (0, S_PERSON): -1.0, (0, O): -4.0, (1, I_PERSON): -0.1, (1, E_PERSON): -2.0, (1, O): -0.2})

    # [B, I] would score -0.2 but ends inside a span. [B, E] = -2.1,
    # [S, O] = -1.2, [O, O] = -4.2.
    assert ViterbiDecoder(space).decode(logp) == [S_PERSON, O]


def test_transition_bias_changes_the_chosen_path(space):
    logp = logp_with(
        3,
        33,
        {
            (0, B_PERSON): -0.1,
            (1, I_PERSON): -1.0,
            (1, E_PERSON): -0.5,
            (2, O): -0.2,
            (2, E_PERSON): -1.0,
        },
    )
    # Unbiased: [B, E, O] = -0.8 beats [B, I, E] = -2.1.
    assert ViterbiDecoder(space).decode(logp) == [B_PERSON, E_PERSON, O]

    biased = ViterbiDecoder(space, {**dict.fromkeys(BIAS_KEYS, 0.0), "transition_bias_inside_to_continue": 2.0})

    # With +2.0 on B->I: [B, I, E] = -0.1.
    assert biased.decode(logp) == [B_PERSON, I_PERSON, E_PERSON]


def test_ties_resolve_to_the_lowest_class_index(space):
    assert ViterbiDecoder(space).decode(np.zeros((3, 33), dtype=np.float32)) == [0, 0, 0]


def test_empty_sequence_decodes_to_nothing(space):
    assert ViterbiDecoder(space).decode(np.zeros((0, 33), dtype=np.float32)) == []


def test_non_finite_scores_fall_back_to_argmax(space):
    logp = np.full((2, 33), np.nan, dtype=np.float32)

    assert ViterbiDecoder(space).decode(logp) == [0, 0]


def test_wrong_shape_is_rejected(space):
    with pytest.raises(ValueError, match="shape"):
        ViterbiDecoder(space).decode(np.zeros((33,), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        ViterbiDecoder(space).decode(np.zeros((2, 32), dtype=np.float32))


def _allowed(prev: str | None, nxt: str) -> bool:
    """BIOES grammar written out by hand, independent of the decoder."""
    nxt_tag, _, nxt_cat = nxt.partition("-")
    if prev is None:  # start of text
        return nxt_tag in ("O", "B", "S")
    prev_tag, _, prev_cat = prev.partition("-")
    if prev_tag in ("O", "E", "S"):
        return nxt_tag in ("O", "B", "S")
    return nxt_tag in ("I", "E") and nxt_cat == prev_cat  # prev is B or I


def test_viterbi_finds_the_best_valid_path_by_brute_force(released_config):
    released_config["span_class_names"] = ["O", "x", "y"]
    released_config["num_labels"] = 9
    space = resolve_label_space(released_config)
    names = space.class_names
    decoder = ViterbiDecoder(space)
    rng = np.random.default_rng(20261123)

    for _ in range(25):
        n_tokens = int(rng.integers(1, 6))
        logp = rng.normal(size=(n_tokens, 9)).astype(np.float32)

        best_score, best_path = -np.inf, None
        for path in itertools.product(range(9), repeat=n_tokens):
            ok = all(_allowed(names[p] if i else None, names[c]) for i, (p, c) in enumerate(zip((0, *path), path)))
            ok = ok and names[path[-1]].partition("-")[0] in ("O", "E", "S")
            if not ok:
                continue
            score = float(sum(np.float64(logp[t, c]) for t, c in enumerate(path)))
            if score > best_score:
                best_score, best_path = score, list(path)

        assert decoder.decode(logp) == best_path


def _calibration(biases: dict[str, object]) -> dict[str, object]:
    return {"operating_points": {"default": {"biases": biases}}}


def test_biases_are_read_from_the_calibration_file(tmp_path):
    biases = {key: float(i) / 10 for i, key in enumerate(BIAS_KEYS)}
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(_calibration(biases)), encoding="utf-8")

    assert load_transition_biases(calibration_path=path) == biases


def test_calibration_next_to_the_checkpoint_is_discovered(tmp_path):
    biases = dict.fromkeys(BIAS_KEYS, 0.25)
    (tmp_path / "viterbi_calibration.json").write_text(json.dumps(_calibration(biases)), encoding="utf-8")

    assert load_transition_biases(checkpoint_dir=tmp_path) == biases


def test_explicit_calibration_path_wins_over_the_checkpoint_one(tmp_path):
    (tmp_path / "viterbi_calibration.json").write_text(json.dumps(_calibration(dict.fromkeys(BIAS_KEYS, 0.25))), encoding="utf-8")
    explicit = tmp_path / "other.json"
    explicit.write_text(json.dumps(_calibration(dict.fromkeys(BIAS_KEYS, 0.5))), encoding="utf-8")

    assert load_transition_biases(calibration_path=explicit, checkpoint_dir=tmp_path) == dict.fromkeys(BIAS_KEYS, 0.5)


def test_no_calibration_means_zero_biases(tmp_path):
    assert load_transition_biases(checkpoint_dir=tmp_path) == dict.fromkeys(BIAS_KEYS, 0.0)


def test_missing_calibration_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transition_biases(calibration_path=tmp_path / "absent.json")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_calibration({k: 0.0 for k in BIAS_KEYS[:-1]}), id="missing bias"),
        pytest.param(_calibration({**dict.fromkeys(BIAS_KEYS, 0.0), "extra": 1.0}), id="extra bias"),
        pytest.param(_calibration({**dict.fromkeys(BIAS_KEYS, 0.0), BIAS_KEYS[0]: "0.1"}), id="non-numeric bias"),
        pytest.param(_calibration({**dict.fromkeys(BIAS_KEYS, 0.0), BIAS_KEYS[0]: True}), id="boolean bias"),
        pytest.param({"operating_points": {"default": {"biases": {}}}, "version": 2}, id="extra top-level key"),
        pytest.param({"operating_points": {"strict": {"biases": {}}}}, id="no default operating point"),
        pytest.param([1, 2], id="not an object"),
    ],
)
def test_malformed_calibration_is_rejected(tmp_path, payload):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_transition_biases(calibration_path=path)
