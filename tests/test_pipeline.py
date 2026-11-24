"""Text in, redaction out: the whole pipeline on small checkpoints.

Most tests use a rigged checkpoint (see `helpers.rigged_weights`): a real model,
run by the real code, whose answer for each token is fixed by the test. The
token ids are those of tiktoken's `o200k_base`:

    "Alice was born on 1990-01-02."
     100151 "Alice" | 673 " was" | 12275 " born" | 402 " on" | 220 " " | 3204 "199"
     | 15 "0" | 12 "-" | 2290 "01" | 12 "-" | 3286 "02" | 13 "."
"""

import json
import os
import subprocess
import sys
import warnings

import mlx.core as mx
import numpy as np
import pytest
from conftest import TINY_CONFIG
from helpers import RIGGED_LOGIT, random_weights, rigged_checkpoint, write_calibration, write_checkpoint
from mlx.utils import tree_flatten

from opf_mlx import DetectedSpan, PrivacyFilter, PrivacyFilterWarning, RedactionResult, env
from opf_mlx.checkpoint import CheckpointError
from opf_mlx.config import ConfigError, ModelConfig
from opf_mlx.env import InsufficientMemoryError
from opf_mlx.weights import WeightsError

SENTENCE = "Alice was born on 1990-01-02."
SENTENCE_IDS = [100151, 673, 12275, 402, 220, 3204, 15, 12, 2290, 12, 3286, 13]

ALICE, WAS, BORN = 100151, 673, 12275
PERSON_AND_DATE = {
    ALICE: {"S-private_person": RIGGED_LOGIT},
    3204: {"B-private_date": RIGGED_LOGIT},
    15: {"I-private_date": RIGGED_LOGIT},
    12: {"I-private_date": RIGGED_LOGIT},
    2290: {"I-private_date": RIGGED_LOGIT},
    3286: {"E-private_date": RIGGED_LOGIT},
}
# " was" opens a person span; " born" prefers background but could end it.
OPEN_SPAN = {
    WAS: {"B-private_person": RIGGED_LOGIT},
    BORN: {"O": RIGGED_LOGIT, "E-private_person": RIGGED_LOGIT - 4},
}


def rigged_filter(tmp_path, token_logits, **options) -> PrivacyFilter:
    return PrivacyFilter(rigged_checkpoint(tmp_path / "checkpoint", token_logits), **options)


@pytest.fixture(scope="module")
def random_checkpoint(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("random"), TINY_CONFIG, seed=3)


# ---------------------------------------------------------------- redaction


def test_detected_spans_are_replaced_by_typed_placeholders(tmp_path):
    result = rigged_filter(tmp_path, PERSON_AND_DATE).redact(SENTENCE)

    assert isinstance(result, RedactionResult)
    assert result.to_dict() == {
        "schema_version": 1,
        "summary": {
            "output_mode": "typed",
            "span_count": 2,
            "by_label": {"private_date": 1, "private_person": 1},
            "decoded_mismatch": False,
        },
        "text": "Alice was born on 1990-01-02.",
        "detected_spans": [
            {"label": "private_person", "start": 0, "end": 5, "text": "Alice", "placeholder": "<PRIVATE_PERSON>"},
            {"label": "private_date", "start": 18, "end": 28, "text": "1990-01-02", "placeholder": "<PRIVATE_DATE>"},
        ],
        "redacted_text": "<PRIVATE_PERSON> was born on <PRIVATE_DATE>.",
    }
    # The per-label counts are sorted by label, not by order of appearance.
    assert list(result.summary["by_label"]) == ["private_date", "private_person"]


def test_spans_are_exposed_as_objects(tmp_path):
    result = rigged_filter(tmp_path, {ALICE: {"S-private_person": RIGGED_LOGIT}}).redact(SENTENCE)

    assert result.detected_spans == (DetectedSpan("private_person", 0, 5, "Alice", "<PRIVATE_PERSON>"),)


def test_json_has_the_reference_key_order_and_keeps_non_ascii_text(tmp_path):
    # "a<|endoftext|>b" is [64, 199999, 65]: special-token text becomes the special id.
    filt = rigged_filter(tmp_path, {199999: {"S-secret": RIGGED_LOGIT}})

    assert filt.redact("a<|endoftext|>b").to_json(indent=None) == (
        '{"schema_version": 1, "summary": {"output_mode": "typed", "span_count": 1, "by_label": {"secret": 1}, '
        '"decoded_mismatch": false}, "text": "a<|endoftext|>b", "detected_spans": [{"label": "secret", "start": 1, '
        '"end": 14, "text": "<|endoftext|>", "placeholder": "<SECRET>"}], "redacted_text": "a<SECRET>b"}'
    )
    assert '"text": "é"' in filt.redact("é").to_json(indent=None)


def test_whitespace_is_trimmed_from_spans_by_default(tmp_path):
    result = rigged_filter(tmp_path, {WAS: {"S-private_person": RIGGED_LOGIT}}).redact(SENTENCE)

    # The token is " was"; the leading space is not part of the span.
    assert result.detected_spans == (DetectedSpan("private_person", 6, 9, "was", "<PRIVATE_PERSON>"),)
    assert result.redacted_text == "Alice <PRIVATE_PERSON> born on 1990-01-02."


def test_trimming_can_be_turned_off(tmp_path):
    filt = rigged_filter(tmp_path, {WAS: {"S-private_person": RIGGED_LOGIT}}, trim_whitespace=False)

    result = filt.redact(SENTENCE)

    assert result.detected_spans == (DetectedSpan("private_person", 5, 9, " was", "<PRIVATE_PERSON>"),)
    assert result.redacted_text == "Alice<PRIVATE_PERSON> born on 1990-01-02."


def test_redacted_output_mode_hides_the_categories(tmp_path):
    result = rigged_filter(tmp_path, PERSON_AND_DATE, output_mode="redacted").redact(SENTENCE)

    assert result.redacted_text == "<REDACTED> was born on <REDACTED>."
    assert [span.label for span in result.detected_spans] == ["redacted", "redacted"]
    assert result.summary == {
        "output_mode": "redacted",
        "span_count": 2,
        "by_label": {"redacted": 2},
        "decoded_mismatch": False,
    }


def test_empty_text_has_nothing_to_redact(tmp_path):
    result = rigged_filter(tmp_path, PERSON_AND_DATE).redact("")

    assert result.to_dict() == {
        "schema_version": 1,
        "summary": {"output_mode": "typed", "span_count": 0, "by_label": {}, "decoded_mismatch": False},
        "text": "",
        "detected_spans": [],
        "redacted_text": "",
    }


def test_a_character_split_across_tokens_is_redacted_once(tmp_path):
    # "x🧿y" is [87, 4103, 100, 123, 88]: the emoji's four bytes are spread over
    # three tokens, and each of them covers the whole character.
    filt = rigged_filter(tmp_path, {4103: {"S-secret": RIGGED_LOGIT}, 100: {"S-private_person": RIGGED_LOGIT}})
    assert filt.tokenizer.encode("x🧿y") == [87, 4103, 100, 123, 88]

    result = filt.redact("x🧿y")

    # Two spans claim character 1. The reference keeps one: earliest, then
    # longest, then first by label name.
    assert result.detected_spans == (DetectedSpan("private_person", 1, 2, "🧿", "<PRIVATE_PERSON>"),)
    assert result.redacted_text == "x<PRIVATE_PERSON>y"


# "x🧿a𝔘" is [87 "x", 4103, 100, 123, 64 "a", 43120, 242, 246]: 🧿 is tokens 4103, 100, 123
# and 𝔘 is tokens 43120, 242, 246. Three spans, each sharing a character with the next:
#   secret          tokens 87..4103      characters 0..2  "x🧿"
#   private_person  tokens 100..43120    characters 1..4  "🧿a𝔘"
#   private_person  token  242           characters 3..4  "𝔘"
CHAINED_SPANS = {
    87: {"B-secret": RIGGED_LOGIT},
    4103: {"E-secret": RIGGED_LOGIT},
    100: {"B-private_person": RIGGED_LOGIT},
    123: {"I-private_person": RIGGED_LOGIT},
    64: {"I-private_person": RIGGED_LOGIT},
    43120: {"E-private_person": RIGGED_LOGIT},
    242: {"S-private_person": RIGGED_LOGIT},
}


def test_overlapping_spans_are_resolved_left_to_right(tmp_path):
    filt = rigged_filter(tmp_path, CHAINED_SPANS)
    assert filt.tokenizer.encode("x🧿a𝔘") == [87, 4103, 100, 123, 64, 43120, 242, 246]

    result = filt.redact("x🧿a𝔘")

    # The first span wins, the second overlaps it and goes, the third is clear of the first.
    assert result.redacted_text == "<SECRET>a<PRIVATE_PERSON>"


def test_discarding_overlaps_per_label_happens_before_spans_are_chosen(tmp_path):
    filt = rigged_filter(tmp_path, CHAINED_SPANS, discard_overlapping_predicted_spans=True)

    result = filt.redact("x🧿a𝔘")

    # The third span overlaps the second, which has the same label, so it is
    # discarded first; then the second loses to the first as before.
    assert result.detected_spans == (DetectedSpan("secret", 0, 2, "x🧿", "<SECRET>"),)
    assert result.redacted_text == "<SECRET>a𝔘"


def test_text_that_does_not_round_trip_is_reported(tmp_path):
    # A lone surrogate cannot be encoded; the tokenizer sees U+FFFD (token 3251).
    filt = rigged_filter(tmp_path, {3251: {"S-secret": RIGGED_LOGIT}})

    with pytest.warns(PrivacyFilterWarning, match="did not exactly match tokenizer round-trip decode"):
        result = filt.redact("a" + chr(0xD800) + "b")

    assert result.text == "a" + chr(0xFFFD) + "b"
    assert result.summary["decoded_mismatch"] is True
    assert result.detected_spans == (DetectedSpan("secret", 1, 2, chr(0xFFFD), "<SECRET>"),)
    assert result.redacted_text == "a<SECRET>b"
    assert result.warning == (
        "Input text did not exactly match tokenizer round-trip decode; spans are based on decoded token text."
    )


# ----------------------------------------------------------------- decoding


def test_viterbi_completes_a_span_that_argmax_would_cut_short(tmp_path):
    # Per token, " born" is most likely background. As a sequence, a person
    # span that was opened must be closed, and closing it on " born" costs 4
    # nats where every alternative costs 20.
    result = rigged_filter(tmp_path, OPEN_SPAN).redact(SENTENCE)

    assert result.detected_spans == (DetectedSpan("private_person", 6, 14, "was born", "<PRIVATE_PERSON>"),)
    assert result.redacted_text == "Alice <PRIVATE_PERSON> on 1990-01-02."


def test_argmax_decoding_takes_each_token_on_its_own(tmp_path):
    result = rigged_filter(tmp_path, OPEN_SPAN, decode_mode="argmax").redact(SENTENCE)

    assert result.detected_spans == (DetectedSpan("private_person", 6, 9, "was", "<PRIVATE_PERSON>"),)
    assert result.redacted_text == "Alice <PRIVATE_PERSON> born on 1990-01-02."


def test_calibration_file_next_to_the_checkpoint_is_applied(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", OPEN_SPAN)
    # Ending a span now costs 30 more, so not opening one (20) is cheaper.
    write_calibration(checkpoint / "viterbi_calibration.json", transition_bias_inside_to_end=-30.0)

    result = PrivacyFilter(checkpoint).redact(SENTENCE)

    assert result.detected_spans == ()
    assert result.redacted_text == SENTENCE


def test_an_explicit_calibration_path_wins(tmp_path):
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", OPEN_SPAN)
    write_calibration(checkpoint / "viterbi_calibration.json", transition_bias_inside_to_end=-30.0)
    neutral = write_calibration(tmp_path / "neutral.json")

    result = PrivacyFilter(checkpoint, viterbi_calibration_path=neutral).redact(SENTENCE)

    assert result.redacted_text == "Alice <PRIVATE_PERSON> on 1990-01-02."


# ------------------------------------------------------------------ windows


def test_long_input_is_split_into_windows_and_nothing_is_dropped(tmp_path):
    filt = rigged_filter(tmp_path, PERSON_AND_DATE, n_ctx=5)

    with pytest.warns(PrivacyFilterWarning, match="3 consecutive windows"):
        result = filt.redact(SENTENCE)

    # 12 tokens in windows of 5, 5 and 2. The date straddles the last cut.
    assert result.redacted_text == "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."
    assert result.warning == (
        "Input has 12 tokens but the context window holds 5 (n_ctx), so it was processed in 3 consecutive windows. "
        "Nothing was dropped, but tokens near a window boundary see less context than they would in one pass, "
        "which can change predictions there."
    )


def test_input_that_fits_the_window_has_no_warning(tmp_path):
    result = rigged_filter(tmp_path, PERSON_AND_DATE, n_ctx=12).redact(SENTENCE)

    assert result.warning is None


def test_each_consecutive_window_sees_only_its_own_tokens(random_checkpoint):
    whole = PrivacyFilter(random_checkpoint, n_ctx=64)
    split = PrivacyFilter(random_checkpoint, n_ctx=5)

    with pytest.warns(PrivacyFilterWarning, match="3 consecutive windows"):
        scores = split.log_probs(SENTENCE_IDS)

    assert scores.shape == (12, 33) and scores.dtype == np.float32
    np.testing.assert_array_equal(scores[0:5], whole.log_probs(SENTENCE_IDS[0:5]))
    np.testing.assert_array_equal(scores[5:10], whole.log_probs(SENTENCE_IDS[5:10]))
    np.testing.assert_array_equal(scores[10:12], whole.log_probs(SENTENCE_IDS[10:12]))
    # In one pass, tokens next to a cut see their neighbours across it.
    assert np.abs(scores - whole.log_probs(SENTENCE_IDS)).max() > 1e-3


def test_log_probabilities_are_normalized(random_checkpoint):
    scores = PrivacyFilter(random_checkpoint).log_probs(SENTENCE_IDS)

    np.testing.assert_allclose(np.exp(scores.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)


# The tiny model sees 6 tokens to the left and 4 to the right, so its default halo is 6.
FORTY_TOKENS = [(7 * i) % 50 + 1 for i in range(40)]


def test_halo_windows_reproduce_a_single_pass(random_checkpoint):
    # float32 weights and 40 positions: here the rounding of the rotary table,
    # which differs between a window and the whole text, is far below 1e-4.
    # (In bfloat16, or thousands of positions apart, it is not: see
    # docs/architecture.md, section 5.)
    whole = PrivacyFilter(random_checkpoint, n_ctx=64, dtype="float32")
    halo = PrivacyFilter(random_checkpoint, n_ctx=20, window_policy="halo", dtype="float32")

    scores = halo.log_probs(FORTY_TOKENS)  # 4 windows

    np.testing.assert_allclose(scores, whole.log_probs(FORTY_TOKENS), atol=1e-4)
    # Consecutive windows of the same size do not.
    with pytest.warns(PrivacyFilterWarning, match="2 consecutive windows"):
        cut = PrivacyFilter(random_checkpoint, n_ctx=20, dtype="float32").log_probs(FORTY_TOKENS)
    assert np.abs(cut - whole.log_probs(FORTY_TOKENS)).max() > 1e-3


def test_halo_windows_that_cover_the_models_reach_come_without_a_warning(random_checkpoint):
    filt = PrivacyFilter(random_checkpoint, n_ctx=20, window_policy="halo")
    text = " ".join([SENTENCE] * 4)
    assert len(filt.tokenizer.encode(text)) > 40  # several windows

    with warnings.catch_warnings():
        warnings.simplefilter("error", PrivacyFilterWarning)
        result = filt.redact(text)

    assert result.warning is None


def test_a_halo_shorter_than_the_models_reach_is_reported(random_checkpoint):
    # 5 tokens cover the model's reach to the right (4) but not to the left (6).
    filt = PrivacyFilter(random_checkpoint, n_ctx=20, window_policy="halo", halo=5)

    with pytest.warns(PrivacyFilterWarning, match="3 windows with a halo of 5 tokens, less than the model's reach"):
        filt.log_probs(FORTY_TOKENS)


@pytest.mark.parametrize("options", [{"n_ctx": 12}, {"n_ctx": 20, "halo": 10}, {"n_ctx": 1, "halo": 1}])
def test_a_window_with_no_room_between_two_halos_is_refused_when_the_filter_is_built(random_checkpoint, options):
    with pytest.raises(ValueError, match="halo.*n_ctx of at least"):
        PrivacyFilter(random_checkpoint, window_policy="halo", **options)


def test_default_window_is_the_checkpoints_on_gpu_and_4096_on_cpu(tmp_path):
    # The reference uses the checkpoint's default_n_ctx on an accelerator and 4096 on the CPU.
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", {})

    assert PrivacyFilter(checkpoint).n_ctx == 64
    assert PrivacyFilter(checkpoint, device="cpu").n_ctx == 4096
    assert PrivacyFilter(checkpoint, device="cpu", n_ctx=7).n_ctx == 7


# ------------------------------------------------------------------ loading


@pytest.mark.parametrize(
    "option, message",
    [
        ({"output_mode": "masked"}, "Unsupported output_mode: 'masked'"),
        ({"decode_mode": "beam"}, "Unsupported decode_mode: 'beam'"),
        ({"device": "cuda"}, "Unsupported device: 'cuda'"),
        ({"window_policy": "sliding"}, "unknown window policy 'sliding'"),
        ({"n_ctx": 0}, "n_ctx must be positive"),
        ({"halo": -1}, "halo must not be negative"),
        ({"dtype": "float16"}, "Unsupported dtype: 'float16'"),
    ],
)
def test_unknown_options_are_rejected(tmp_path, option, message):
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", {})

    with pytest.raises(ValueError, match=message):
        PrivacyFilter(checkpoint, **option)


def test_a_checkpoint_that_does_not_match_the_architecture_is_refused(tmp_path):
    weights = random_weights(ModelConfig.from_dict(TINY_CONFIG))
    del weights["unembedding.weight"]
    checkpoint = write_checkpoint(tmp_path / "checkpoint", TINY_CONFIG, weights)

    with pytest.raises(WeightsError, match="missing tensor: unembedding.weight"):
        PrivacyFilter(checkpoint)


@pytest.mark.parametrize("vocab_size", [1000, 200018])
def test_a_vocabulary_smaller_than_the_tokenizer_is_refused(tmp_path, vocab_size):
    # o200k_base ids run from 0 to 200018, so 200019 embedding rows are needed.
    checkpoint = write_checkpoint(tmp_path / "checkpoint", {**TINY_CONFIG, "vocab_size": vocab_size})

    with pytest.raises(ConfigError, match=f"vocab_size {vocab_size} .* o200k_base .* 200018"):
        PrivacyFilter(checkpoint)


def test_a_vocabulary_that_just_covers_the_tokenizer_is_accepted(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "checkpoint", {**TINY_CONFIG, "vocab_size": 200019})

    assert PrivacyFilter(checkpoint).config.vocab_size == 200019


def test_a_missing_checkpoint_directory_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="Checkpoint directory not found"):
        PrivacyFilter(tmp_path / "nowhere")


def test_weights_keep_the_checkpoint_precision_by_default(tmp_path):
    filt = rigged_filter(tmp_path, {})

    dtypes = {name: value.dtype for name, value in tree_flatten(filt.model.parameters())}
    assert filt.dtype == "bfloat16"
    assert dtypes["embedding.weight"] == mx.bfloat16
    assert dtypes["block.0.attn.sinks"] == mx.float32


def test_float32_can_be_requested(tmp_path):
    filt = rigged_filter(tmp_path, PERSON_AND_DATE, dtype="float32")

    assert filt.dtype == "float32"
    assert {value.dtype for _, value in tree_flatten(filt.model.parameters())} == {mx.float32}
    assert filt.redact(SENTENCE).redacted_text == "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."


def test_a_model_larger_than_the_device_limit_is_refused(tmp_path, monkeypatch):
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", {})
    monkeypatch.setattr(env, "gpu_memory_limit", lambda: 1024)  # a machine with 1 KiB for the GPU

    with pytest.raises(InsufficientMemoryError, match="MLX can use at most 1.0 KiB"):
        PrivacyFilter(checkpoint)


def test_low_free_memory_warns_but_loads(tmp_path, monkeypatch):
    checkpoint = rigged_checkpoint(tmp_path / "checkpoint", PERSON_AND_DATE)
    monkeypatch.setattr(env, "available_memory", lambda: 1024)

    with pytest.warns(PrivacyFilterWarning, match="only 1.0 KiB of memory is free"):
        filt = PrivacyFilter(checkpoint)

    assert filt.redact(SENTENCE).redacted_text == "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."


# ------------------------------------------------------------------- device


def test_the_cpu_gives_the_same_redaction(tmp_path):
    filt = rigged_filter(tmp_path, PERSON_AND_DATE, device="cpu")

    assert filt.device == "cpu"
    assert filt.redact(SENTENCE).redacted_text == "<PRIVATE_PERSON> was born on <PRIVATE_DATE>."


def test_cpu_and_gpu_agree_on_a_random_model(random_checkpoint):
    gpu = PrivacyFilter(random_checkpoint, dtype="float32")
    cpu = PrivacyFilter(random_checkpoint, dtype="float32", device="cpu")

    np.testing.assert_allclose(cpu.log_probs(SENTENCE_IDS), gpu.log_probs(SENTENCE_IDS), atol=1e-4)


REDUCED_PRECISION = """
import json, sys
import mlx.core as mx
import numpy as np
from opf_mlx import PrivacyFilter
from opf_mlx.precision import PrecisionError

checkpoint, ids = sys.argv[1], json.loads(sys.argv[2])
try:
    PrivacyFilter(checkpoint, dtype="float32")
    gpu = "loaded"
except PrecisionError as exc:
    gpu = f"refused: {exc}"
cpu = PrivacyFilter(checkpoint, dtype="float32", device="cpu")
on_gpu = np.array(cpu.model(mx.array([ids])))
with mx.stream(mx.cpu):
    on_cpu = np.array(cpu.model(mx.array([ids])))
print(json.dumps({"gpu": gpu, "gap": float(np.abs(on_gpu - on_cpu).max()), "cpu": cpu.log_probs(ids).tolist()}))
"""


def test_with_reduced_precision_forced_on_the_gpu_is_refused_and_the_cpu_is_exact(random_checkpoint):
    expected = PrivacyFilter(random_checkpoint, dtype="float32").log_probs(SENTENCE_IDS)
    environment = {**os.environ, "MLX_ENABLE_TF32": "1"}

    done = subprocess.run(
        [sys.executable, "-c", REDUCED_PRECISION, str(random_checkpoint), json.dumps(SENTENCE_IDS)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)

    if report["gap"] < 1e-4:
        pytest.skip("this GPU has no reduced-precision float32 matmul")
    assert report["gpu"].startswith("refused:") and "MLX_ENABLE_TF32=0" in report["gpu"]
    # The CPU ignores the switch, so device="cpu" must give what an exact run gives.
    np.testing.assert_allclose(np.array(report["cpu"], dtype=np.float32), expected, atol=1e-4)
