"""Tokenization and byte-accurate character offsets.

The literal token ids were recorded from tiktoken 0.14.0 `o200k_base`.
"""

import pytest
import tiktoken

from opf_mlx.tokenizer import Tokenizer


def byte_token(value: int) -> int:
    """Id of the vocabulary entry that is the single byte `value`."""
    return tiktoken.get_encoding("o200k_base").encode_single_token(bytes([value]))


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer("o200k_base")


def test_encode_matches_o200k_base(tok):
    assert tok.encode("Alice was born on 1990-01-02.") == [
        100151, 673, 12275, 402, 220, 3204, 15, 12, 2290, 12, 3286, 13,
    ]  # fmt: skip


def test_special_token_text_in_the_input_becomes_the_special_id(tok):
    # The reference encodes with allowed_special="all".
    assert tok.encode("a<|endoftext|>b") == [64, 199999, 65]


def test_empty_text_has_no_tokens(tok):
    assert tok.encode("") == []


def test_offsets_for_ascii_tokens(tok):
    text, starts, ends = tok.decode_with_offsets([100151, 673, 12275])

    assert text == "Alice was born"
    assert starts == [0, 5, 9]
    assert ends == [5, 9, 14]


def test_tokens_that_split_a_multibyte_character_widen_to_the_whole_character(tok):
    # "é" is C3 A9 in UTF-8. Feed its two bytes as separate tokens, then "a".
    ids = [byte_token(0xC3), byte_token(0xA9), byte_token(ord("a"))]

    text, starts, ends = tok.decode_with_offsets(ids)

    assert text == "éa"
    assert starts == [0, 0, 1]
    assert ends == [1, 1, 2]


def test_offsets_count_code_points_not_bytes(tok):
    text = "naïve 日本語 🙂 ok"
    ids = tok.encode(text)

    decoded, starts, ends = tok.decode_with_offsets(ids)

    assert decoded == text
    assert starts[0] == 0
    assert ends[-1] == len(text) == 14
    assert all(s <= e for s, e in zip(starts, ends))
    assert all(a <= b for a, b in zip(starts, starts[1:]))
    # Every character belongs to at least one token.
    covered = set()
    for s, e in zip(starts, ends):
        covered.update(range(s, e))
    assert covered == set(range(len(text)))


def test_lone_surrogate_does_not_round_trip(tok):
    text = "a\ud800b"

    decoded, _, _ = tok.decode_with_offsets(tok.encode(text))

    assert decoded == "a\ufffdb"
    assert decoded != text


def test_token_bytes_that_are_not_valid_utf8_raise_instead_of_misaligning(tok):
    # A lone continuation byte decodes to U+FFFD, which is three bytes long, so
    # byte offsets no longer line up with the decoded text. The reference
    # raises in this case rather than return shifted spans.
    ids = [byte_token(0x80), *tok.encode("abc")]

    with pytest.raises(ValueError, match="mismatch"):
        tok.decode_with_offsets(ids)

