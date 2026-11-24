import pytest

from opf_mlx.labels import resolve_label_space
from opf_mlx.spans import (
    DetectedSpan,
    discard_overlapping_by_label,
    labels_to_spans,
    placeholder_for,
    redact_text,
    select_non_overlapping,
    token_spans_to_char_spans,
    trim_whitespace,
)

O = 0
B_EMAIL, I_EMAIL, E_EMAIL, S_EMAIL = 13, 14, 15, 16
B_PERSON, I_PERSON, E_PERSON, S_PERSON = 17, 18, 19, 20
EMAIL, PERSON = 4, 5


@pytest.fixture
def space(released_config):
    return resolve_label_space(released_config)


@pytest.mark.parametrize(
    "labels, expected",
    [
        pytest.param([O, B_PERSON, I_PERSON, E_PERSON, O], [(PERSON, 1, 4)], id="well-formed span"),
        pytest.param([S_PERSON, O, S_PERSON], [(PERSON, 0, 1), (PERSON, 2, 3)], id="single-token spans"),
        pytest.param([O, O, O], [], id="background only"),
        pytest.param([O, I_PERSON, E_PERSON], [(PERSON, 1, 3)], id="inside without begin opens a span"),
        pytest.param([O, E_PERSON, O], [(PERSON, 1, 2)], id="end without begin is a one-token span"),
        pytest.param([B_PERSON, B_PERSON, E_PERSON], [(PERSON, 0, 1), (PERSON, 1, 3)], id="begin closes an open span"),
        pytest.param([B_PERSON, I_EMAIL, E_EMAIL], [(PERSON, 0, 1), (EMAIL, 1, 3)], id="category change mid-span"),
        pytest.param([O, B_PERSON, I_PERSON], [(PERSON, 1, 3)], id="open span is closed at the end"),
        pytest.param([B_PERSON, O], [(PERSON, 0, 1)], id="background closes an open span"),
        pytest.param(
            [B_PERSON, S_PERSON, E_PERSON],
            [(PERSON, 0, 1), (PERSON, 1, 2), (PERSON, 2, 3)],
            id="single inside an open span",
        ),
        pytest.param([B_PERSON, E_EMAIL], [(PERSON, 0, 1), (EMAIL, 1, 2)], id="end of another category"),
        pytest.param([], [], id="no tokens"),
    ],
)
def test_labels_to_spans(space, labels, expected):
    assert labels_to_spans(labels, space) == expected


def test_token_spans_become_character_spans():
    char_starts = [0, 5, 6]
    char_ends = [5, 6, 11]

    assert token_spans_to_char_spans([(PERSON, 1, 3)], char_starts, char_ends) == [(PERSON, 5, 11)]


def test_token_spans_with_no_characters_or_out_of_range_are_dropped():
    char_starts = [0, 3, 3]
    char_ends = [3, 3, 7]

    spans = [(PERSON, 1, 2), (EMAIL, 2, 5), (EMAIL, 2, 3)]

    assert token_spans_to_char_spans(spans, char_starts, char_ends) == [(EMAIL, 3, 7)]


def test_trim_whitespace_shrinks_spans_and_drops_blank_ones():
    text = " Alice  \n\t "

    assert trim_whitespace([(PERSON, 0, 8), (EMAIL, 6, 11)], text) == [(PERSON, 1, 6)]


def test_discard_overlapping_keeps_the_earliest_then_longest_span_per_label():
    spans = [(1, 3, 8), (1, 0, 5), (2, 3, 8), (1, 0, 2)]

    assert discard_overlapping_by_label(spans) == [(1, 0, 5), (2, 3, 8)]


def test_select_non_overlapping_prefers_earlier_then_longer_then_label_order():
    def span(label, start, end):
        return DetectedSpan(label=label, start=start, end=end, text="x" * (end - start), placeholder="<X>")

    spans = [span("b", 0, 5), span("c", 6, 9), span("a", 0, 8), span("d", 8, 10), span("e", 8, 10)]

    kept = select_non_overlapping(spans)

    assert [(s.label, s.start, s.end) for s in kept] == [("a", 0, 8), ("d", 8, 10)]


@pytest.mark.parametrize(
    "label, expected",
    [
        ("private_person", "<PRIVATE_PERSON>"),
        ("account_number", "<ACCOUNT_NUMBER>"),
        ("a-b c", "<A_B_C>"),
        ("__secret__", "<SECRET>"),
        ("!!!", "<REDACTED>"),
    ],
)
def test_placeholder_for(label, expected):
    assert placeholder_for(label) == expected


def test_redact_text_substitutes_placeholders_and_keeps_everything_else():
    text = "Hi Alice, mail a@b.co"
    spans = [
        DetectedSpan("private_person", 3, 8, "Alice", "<PRIVATE_PERSON>"),
        DetectedSpan("private_email", 15, 21, "a@b.co", "<PRIVATE_EMAIL>"),
    ]

    assert redact_text(text, spans) == "Hi <PRIVATE_PERSON>, mail <PRIVATE_EMAIL>"


def test_redact_text_without_spans_returns_the_text_unchanged():
    assert redact_text("nothing here", []) == "nothing here"
