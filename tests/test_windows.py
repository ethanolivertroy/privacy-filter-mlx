"""Splitting a long token sequence into model calls."""

import math
import random

import pytest

from opf_mlx.windows import Window, plan_windows


def test_original_policy_cuts_consecutive_windows_and_keeps_every_prediction():
    assert plan_windows(10, 4) == [
        Window(start=0, stop=4, keep_start=0, keep_stop=4),
        Window(start=4, stop=8, keep_start=4, keep_stop=8),
        Window(start=8, stop=10, keep_start=8, keep_stop=10),
    ]


def test_text_that_fits_is_one_window():
    assert plan_windows(7, 128000) == [Window(start=0, stop=7, keep_start=0, keep_stop=7)]
    assert plan_windows(4, 4) == [Window(start=0, stop=4, keep_start=0, keep_stop=4)]


def test_no_tokens_means_no_windows():
    assert plan_windows(0, 16) == []


@pytest.mark.parametrize("n_ctx", [0, -1])
def test_non_positive_context_is_rejected(n_ctx):
    with pytest.raises(ValueError, match="n_ctx"):
        plan_windows(10, n_ctx)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="policy"):
        plan_windows(10, 4, policy="sliding")


def _kept(windows: list[Window]) -> list[int]:
    return [t for w in windows for t in range(w.keep_start, w.keep_stop)]


@pytest.mark.parametrize("n_tokens, n_ctx", [(1, 1), (9, 3), (10, 3), (1000, 7), (5, 100)])
def test_original_policy_never_drops_or_repeats_a_token(n_tokens, n_ctx):
    windows = plan_windows(n_tokens, n_ctx)

    assert _kept(windows) == list(range(n_tokens))
    assert all(0 < w.stop - w.start <= n_ctx for w in windows)


# --- halo policy -----------------------------------------------------------
# The halo policy gives every kept token `halo` tokens of context on each side
# (or as many as the text has), so predictions do not depend on where the
# windows fall.

HALO_SHAPES = [(100, 20, 4), (101, 20, 4), (37, 12, 5), (9, 20, 4), (64, 9, 4)]


def _random_shapes(count: int) -> list[tuple[int, int, int]]:
    """(n_tokens, n_ctx, halo) triples in which `n_ctx` has room for a kept token."""
    rng = random.Random(20261123)
    shapes = []
    for _ in range(count):
        halo = rng.randint(0, 12)
        shapes.append((rng.randint(1, 400), rng.randint(2 * halo + 1, 2 * halo + 30), halo))
    return shapes


def test_halo_policy_keeps_all_of_a_window_but_its_halos():
    # 30 tokens, 12 per window, 3 of context on each side. The first window has
    # no left halo to pay for and the last no right one.
    assert plan_windows(30, 12, policy="halo", halo=3) == [
        Window(start=0, stop=12, keep_start=0, keep_stop=9),
        Window(start=6, stop=18, keep_start=9, keep_stop=15),
        Window(start=12, stop=24, keep_start=15, keep_stop=21),
        Window(start=18, stop=30, keep_start=21, keep_stop=30),
    ]


@pytest.mark.parametrize("n_tokens, n_ctx, halo", [*HALO_SHAPES, (1, 3, 1)])
def test_halo_policy_keeps_every_token_exactly_once(n_tokens, n_ctx, halo):
    windows = plan_windows(n_tokens, n_ctx, policy="halo", halo=halo)

    assert _kept(windows) == list(range(n_tokens))


@pytest.mark.parametrize("n_tokens, n_ctx, halo", HALO_SHAPES)
def test_halo_policy_never_feeds_more_than_n_ctx_tokens(n_tokens, n_ctx, halo):
    for w in plan_windows(n_tokens, n_ctx, policy="halo", halo=halo):
        assert 0 <= w.start <= w.keep_start < w.keep_stop <= w.stop <= n_tokens
        assert w.stop - w.start <= n_ctx


@pytest.mark.parametrize("n_tokens, n_ctx, halo", HALO_SHAPES)
def test_halo_policy_gives_each_kept_token_full_context_on_both_sides(n_tokens, n_ctx, halo):
    for w in plan_windows(n_tokens, n_ctx, policy="halo", halo=halo):
        assert w.keep_start - w.start >= min(halo, w.keep_start)
        assert w.stop - w.keep_stop >= min(halo, n_tokens - w.keep_stop)


def test_halo_policy_holds_its_contract_for_any_shape():
    for n_tokens, n_ctx, halo in _random_shapes(2000):
        windows = plan_windows(n_tokens, n_ctx, policy="halo", halo=halo)

        assert _kept(windows) == list(range(n_tokens)), (n_tokens, n_ctx, halo)
        for w in windows:
            assert 0 <= w.start <= w.keep_start < w.keep_stop <= w.stop <= n_tokens, (n_tokens, n_ctx, halo)
            assert w.stop - w.start <= n_ctx, (n_tokens, n_ctx, halo)
            assert w.keep_start - w.start >= min(halo, w.keep_start), (n_tokens, n_ctx, halo)
            assert w.stop - w.keep_stop >= min(halo, n_tokens - w.keep_stop), (n_tokens, n_ctx, halo)


def test_halo_policy_uses_as_few_windows_as_possible():
    # A window that reaches neither end of the text keeps at most n_ctx - 2 * halo
    # tokens; the first and the last keep at most n_ctx - halo.
    for n_tokens, n_ctx, halo in _random_shapes(2000):
        if n_tokens <= n_ctx:
            fewest = 1
        else:
            fewest = 2 + math.ceil(max(0, n_tokens - 2 * (n_ctx - halo)) / (n_ctx - 2 * halo))

        assert len(plan_windows(n_tokens, n_ctx, policy="halo", halo=halo)) == fewest, (n_tokens, n_ctx, halo)


@pytest.mark.parametrize("n_tokens, n_ctx", [(1, 1), (9, 3), (10, 3), (1000, 7), (5, 100)])
def test_a_halo_of_zero_is_the_original_policy(n_tokens, n_ctx):
    assert plan_windows(n_tokens, n_ctx, policy="halo", halo=0) == plan_windows(n_tokens, n_ctx)


def test_halo_policy_with_no_tokens_means_no_windows():
    assert plan_windows(0, 16, policy="halo", halo=4) == []


@pytest.mark.parametrize("n_tokens", [100, 5, 0])
@pytest.mark.parametrize("n_ctx, halo", [(8, 4), (7, 4), (1, 1)])
def test_halo_policy_needs_room_for_at_least_one_kept_token(n_tokens, n_ctx, halo):
    # The answer does not depend on the text: a setting that cannot work for a
    # long input is refused for a short one too, so it is found on the first call.
    with pytest.raises(ValueError, match=f"halo.*n_ctx of at least {2 * halo + 1}"):
        plan_windows(n_tokens, n_ctx, policy="halo", halo=halo)


def test_negative_halo_is_rejected():
    with pytest.raises(ValueError, match="halo must not be negative"):
        plan_windows(10, 4, policy="halo", halo=-1)
