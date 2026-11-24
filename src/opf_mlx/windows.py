"""Deciding which token ranges go through the model together.

A window is one forward pass. `start:stop` is what the model is fed and
`keep_start:keep_stop` is the part of its output that is kept. Whatever the
policy, the kept ranges must cover every token exactly once: inputs are split,
never truncated.
"""

from __future__ import annotations

from dataclasses import dataclass

POLICIES: tuple[str, ...] = ("original", "halo")


@dataclass(frozen=True)
class Window:
    start: int  # first token fed to the model
    stop: int  # one past the last token fed to the model
    keep_start: int  # first token whose prediction is kept
    keep_stop: int  # one past the last token whose prediction is kept


def plan_windows(n_tokens: int, n_ctx: int, *, policy: str = "original", halo: int = 0) -> list[Window]:
    """Split `n_tokens` tokens into model calls of at most `n_ctx` tokens.

    `original` reproduces the reference: consecutive windows with no overlap.
    `halo` surrounds each kept range with `halo` tokens of context.
    """
    if n_ctx <= 0:
        raise ValueError(f"n_ctx must be positive (got {n_ctx})")
    if policy == "original":
        return _consecutive(n_tokens, n_ctx)
    if policy == "halo":
        return _with_halo(n_tokens, n_ctx, halo)
    raise ValueError(f"unknown window policy {policy!r}; expected one of {POLICIES}")


def _consecutive(n_tokens: int, n_ctx: int) -> list[Window]:
    """The reference's `example_to_windows`: stride equals window size."""
    return [
        Window(start, min(start + n_ctx, n_tokens), start, min(start + n_ctx, n_tokens))
        for start in range(0, n_tokens, n_ctx)
    ]


def require_room_for_halo(n_ctx: int, halo: int) -> None:
    """Refuse a window too small to keep one token between two halos.

    The answer does not depend on the text. A setting that cannot work for a
    long input is refused for a short one too, so it is found on the first
    call and not with the first long document.
    """
    if halo < 0:
        raise ValueError(f"halo must not be negative (got {halo})")
    if n_ctx <= 2 * halo:
        raise ValueError(
            f"a halo of {halo} tokens on each side leaves no room for a kept token in a window of {n_ctx}; "
            f"the halo policy needs n_ctx of at least {2 * halo + 1}"
        )


def _with_halo(n_tokens: int, n_ctx: int, halo: int) -> list[Window]:
    """Windows whose kept tokens all see `halo` tokens of context on both sides.

    Why this exists: the model is local. Each layer's attention reaches a fixed
    distance left and right, and nothing else mixes tokens, so a token's logits
    depend only on tokens within `ModelConfig.receptive_field` of it (1024 each
    way for the released model). If every kept token has that much real context
    inside its window, or the text simply ends first, then it sees exactly the
    tokens it would see in one pass over the whole text. With the `original`
    policy, tokens near a cut see less context than they would have.

    The scores are still not those of one pass bit for bit. Positions restart
    at 0 in every window, and the rotary table is rounded differently at
    different positions (docs/architecture.md, section 5). What a halo removes
    is the missing context, which is the larger effect.

    Each window is as long as `n_ctx` allows and keeps everything but its
    halos. The first window has no left halo to pay for and the last no right
    one, so no plan with fewer windows exists.
    """
    require_room_for_halo(n_ctx, halo)
    windows: list[Window] = []
    keep_start = 0
    while keep_start < n_tokens:
        start = max(0, keep_start - halo)
        stop = min(start + n_ctx, n_tokens)
        keep_stop = n_tokens if stop == n_tokens else stop - halo
        windows.append(Window(start, stop, keep_start, keep_stop))
        keep_start = keep_stop
    return windows
