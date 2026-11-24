"""Tokenization and character offsets, using the original tiktoken encoding.

The reference tokenizes with tiktoken (`config.encoding`, which is
`o200k_base`), not with the `tokenizer.json` that ships beside the
transformers-format weights. Offsets are computed from token bytes, exactly as
in `opf/_core/spans.py`.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Sequence

import tiktoken


class Tokenizer:
    """A tiktoken encoding plus the byte-to-character bookkeeping spans need."""

    def __init__(self, encoding_name: str) -> None:
        self.name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def max_token_id(self) -> int:
        return int(self._encoding.max_token_value)

    def encode(self, text: str) -> list[int]:
        """Token ids for `text`. Special-token text in the input maps to its special id."""
        return [int(token) for token in self._encoding.encode(text, allowed_special="all")]

    def decode_with_offsets(self, token_ids: Sequence[int]) -> tuple[str, list[int], list[int]]:
        """Decode tokens and give each one a `[start, end)` range of characters.

        Tokens are byte strings, and a token boundary can fall inside a
        multi-byte character. A token's range therefore starts at the character
        containing its first byte and ends after the character containing its
        last byte. Offsets index the decoded string in code points.
        """
        token_bytes = [self._encoding.decode_single_token_bytes(int(token)) for token in token_ids]
        text = b"".join(token_bytes).decode("utf-8", errors="replace")
        if not token_bytes:
            return text, [], []

        char_byte_starts: list[int] = []
        char_byte_ends: list[int] = []
        cursor = 0
        for char in text:
            char_byte_starts.append(cursor)
            cursor += len(char.encode("utf-8"))
            char_byte_ends.append(cursor)

        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for raw in token_bytes:
            byte_start, byte_end = cursor, cursor + len(raw)
            cursor = byte_end
            start = bisect_right(char_byte_ends, byte_start)
            end = max(bisect_left(char_byte_starts, byte_end), start)
            starts.append(start)
            ends.append(end)

        if ends[-1] != len(text):
            raise ValueError(
                f"Character length mismatch for decoded text (tokens={ends[-1]}, text={len(text)}); "
                "the token bytes are not valid UTF-8"
            )
        return text, starts, ends
