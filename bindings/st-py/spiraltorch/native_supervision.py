"""Lossless text ingress for the Rust-owned causal training/evaluation contract."""
from __future__ import annotations

from collections.abc import Sized
from typing import Protocol, Sequence


class _Encoding(Protocol):
    @property
    def ids(self) -> Sequence[int]: ...
    @property
    def offsets(self) -> Sequence[Sequence[int]]: ...


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> _Encoding: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


def encode_supervised_document(
    tokenizer: _Tokenizer, text: str, target_spans: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Encode the whole text once. Spans index Python Unicode characters, not UTF-8 bytes.

    Return one document's IDs and unshifted labels. The supplied codec must report
    full-source offsets and decode each contiguous prompt/target run losslessly.
    No templates, separators, EOS tokens, normalization or padding are added here.
    """
    from . import CausalTargetAlignment

    if type(text) is not str or not 1 <= len(text) <= CausalTargetAlignment.MAX_SOURCE_UNITS:
        raise ValueError("text must be a nonempty string within the causal alignment budget")
    if not 1 <= len(target_spans) <= min(len(text), CausalTargetAlignment.MAX_TARGET_SPANS):
        raise ValueError("supervised span count outside causal alignment budget")
    if getattr(tokenizer, "padding", None) is not None or getattr(tokenizer, "truncation", None) is not None:
        raise ValueError("disable tokenizer padding and truncation for native text supervision")
    if getattr(getattr(tokenizer, "model", None), "dropout", None) not in (0, None):
        raise ValueError("disable tokenizer model dropout for reproducible supervision")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    # tokenizers.Encoding exposes its size without materializing two Python lists.
    n = len(encoding) if isinstance(encoding, Sized) else None
    if n is not None and not 2 <= n <= CausalTargetAlignment.MAX_TOKENS:
        raise ValueError("encoded token count outside causal alignment budget")
    encoded_ids = encoding.ids
    if not 2 <= len(encoded_ids) <= CausalTargetAlignment.MAX_TOKENS or (n is not None and n != len(encoded_ids)):
        raise ValueError("encoded token count differs or exceeds causal alignment budget")
    alignment = CausalTargetAlignment(len(text), encoded_ids, encoding.offsets, target_spans)
    ids = alignment.token_ids
    for first, stop, left, right in alignment.runs:
        decoded = tokenizer.decode(ids[first:stop], skip_special_tokens=False)
        if type(decoded) is not str or decoded != text[left:right]:
            raise ValueError(f"tokenizer does not reproduce the exact source at [{left}, {right})")
    return ids, alignment.labels
