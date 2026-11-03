"""Lossless text ingress for the Rust-owned causal training/evaluation contract."""
from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from . import CausalCompletion, CausalTargetAlignment


class _Encoding(Protocol):
    @property
    def ids(self) -> Sequence[int]: ...
    @property
    def offsets(self) -> Sequence[Sequence[int]]: ...


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> _Encoding: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


def _bounded_ids(encoding: _Encoding, minimum: int, maximum: int) -> Sequence[int]:
    # tokenizers.Encoding exposes its size before allocating its Python lists.
    n = len(encoding) if isinstance(encoding, Sized) else None
    if n is not None and not minimum <= n <= maximum:
        raise ValueError("encoded token count outside causal alignment budget")
    ids = encoding.ids
    if not minimum <= len(ids) <= maximum or (n is not None and n != len(ids)):
        raise ValueError("encoded token count differs or exceeds causal alignment budget")
    return ids


def _aligned_source(
    tokenizer: _Tokenizer, text: str, target_spans: Sequence[Sequence[int]],
) -> tuple[CausalTargetAlignment, list[int]]:
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
    encoded_ids = _bounded_ids(encoding, 2, CausalTargetAlignment.MAX_TOKENS)
    alignment = CausalTargetAlignment(len(text), encoded_ids, encoding.offsets, target_spans)
    ids = alignment.token_ids
    for first, stop, left, right in alignment.runs:
        decoded = tokenizer.decode(ids[first:stop], skip_special_tokens=False)
        if type(decoded) is not str or decoded != text[left:right]:
            raise ValueError(f"tokenizer does not reproduce the exact source at [{left}, {right})")
    return alignment, ids


def encode_supervised_document(
    tokenizer: _Tokenizer, text: str, target_spans: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Encode once and return IDs and unshifted labels. Spans index Python Unicode characters.

    The codec must report full-source offsets and reproduce each contiguous prompt/target
    run exactly. No templates, EOS tokens, normalization or padding are added.
    """
    alignment, ids = _aligned_source(tokenizer, text, target_spans)
    return ids, alignment.labels


def encode_supervised_completion(
    tokenizer: _Tokenizer, prompt: str, answer: str, *, terminator: str = "",
) -> CausalCompletion:
    """Bind independent inference IDs to one training example; never infer a delimiter.

    Training targets the answer and optional terminator. Evaluation scores only the
    answer. Generation receives the independently encoded prompt, without an answer length.
    Refuse tokenizers that change the prefix depending on the following text.
    """
    from . import CausalTargetAlignment

    if type(prompt) is not str or type(answer) is not str or type(terminator) is not str or not prompt or not answer:
        raise ValueError("expected a nonempty prompt and answer and a literal string terminator")
    if len(prompt) + len(answer) + len(terminator) > CausalTargetAlignment.MAX_SOURCE_UNITS:
        raise ValueError("completion source exceeds causal alignment budget")
    content = prompt + answer
    text = content + terminator
    alignment, ids = _aligned_source(tokenizer, text, [(len(prompt), len(content))])
    prompt_ids = _bounded_ids(tokenizer.encode(prompt, add_special_tokens=False), 1, len(ids) - 1)
    content_ids = _bounded_ids(tokenizer.encode(content, add_special_tokens=False), 2, len(ids)) if terminator else ids
    completion = alignment.completion(prompt_ids, content_ids)
    if (tokenizer.decode(ids, skip_special_tokens=False) != text or
            (terminator and tokenizer.decode(ids[:completion.answer_end], skip_special_tokens=False) != content)):
        raise ValueError("tokenizer does not reproduce the exact joined completion source")
    return completion
