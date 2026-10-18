"""In-memory HF map dataset driven by Rust-owned single-document windows."""
from __future__ import annotations

import operator
from typing import Any, Sequence

__all__ = ["HfGeometryDataset"]


def _integer(value):
    dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or str(dtype) in ("bool", "torch.bool"):
        raise TypeError("expected integers, not booleans")
    return operator.index(value)


class HfGeometryDataset:
    """Freeze encoded documents; never pack documents, pad, or invent EOS.

    Each row uses a complete Rust-planned fixed-width context. The first label
    is ignored and incomplete document tails are dropped, as reported by plan.
    A plan ID binds lengths/layout, not token content or tokenizer identity.
    This map dataset needs neither Torch nor Transformers at import time.
    """

    def __init__(self, token_documents: Sequence[Sequence[int]], context_length: int):
        from . import CausalDocumentWindows
        context_length = _integer(context_length)
        count = len(token_documents)
        if count > CausalDocumentWindows.MAX_DOCUMENTS:
            raise ValueError("document count exceeds budget")
        lengths = [len(token_documents[i]) for i in range(count)]
        self._plan = CausalDocumentWindows(lengths, context_length)
        if not len(self._plan):
            raise ValueError("no full document windows; shorten context or supply longer documents")
        documents = []
        for index in range(count):
            document = token_documents[index]
            row = []
            for value in document:
                token = _integer(value)
                if token < 0 or token > 0xFFFFFFFF:
                    raise ValueError("token IDs must fit unsigned 32-bit integers")
                row.append(token)
            documents.append(tuple(row))
        if [len(d) for d in documents] != lengths:
            raise ValueError("documents changed while constructing the dataset")
        self._documents = tuple(documents)
        self._context_length = context_length

    def __reduce__(self):
        # Spawned workers rebuild the native plan through its validated ctor.
        return type(self), (self._documents, self._context_length)

    @classmethod
    def from_texts(cls, tokenizer: Any, documents: Sequence[str], context_length: int,
                   *, add_special_tokens: bool = True) -> HfGeometryDataset:
        if isinstance(documents, (str, bytes)):
            raise TypeError("pass a sequence of documents, not one bare string")
        from . import CausalDocumentWindows
        count = len(documents)
        if count > CausalDocumentWindows.MAX_DOCUMENTS:
            raise ValueError("document count exceeds budget")
        texts = [documents[i] for i in range(count)]
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("provide a nonempty sequence of text documents")
        encoded = tokenizer(texts, padding=False, truncation=False,
                            add_special_tokens=add_special_tokens, return_attention_mask=False)
        rows = encoded["input_ids"]
        if len(rows) != len(texts):
            raise ValueError("tokenizer must return one token sequence per document")
        return cls(rows, context_length)

    @property
    def plan(self):
        return self._plan

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("window index out of range")
        document, start, length = self._plan.window(index)
        tokens = list(self._documents[document][start:start + length])
        labels = list(tokens)
        labels[0] = -100
        return {"input_ids": tokens, "labels": labels, "attention_mask": [1] * length}
