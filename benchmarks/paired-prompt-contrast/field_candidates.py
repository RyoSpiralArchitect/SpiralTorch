"""Train-only first-token candidate sets for the closed three-field extraction benchmark."""
from collections import Counter
from dataclasses import dataclass
from itertools import islice
import math
import struct


FIELDS = ("color", "shape", "material")
MAX_VOCABULARY = 1_048_576


def _size(v):
    if type(v) is not int or not 2 <= v <= MAX_VOCABULARY:
        raise ValueError("invalid vocabulary size")


def _id(i, size):
    return type(i) is int and 0 <= i < size


@dataclass(frozen=True)
class RequestRouter:
    vocabulary: int
    rules: tuple

    def __post_init__(self):
        _size(self.vocabulary)
        if (type(self.rules) is not tuple or len(self.rules) != 6 or
                any(type(r) is not tuple or len(r) != 2 or r[0] not in FIELDS for r in self.rules) or
                Counter(f for f, _ in self.rules) != {f: 2 for f in FIELDS}):
            raise ValueError("expected two request forms per field")
        patterns = [s for _, s in self.rules]
        if any(type(s) is not tuple or not 1 <= len(s) <= 32 or any(not _id(i, self.vocabulary) for i in s) for s in patterns):
            raise ValueError("invalid request suffix")
        for i, a in enumerate(patterns):
            if any(i != j and a[-len(b):] == b for j, b in enumerate(patterns)):
                raise ValueError("ambiguous request suffix")

    @classmethod
    def gpt2(cls, vocab, vocabulary=50_257):
        """Use only the fixed request grammar, never answer words, from the pinned GPT-2 token dictionary."""
        lf, space = "\u010a", "\u0120"
        rules = []
        for field in FIELDS:
            for request in [("What", space+"is", space+"the", space+field, "?"),
                            ("Return", space+"the", space+field, ".")]:
                try:
                    ids = tuple(vocab[token] for token in (lf, *request, lf, "Answer", ":"))
                except KeyError as exc:
                    raise ValueError("token dictionary does not encode this request grammar") from exc
                rules.append((field, ids))
        return cls(vocabulary, tuple(rules))

    def match(self, prompt):
        """Identify the final request and return (field, matched suffix). Never parse the record values."""
        remaining = self.rules
        length = len(prompt)
        for pos in range(1, min(length, max(len(s) for _, s in remaining)) + 1):
            token = prompt[length - pos]
            if not _id(token, self.vocabulary):
                raise ValueError("invalid request token")
            remaining = tuple((f, s) for f, s in remaining if len(s) >= pos and s[-pos] == token)
            if not remaining:
                break
            for field, suffix in remaining:
                if pos == len(suffix):
                    return field, suffix
        raise ValueError("unknown request: not routing from record contents or answer labels")


@dataclass(frozen=True)
class FieldVocabulary:
    vocabulary: int
    groups: tuple

    def __post_init__(self):
        _size(self.vocabulary)
        if type(self.groups) is not tuple or len(self.groups) != len(FIELDS):
            raise ValueError("missing field vocabulary")
        for ids in self.groups:
            if (type(ids) is not tuple or not 0 < len(ids) < self.vocabulary or
                    any(not _id(i, self.vocabulary) for i in ids) or tuple(sorted(set(ids))) != ids):
                raise ValueError("expected a sorted, nonempty field support")

    @classmethod
    def fit(cls, examples, vocabulary):
        """Only (requested field, supervised first token) pairs from the caller's training split."""
        _size(vocabulary)
        seen = {f: set() for f in FIELDS}
        for field, target in examples:
            if field not in seen or not _id(target, vocabulary):
                raise ValueError("invalid training example")
            seen[field].add(target)
        return cls(vocabulary, tuple(tuple(sorted(seen[f])) for f in FIELDS))

    def support(self, field):
        if field not in FIELDS:
            raise ValueError("unknown field; no pooled vocabulary fallback")
        return self.groups[FIELDS.index(field)]

    def pooled(self):
        return tuple(sorted(set().union(*self.groups)))


def rank_support(raw, members, vocabulary):
    """Rank only the listed first tokens; no gold label, teacher, or forced padding. Ties use low IDs."""
    _size(vocabulary)
    if not isinstance(raw, bytes) or len(raw) != 4 * vocabulary:
        raise ValueError("expected one full little-endian f32 row")
    ids = tuple(islice(members, vocabulary + 1))
    if not 0 < len(ids) <= vocabulary or any(not _id(i, vocabulary) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid candidate IDs")
    def order(i):
        v, = struct.unpack_from("<f", raw, 4*i)
        if not math.isfinite(v):
            raise ValueError("non-finite candidate logit")
        return -v, i
    return tuple(sorted(ids, key=order))
