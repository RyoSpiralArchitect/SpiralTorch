"""Closed-menu whole-line comparisons. Fitting and ranking never take query labels."""
from dataclasses import dataclass
import math

from field_candidates import FIELDS, _id, _size

MAX_CHOICES = 64
MAX_CONTENT = 32


def _content(ids, vocabulary, stop):
    if (type(ids) is not tuple or not 1 <= len(ids) <= MAX_CONTENT or
            any(not _id(i, vocabulary) or i == stop for i in ids)):
        raise ValueError("expected nonempty, bounded content without a stop ID")
    return ids


def _domain(vocabulary, stop):
    _size(vocabulary)
    if not _id(stop, vocabulary):
        raise ValueError("invalid stop ID")


def supervised_content(prompt, document, labels, vocabulary, stop):
    """Extract exactly one fully supervised, stop-terminated line after this exact prompt."""
    _domain(vocabulary, stop)
    p = len(prompt)
    if (not p or len(document) != len(labels) or not 2 <= len(document)-p <= MAX_CONTENT+1 or
            any(not _id(i, vocabulary) for i in document) or
            any(type(i) is not int for i in prompt) or any(type(i) is not int for i in labels) or
            tuple(document[:p]) != tuple(prompt) or any(i != -100 for i in labels[:p]) or
            tuple(labels[p:]) != tuple(document[p:]) or document[-1] != stop):
        raise ValueError("prompt, answer mask, or final stop differs")
    return _content(tuple(document[p:-1]), vocabulary, stop)


@dataclass(frozen=True)
class CompletionMenu:
    vocabulary: int
    stop: int
    choices: tuple

    def __post_init__(self):
        _domain(self.vocabulary, self.stop)
        if type(self.choices) is not tuple or not 1 <= len(self.choices) <= MAX_CHOICES:
            raise ValueError("invalid menu size")
        for answer in self.choices:
            _content(answer, self.vocabulary, self.stop)
        if tuple(sorted(set(self.choices))) != self.choices:
            raise ValueError("menu must contain unique answers in token-ID order")

    @property
    def first_ids(self):
        return tuple(sorted({a[0] for a in self.choices}))

    @property
    def edges(self):
        """Non-root prefix/next-ID pairs; a candidate that ends needs a stop edge."""
        return tuple(sorted({(a[:i], t) for a in self.choices
                             for i, t in enumerate((*a[1:], self.stop), 1)}))


def fit_menus(training, vocabulary, stop):
    """Only (requested field, full training content IDs); no question text or gold at inference."""
    _domain(vocabulary, stop)
    groups = {f: set() for f in FIELDS}
    for field, content in training:
        if field not in groups:
            raise ValueError("unknown training field")
        group = groups[field]
        group.add(_content(content, vocabulary, stop))
        if len(group) > MAX_CHOICES:
            raise ValueError("field menu exceeds budget")
    return {f: CompletionMenu(vocabulary, stop, tuple(sorted(g))) for f, g in groups.items()}


def _nll(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative full-vocabulary NLL")
    return float(value)


@dataclass(frozen=True)
class ScoredLine:
    content: tuple
    token_nll: tuple  # Includes exactly one stop.

    @property
    def line_nll(self):
        return math.fsum(self.token_nll)

    @property
    def content_nll(self):
        return math.fsum(self.token_nll[:-1])


def score_menu(menu, first_nll, continuation_nll, *, calls):
    """Rank exact terminated lines by joint likelihood. Neither callback nor rank uses a label.

    first_nll maps candidate heads to NLLs on the unmodified prompt, normalized
    over the whole vocabulary. continuation_nll(prefix, next_id) must append
    *only* prefix to that prompt and return the next ID's full-vocabulary NLL.
    Shared edges are evaluated once; exceptions propagate without retry.
    """
    edges = menu.edges
    if type(calls) is not int or calls != len(edges):
        raise ValueError("continuation budget differs from the menu")
    if (any(not _id(i, menu.vocabulary) for i in first_nll) or
            set(first_nll) != set(menu.first_ids)):
        raise ValueError("first readout must cover exactly the candidate heads")
    first = {i: _nll(first_nll[i]) for i in menu.first_ids}
    observed = {e: _nll(continuation_nll(*e)) for e in edges}
    lines = []
    for a in menu.choices:
        suffix = (observed[a[:i], t] for i, t in enumerate((*a[1:], menu.stop), 1))
        lines.append(ScoredLine(a, (first[a[0]], *suffix)))
    return tuple(sorted(lines, key=lambda r: (r.line_nll, r.content)))
