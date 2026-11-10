"""Model-free readback of closed-menu scores. No decoder, label, or scorer callback in reconstruction."""
from dataclasses import dataclass
import math
import struct

from field_candidates import _id


def retained_heads(raw, menu):
    """Compute full-vocabulary head NLLs independently of the native scorer."""
    if not isinstance(raw, bytes) or len(raw) != 4 * menu.vocabulary:
        raise ValueError("expected a full little-endian f32 row")
    logits = [x for x, in struct.iter_unpack("<f", raw)]
    if not all(map(math.isfinite, logits)):
        raise ValueError("non-finite retained logit")
    winner = max(range(len(logits)), key=logits.__getitem__)
    peak = logits[winner]
    remainder = math.fsum(math.exp(x - peak) for i, x in enumerate(logits) if i != winner)
    normalizer = math.log1p(remainder)
    return {t: (peak - logits[t]) + normalizer for t in menu.first_ids}


def _cost(n):
    if type(n) not in (float, int) or not math.isfinite(n) or n < 0:
        raise ValueError("invalid token NLL")
    return float(n)


def checked_native_nll(score, target, vocabulary):
    """Check sparse native score invariants, not a fresh calculation of the vocabulary normalizer."""
    t, r, g = (score[k] for k in ("target_id", "strongest_rival_id", "greedy_id"))
    rank = score["target_rank"]
    if (type(vocabulary) is not int or vocabulary < 2 or type(score["vocabulary"]) is not int or
            score["vocabulary"] != vocabulary or not _id(target, vocabulary) or t != target or
            not all(_id(i, vocabulary) for i in (t, r, g)) or r == t or
            type(rank) is not int or not 1 <= rank <= vocabulary):
        raise ValueError("wrong native target, vocabulary or rank")
    gold, rival, margin = (score[k] for k in ("target_logit", "strongest_rival_logit", "target_minus_best_rival"))
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (gold, rival, margin)):
        raise ValueError("non-finite target or rival")
    winner = t if (gold > rival or (gold == rival and t < r)) else r
    if margin != gold - rival or g != winner or (g == t) != (rank == 1):
        raise ValueError("inconsistent native competitor")
    # At least the strongest rival exists; all V-1 rivals are at most as strong.
    def softplus(x):
        return max(x, 0.) + math.log1p(math.exp(-abs(x)))
    lo = max(math.log(rank), softplus(-margin))
    hi = softplus(math.log(vocabulary - 1) - margin)
    nll = _cost(score["nll"])
    tol = 1e-12 * max(1., nll, lo, hi)
    if not lo - tol <= nll <= hi + tol:
        raise ValueError("impossible full-vocabulary NLL for the reported margin/rank")
    return nll


@dataclass(frozen=True)
class LineEvidence:
    content: tuple
    parts: tuple  # Head, remaining content, exactly one stop.

    @property
    def total(self):
        return math.fsum(self.parts)

    def cost(self, mode):
        if mode == "line":
            return self.total
        if mode == "first":
            return self.parts[0]
        if mode == "content":
            return math.fsum(self.parts[:-1])
        if mode == "mean_line":
            return self.total / len(self.parts)
        raise ValueError("unknown scoring mode")


def rebuild(menu, first_nll, observations):
    """Accept each expected (content prefix, next ID, NLL) once; reject loss, repeats or extras.

    Per-call source/owner, prefix hash and start/return identity must be bound
    by the caller. The reconstructed lines are ordered before any label join.
    """
    if set(first_nll) != set(menu.first_ids) or any(not _id(i, menu.vocabulary) for i in first_nll):
        raise ValueError("different first-token support")
    heads = {t: _cost(n) for t, n in first_nll.items()}
    paths = {}
    for c in menu.choices:
        seq = (*c, menu.stop)
        paths[c] = [(seq[:i], seq[i]) for i in range(1, len(seq))]
    missing = {edge for path in paths.values() for edge in path}
    costs = {}
    for prefix, token, nll in observations:
        if (type(prefix) is not tuple or not _id(token, menu.vocabulary) or
                any(not _id(t, menu.vocabulary) for t in prefix) or (prefix, token) not in missing):
            raise ValueError("unexpected or duplicate continuation")
        edge = (prefix, token)
        missing.remove(edge)
        costs[edge] = _cost(nll)
    if missing:
        raise ValueError("missing content or stop score")
    lines = (LineEvidence(c, (heads[c[0]], *(costs[e] for e in path))) for c, path in paths.items())
    return tuple(sorted(lines, key=lambda x: (x.total, x.content)))


def contrast(lines, gold):
    """Post-score diagnosis against the best *complete* wrong line, never a different rival per component.

    Positive is a disadvantage to gold. A missing gold or missing rival has no
    identifiable contrast; a tied total does not imply a correct prediction.
    """
    by_content = {x.content: x for x in lines}
    if gold not in by_content or len(by_content) < 2:
        return None
    a = by_content[gold]
    b = min((x for x in lines if x.content != gold), key=lambda x: (x.total, x.content))
    head = a.parts[0] - b.parts[0]
    body = math.fsum((*a.parts[1:-1], *(-x for x in b.parts[1:-1])))
    stop = a.parts[-1] - b.parts[-1]
    return dict(rival=b.content, head=head, body=body, stop=stop, total=a.total - b.total)
