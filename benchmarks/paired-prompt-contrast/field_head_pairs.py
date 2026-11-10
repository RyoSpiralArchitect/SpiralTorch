"""Training-only first-token rivals for answers with a literal shorter suffix in the same field."""


def suffix_pairs(menu):
    """Return (short, long) content tuples. No text normalizer, logits or evaluation examples."""
    members = set(menu.choices)
    return tuple((short, long) for long in menu.choices for start in range(1, len(long))
                 if (short := long[start:]) in members and short[0] != long[0])


def training_head_rivals(menu):
    """Map each complete training label to competing head IDs, in both directions.

    An empty tuple means no contrast (keep the ordinary objective). Multiple
    longer forms with the same head contribute one class, not extra weight.
    These labels are for supervised updates, never for selecting an inference menu.
    """
    rivals = {c: set() for c in menu.choices}
    for short, long in suffix_pairs(menu):
        rivals[short].add(long[0])
        rivals[long].add(short[0])
    return {c: tuple(sorted(heads)) for c, heads in rivals.items()}


def rotated_pair_control(menu):
    """Break a disjoint suffix matching without changing field, sides, or rival-head counts.

    Sort by short label, then rotate its longer partner by one. Reject nested or
    ambiguous matches instead of silently changing class exposure in the control.
    """
    pairs = sorted(suffix_pairs(menu))
    heads = [c[0] for pair in pairs for c in pair]
    if len(pairs) < 2 or len(set(heads)) != len(heads):
        raise ValueError("control needs disjoint pairs with unique heads")
    return tuple((s, pairs[(i + 1) % len(pairs)][1]) for i, (s, _) in enumerate(pairs))
