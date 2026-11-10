"""Post-hoc, model-free intercept check. Only fit labels enter the deployable threshold."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_prefix_study import obj, put, read, require
from partner_routing_study import ROW


def sigmoid(x):
    e = math.exp(-abs(x))
    return (1.0 if x >= 0 else e) / (1.0 + e)


def offset(gaps, labels):
    require(
        len(gaps) == len(labels) and set(labels) == {False, True} and all(map(math.isfinite, gaps)),
        "invalid calibration rows",
    )
    p = sum(labels) / len(labels)
    lo, hi = -max(gaps) - 40.0, -min(gaps) + 40.0
    for _ in range(80):
        mid = lo + (hi - lo) * 0.5
        if math.fsum(sigmoid(x + mid) for x in gaps) / len(gaps) < p:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) * 0.5
    require(
        abs(math.fsum(sigmoid(x + result) for x in gaps) / len(gaps) - p) < 1e-12,
        "binary likelihood root not bracketed",
    )
    return result


def correct(gaps, labels, bias, eps=5e-6):
    return sum((x + bias if y else -x - bias) > eps for x, y in zip(gaps, labels, strict=True))


def ceiling(gaps, labels):
    # With ties a threshold exactly on an observed gap cannot beat both adjacent open intervals.
    events = sorted({-x for x in gaps})
    biases = [events[0] - 1.0, events[-1] + 1.0, *((x + (y - x) / 2) for x, y in zip(events, events[1:]))]
    return max(correct(gaps, labels, b, 0.0) for b in biases)


def run(root, pp, sp, out):
    p = obj(root / "protocol/plan.json", pp)
    s = obj(root / "scored/complete.json", sp)
    require(s["plan"] == pp and s["rows"] == 6528, "wrong scoring job")
    folds = ("train0", "train1", "held0", "held1", "transfer0", "transfer1")
    qs = [q for f in folds for q in obj(root / "protocol" / (f + ".json"), p["folds"][f])]
    labels = [q[2 if q[1] == q[7] else 5] == 26 for q in qs]
    rows = {}
    for lane, rec in s["records"].items():
        raw = read(root / "scored" / (lane + ".full"), rec["ordinary"])
        gaps = [z[16] - z[2] for z in ROW.iter_unpack(raw)]
        require(len(gaps) == len(labels) == 192, "wrong logit inventory")
        b = offset(gaps[:64], labels[:64])
        result = {"fit_bias": b}
        for i, k in enumerate(("fit", "held", "transfer")):
            gs, ys = gaps[i * 64 : (i + 1) * 64], labels[i * 64 : (i + 1) * 64]
            result[k] = dict(raw=correct(gs, ys, 0.0), fixed=correct(gs, ys, b), oracle_upper=ceiling(gs, ys))
            require(result[k]["fixed"] <= result[k]["oracle_upper"], "bound violated")
        rows[lane] = result
    return put(
        out,
        dict(
            plan=pp,
            scoring=sp,
            post_hoc=True,
            procedure="binary NLL intercept fitted on first 64 fit rows only; split oracle is label-informed, not deployable",
            rows=rows,
        ),
    )


if __name__ == "__main__":
    root, pp, sp, out = sys.argv[1:]
    print(run(Path(root), pp, sp, Path(out)))
