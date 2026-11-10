"""Fit-only diagnostic: can a linear readout of the frozen vocabulary logits rank a quartet?"""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_prefix_study import obj, put, require, sha


def fit(rows, labels, relative_ridge):
    """Dual ridge on shift-invariant rows, centered using only the supplied fitting edges."""
    require(
        len(rows) == len(labels) > 0 and relative_ridge > 0 and set(labels) == {-1, 1}, "invalid probe roster"
    )
    n = len(rows)
    centered = []
    for z in rows:
        shift = math.fsum(z) / len(z)
        centered.append([v - shift for v in z])
    mu = [math.fsum(col) / n for col in zip(*centered, strict=True)]
    x = [[v - u for v, u in zip(z, mu, strict=True)] for z in centered]
    gram = [[math.sumprod(a, b) for b in x] for a in x]
    ridge = relative_ridge * math.fsum(gram[i][i] for i in range(n)) / n
    require(ridge > 0 and math.isfinite(ridge), "no finite feature variation")
    for i in range(n):
        gram[i][i] += ridge
    lower = [[0.0] * n for _ in rows]
    for i in range(n):
        for j in range(i + 1):
            v = gram[i][j] - math.sumprod(lower[i][:j], lower[j][:j])
            if i == j:
                require(v > 0, "singular probe factorization")
                lower[i][j] = math.sqrt(v)
            else:
                lower[i][j] = v / lower[j][j]
    avg = math.fsum(labels) / n
    y = [t - avg for t in labels]
    tmp = []
    for i in range(n):
        tmp.append((y[i] - math.sumprod(lower[i][:i], tmp)) / lower[i][i])
    alpha = [0.0] * n
    for i in reversed(range(n)):
        alpha[i] = (tmp[i] - math.fsum(lower[j][i] * alpha[j] for j in range(i + 1, n))) / lower[i][i]
    residual = math.hypot(
        *(math.sumprod(row, alpha) - t for row, t in zip(gram, y, strict=True))
    ) / math.hypot(*y)
    require(residual < 5e-5, "normal equations did not close")
    w = [math.sumprod(alpha, col) for col in zip(*x, strict=True)]
    mean = math.fsum(w) / len(w)
    weights = [v - mean for v in w]
    return dict(
        weights=weights,
        bias=avg - math.sumprod(weights, mu),
        ridge=ridge,
        relative_residual=residual,
        score_error_bound=1e-4 * math.fsum(map(abs, weights)),
    )


def predictions(model, rows):
    return [(math.sumprod(model["weights"], z) + model["bias"], model["score_error_bound"]) for z in rows]


def measure(pred, labels):
    require(len(pred) == len(labels) and len(labels) % 4 == 0, "incomplete quartet")

    def score(robust):
        correct = sum(t * x > 5e-6 + (u if robust else 0.0) for (x, u), t in zip(pred, labels, strict=True))
        ranked = 0
        for j in range(0, len(labels), 4):
            z = pred[j : j + 4]
            ranked += (
                min(
                    x - (u if robust else 0.0)
                    for (x, u), t in zip(z, labels[j : j + 4], strict=True)
                    if t == 1
                )
                > max(
                    x + (u if robust else 0.0)
                    for (x, u), t in zip(z, labels[j : j + 4], strict=True)
                    if t == -1
                )
                + 5e-6
            )
        return dict(correct=correct, ranked=ranked)

    return dict(nominal=score(False), robust=score(True))


def study(base, pp, tp, wp, out):
    import guarded_answer_learning as g

    out.mkdir(mode=0o700)
    root = base / "protocol"
    p = obj(root / "plan.json", pp)
    t = obj(base / "trained/complete.json", tp)
    w = obj(base / "witness/complete.json", wp)
    require(w["training"] == tp and w["plan"] == t["plan"] == pp, "foreign fit witness")
    owners = obj(base / "witness/owners.json", w["owners"], 16 << 20)
    queries = sum(g.lookup(root, p), [])
    ys = [1 if g.a.gold(q) == 16 else -1 for q in queries]
    edges = [tuple(sorted((q[1], q[4]))) for q in queries]
    require(
        len(set(edges)) == 8 and all(len(set(edges[j : j + 8])) == 1 for j in range(0, 64, 8)),
        "wrong fit edges",
    )
    summaries = {}
    models = {}
    for tag, state, right in g.snapshots(root, p, t):
        if tag.rsplit(".", 1)[-1] not in ("hold", "answer", "correct"):
            continue
        with (base / "scored" / (tag + ".full")).open("rb") as f:
            raw = f.read(64 * 256)
        require(
            sha(raw) == owners[g.cx.owner_id(p, state, *g.a.SITE, right)]["logit_sha"],
            "fit rows differ from independent wheel witness",
        )
        rows = list(g.old.ROW.iter_unpack(raw))
        for strength in (1e-4, 1e-10):
            k = f"{tag}/{strength:g}"
            m = fit(rows, ys, strength)
            all_models = [m]
            oof = [None] * 64
            for edge in dict.fromkeys(edges):
                train = [i for i, e in enumerate(edges) if e != edge]
                check = [i for i, e in enumerate(edges) if e == edge]
                mm = fit([rows[i] for i in train], [ys[i] for i in train], strength)
                all_models.append(mm)
                for i, pred in zip(check, predictions(mm, [rows[i] for i in check]), strict=True):
                    oof[i] = pred
            models[k] = all_models
            summaries[k] = dict(
                fit=measure(predictions(m, rows), ys),
                leave_one_fit_edge_out=measure(oof, ys),
                error_bound=m["score_error_bound"],
                max_residual=max(z["relative_residual"] for z in all_models),
            )
    h = put(out / "weights.json", models)
    return put(
        out / "summary.json",
        dict(
            post_hoc=True,
            plan=pp,
            training=tp,
            witness=wp,
            weights=h,
            rows_per_state=64,
            limits="Read only the first 64 fit rows. Fit and 8 leave-one-fit-edge-out probes, fixed ridge levels, +/-1e-4 logit jitter certificate. Logistic decisions of the original LM are unchanged; no decoder calls or held/transfer rows.",
            results=summaries,
        ),
    )


if __name__ == "__main__":
    root, pp, tp, wp, out = sys.argv[1:]
    print(study(Path(root), pp, tp, wp, Path(out)))
