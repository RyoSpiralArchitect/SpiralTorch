"""Model-free: separate answer-set mass from answer choice and inspect local descent conflicts."""

import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_prefix_study import obj, put, read, require


def loss_parts(logits, gold, support):
    """Both terms are in nats. No full-vocab probability is exponentiated before taking a log."""
    require(
        len(set(support)) == len(support)
        and gold in support
        and all(0 <= j < len(logits) for j in support)
        and all(map(math.isfinite, logits)),
        "invalid support or logits",
    )

    def lse(v):
        mx = max(v)
        return mx + math.log(math.fsum(math.exp(x - mx) for x in v))

    whole, menu = lse(logits), lse([logits[j] for j in support])
    return whole - menu, menu - logits[gold]


def slopes(steps):
    values = []
    cosines = []
    for step in steps:
        r = step["report"]
        a, i, c = (r[k] for k in ("answer", "interaction", "combined"))
        require(0.0 < r["mix"] < 1.0, "expected a mixed step")
        values.append(r["mix"] * i["norm"] / ((1 - r["mix"]) * a["norm"]) if a["norm"] else None)
        if a["norm"] and c["norm"]:
            cosines.append(math.sumprod(a["gradient"], c["gradient"]) / (a["norm"] * c["norm"]))
    ratios = [v for v in values if v is not None]
    return dict(
        weighted_ratio_min=min(ratios, default=None),
        weighted_ratio_max=max(ratios, default=None),
        weighted_ratio_last=values[-1],
        ce_uphill_steps=sum(x < 0.0 for x in cosines),
        undefined_ce_angles=len(steps) - len(cosines),
        worst_ce_cosine=min(cosines, default=None),
    )


def run(base, pp, tp, sp, summary_path, summary_sha, out):
    p = obj(base / "protocol/plan.json", pp)
    t = obj(base / "trained/complete.json", tp)
    s = obj(base / "scored/complete.json", sp)
    audit = obj(summary_path, summary_sha, 2 << 20)
    require(
        p["kind"] == "spiraltorch.full_answer_quartets.v1"
        and t["plan"] == s["plan"] == audit["plan"] == pp
        and s["train"] == audit["training"] == tp
        and audit["serving"] == sp
        and set(audit["scores"]) == set(s["records"]),
        "wrong run",
    )
    folds = ("train0", "train1", "held0", "held1", "transfer0", "transfer1")
    questions = [q for f in folds for q in obj(base / "protocol" / (f + ".json"), p["folds"][f])]
    decomposed = {}
    directions = {}
    for lane, rec in s["records"].items():
        raw = read(base / "scored" / (lane + ".full"), rec["full"])
        require(len(raw) == 192 * 256, "wrong number of rows")
        parts = [
            loss_parts(row, 16 if q[2 if q[1] == q[7] else 5] == 26 else 2, (2, 16))
            for row, q in zip(struct.iter_unpack("<64f", raw), questions, strict=True)
        ]
        decomposed[lane] = {}
        for j, label in enumerate(("fit", "held", "transfer")):
            sample = parts[64 * j : 64 * (j + 1)]
            mass, conditional = [math.fsum(v) / 64 for v in zip(*sample, strict=True)]
            recorded = math.fsum(audit["scores"][lane][f]["answer_nll"] for f in folds[2 * j : 2 * j + 2]) / 2
            require(abs(mass + conditional - recorded) < 1e-12, "loss components do not reconstruct gold NLL")
            decomposed[lane][label] = dict(
                support_nll=mass,
                conditional_nll=conditional,
                mean_support_mass=math.fsum(math.exp(-m) for m, _ in sample) / 64,
            )
        if lane.endswith((".half", ".fit_scale", ".fit_control", ".half_control")):
            packet = obj(base / "trained" / (lane + ".json"), t["ends"][lane], 2 << 20)
            directions[lane] = slopes(packet["steps"])
    return put(
        out,
        dict(
            post_hoc=True,
            plan=pp,
            training=tp,
            serving=sp,
            summary=summary_sha,
            components=decomposed,
            local_directions=directions,
        ),
    )


if __name__ == "__main__":
    root, pp, tp, sp, path, summary, out = sys.argv[1:]
    print(run(Path(root), pp, tp, sp, Path(path), summary, Path(out)))
