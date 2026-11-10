"""Read-only reconstruction of the factorial from ordinary full-vocabulary rows, without a model."""

import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partner_routing_study as study
from shared_prefix_study import obj, put, read, require, sha


def from_rows(prompts, blob):
    values = list(struct.iter_unpack("<64f", blob))
    require(len(values) == len(prompts) == 32, "wrong block length")
    tables = {}
    correct = full = 0
    for prompt, v in zip(prompts, values, strict=True):
        fields = {prompt[1]: prompt[2], prompt[4]: prompt[5]}
        query = prompt[7]
        idle = next(k for k in fields if fields[k] == 26)
        gold, other = (16, 2) if query == idle else (2, 16)
        correct += v[gold] > v[other] + 5e-6
        full += v[gold] > max(v[:gold] + v[gold + 1 :])
        pair = tuple(sorted(fields))
        tables.setdefault(pair, {}).setdefault((idle, query), []).append(v[16] - v[2])
    edges = []
    losses = []
    for (a, b), cells in tables.items():
        require(len(cells) == 4 and all(len(x) == 2 for x in cells.values()), "unbalanced edge")

        def diff(u, v):
            return [x - y for x, y in zip(cells[u], cells[v], strict=True)]

        # Report has the rendering's first entity first, which is not necessarily the smaller ID.
        first = next(q[1] for q in prompts if tuple(sorted((q[1], q[4]))) == (a, b))
        a, b = (a, b) if a == first else (b, a)
        by_type = [diff((a, a), (b, a)), diff((b, b), (a, b)), diff((a, a), (a, b)), diff((b, b), (b, a))]
        for ds in by_type:
            losses.extend(max(0.0, -x) + math.log1p(math.exp(-abs(x))) for x in ds)
        edges.append([sum(ds) / 2 for ds in by_type])
    return dict(
        edges=edges,
        shared=sum(sum(e) for e in edges) / 16,
        correct=correct,
        full_correct=full,
        loss=math.fsum(losses) / len(losses),
    )


def audit(root, pin, tp, sp, dest):
    p = obj(root / "plan.json", pin)
    require(p["code"] == study.code() and p["dependencies"] == study.prior.sources(), "source drift")
    fit, out = root.parent / "trained", root.parent / "scored"
    t, s = obj(fit / "complete.json", tp), obj(out / "complete.json", sp)
    require(t["plan"] == s["plan"] == pin and s["training"] == tp, "provenance mismatch")
    for d, h in ((fit, tp), (out, sp)):
        receipt = json.loads((d / "finished.json").read_bytes())
        require(receipt["complete"] == h and 0 < receipt["elapsed"] <= 240, "incomplete stage")
    arch = study.pp.Archive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
    log = [json.loads(line) for line in read(fit / "steps.jsonl", t["journal"], 1 << 20).splitlines()]
    require(len(log) == t["steps"] == 128 and t["vjp_per_step"] == 96, "wrong exposure budget")
    dirs = study.pp.roster()
    require(
        all(obj(root / "folds" / (k + ".json"), p["folds"][k]) == v for k, v in dirs.items()),
        "changed prompts",
    )
    endpoints = {}
    for ln in study.lanes():
        name = ".".join(ln)
        theta = study.pp.ZERO
        trace = [x for x in log if x["lane"] == name]
        require(len(trace) == study.STEPS, "missing steps")
        for j, r in enumerate(trace):
            require(
                (r["step"], r["fold"]) == (j, "odd" if ln[1] == "cycle" and j % 2 else "even")
                and r["before"] == sha(arch.edited(study.pp.lift(ln[2], theta))),
                "wrong batch or step lineage",
            )
            d = math.dist(theta, r["theta"])
            require(
                d == r["distance"] and abs(d - (study.LENGTH if r["norm"] else 0.0)) < 1e-12,
                "wrong step length",
            )
            theta = r["theta"]
        end = t["ends"][name]
        require(
            end["theta"] == theta
            and arch.edited(study.pp.lift(ln[2], theta))
            == read(fit / (name + ".safetensors"), end["bytes"], 100000),
            "bad endpoint weights",
        )
        endpoints[name] = math.hypot(*theta)
    metrics = {}
    n = 0
    expected = (
        {s + ".hold" for s in study.STATES}
        | set(endpoints)
        | {k + "." + site for k in endpoints if k.endswith(".qk.aligned") for site in ("query", "key")}
    )
    require(set(s["scores"]) == expected, "missing or foreign score lane")
    for name, item in s["scores"].items():
        order = item["order"]
        require(
            order == (["check"] if name.endswith((".query", ".key")) else ["even", "odd", "check"]),
            "wrong score folds",
        )
        raw = read(out / (name + ".f32le"), item["binary"], 1024 * 64)
        require(len(raw) == len(order) * 8192, "invalid scored rows")
        for i, k in enumerate(order):
            m = from_rows(dirs[k], raw[i * 8192 : (i + 1) * 8192])
            x = item["folds"][k]
            require(
                x["correct"] == m["correct"]
                and x["full_correct"] == m["full_correct"]
                and all(
                    abs(a - b) < 1e-10 for a, b in zip(sum(x["edges"], []), sum(m["edges"], []), strict=True)
                )
                and abs(m["shared"] - x["shared"]) < 1e-10
                and abs(x["loss"] - m["loss"]) < 1e-10,
                "score mismatch",
            )
            metrics[name, k] = m
            n += 32
    effects = {}
    for state in study.STATES:

        def f(k):
            return metrics[state + "." + k, "check"]

        qk = f("cycle.qk.aligned")
        wins = {
            k: sum(
                min(a - b for a, b in zip(x, y, strict=True)) > 5e-6
                for x, y in zip(qk["edges"], f(k)["edges"], strict=True)
            )
            for k in ("hold", "matching.qk.aligned", "cycle.output.aligned", "cycle.qk.broken")
        }
        gate = min(wins.values()) >= 3 and qk["correct"] > f("hold")["correct"]
        require(
            s["decisions"][state]
            == dict(edges=wins, correct=qk["correct"], correct_hold=f("hold")["correct"], passed=gate),
            "decision mismatch",
        )

        def delta(g):
            return f(g + ".qk.aligned")["shared"] - f(g + ".output.aligned")["shared"]

        effects[state] = dict(
            shared_qk_minus_output_matching=delta("matching"),
            shared_qk_minus_output_cycle=delta("cycle"),
            interaction=delta("cycle") - delta("matching"),
            gate=gate,
        )
    return put(
        dest,
        dict(
            plan=pin,
            training=tp,
            scoring=sp,
            rows_reconstructed=n,
            endpoints=len(endpoints),
            optimizer_steps=len(log),
            nonzero_steps=sum(x["norm"] > 0 for x in log),
            geometry_origins=effects,
        ),
    )


if __name__ == "__main__":
    root, pp, tp, sp, out = sys.argv[1:]
    print(audit(Path(root), pp, tp, sp, Path(out)))
