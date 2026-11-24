"""Read only authenticated training evidence; never instantiate a language model."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import runpy

HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE.parent / "radius-transfer/training_anchors.py"))
M = runpy.run_path(str(HERE / "metrics.py"))
S, D, E = A["S"], A["D"], A["E"]
need, sha, enc, f32 = A["need"], A["sha"], A["enc"], A["f32"]


def checked_path(read, sealed, stage, arm, cases, initial, get_state, spec):
    journal = [json.loads(line) for line in read(stage+".jsonl").splitlines()]
    tape = read(stage+".g32")
    stride = 8*spec["shape"][1]
    need(len(journal) == len(cases) and len(tape) == stride*len(cases), "wrong path length")
    before = initial
    for j, (case, row) in enumerate(zip(cases, journal)):
        A["check_join"](case, row, arm, before)
        raw = tape[j*stride:(j+1)*stride]
        need(sha(raw) == row["grad_sha256"], "wrong native training gradient")
        grad = S["vector"](raw, stride//4)
        after = row["step"]["result_id"]
        pre = get_state(before)
        D["check_step"](pre[0], get_state(after)[0], grad, row["step"], spec)
        yield row, pre
        before = after
    need(before == sealed[stage], "path did not terminate at sealed endpoint")


def initial_observations(p, native, host):
    observations = {}
    for row in native:
        if row["host"] != host:
            continue
        r = row["receipt"]
        c = r["contrast"]
        m = c["target_minus_alternative"]
        need(math.isfinite(m) and m == c["target_logit"]-c["alternative_logit"],
             "initial margin disagrees with original logits")
        raw = b"".join(S["unpack"](r["arms"][host][k]) for k in ("gate", "bias"))
        need(len(raw) == 8*p["shape"][1], "wrong factual gradient width")
        observations[row["key"]] = m, M["coords"](raw)
    return observations


def visits(p, native, read, sealed, host, family, order, controls):
    pin = sealed["initial-"+family]
    need(pin == p["geometry"][host]["pin"], "wrong factual initial geometry")
    initial = S["unpack"](p["geometry"][host]["bytes"])
    need(initial == read("states/"+pin[7:]+".json"), "different initial state in the two archives")
    start, layout = A["state"](initial, p, family, pin)
    start = M["coords"](start.tobytes())

    @lru_cache(maxsize=64)
    def get_state(owner):
        need(re.fullmatch(r"sha256:[0-9a-f]{64}", owner) is not None, "invalid geometry pin")
        vec, frozen = A["state"](read("states/"+owner[7:]+".json"), p, family, owner)
        need(frozen == layout, "unexpected change outside the trainable vector")
        return M["coords"](vec.tobytes())

    baselines = initial_observations(p, native, host)
    starts = {rows[0]["key"] for rows in order.values()}
    need(set(p["anchors"][host]) == starts and len(starts) == len(order), "wrong training anchors")
    spec = dict(shape=p["shape"], target_l2=D["DISTANCE"], tolerance=1e-4)
    rows_out = []
    for seed, cases in order.items():
        paths = {arm: list(checked_path(read, sealed, "%s-%s_%s"%(seed, family, arm), arm,
                                        cases, pin, get_state, spec)) for arm in ("full", "rival")}
        for j, case in enumerate(cases):
            key = case["key"]
            fr = paths["full"][j][0]
            cr, before = paths["rival"][j]
            declared = cr["declared"]
            margin0, gradient = baselines[key]
            observed = declared["mean_target_minus_declared_rival"]
            full_ce, rival_ce = f32(fr["loss"]), f32(declared["full_ce"])
            move = M["displacement"](before, start)
            own = M["tangent"](gradient, move)
            others = controls[key]
            lexical = M["mean"]([M["tangent"](baselines[k][1], move) for k in others]) if others else None
            if j == 0:
                need(observed == margin0 == p["anchors"][host][key]["margin"] and not any(move[1])
                     and full_ce == rival_ce and own == 0., "j=0 origins or measurements disagree")
            rows_out.append(dict(host=host, seed=seed, prior=j, key=key,
                                 pair=case["pair"], plural=case["plural"], conflict=case["conflict"],
                                 initial=pin, before_rival=cr["step"]["origin_id"], before_full=fr["step"]["origin_id"],
                                 margin0=margin0, margin=observed, change=observed-margin0, zero=0.,
                                 tangent=own, lexical=lexical, control_keys=others,
                                 full_nll_rival=rival_ce, full_nll_ordinary=full_ce,
                                 rival_minus_full_nll=rival_ce-full_ce))
    return rows_out


def anchor_changes(rows):
    starts = {(r["host"], r["key"]): r for r in rows if r["prior"] == 0}
    out = []
    for r in rows:
        base = starts.get((r["host"], r["key"]))
        if base is None or r["prior"] == 0:
            continue
        full = r["full_nll_rival"]-base["full_nll_rival"]
        choice = D["binary_loss"](r["margin"])-D["binary_loss"](base["margin"])
        out.append(dict(key=r["key"], host=r["host"], seed=r["seed"], prior=r["prior"],
                        initial_seed=base["seed"], full_rival=full, within_pair=choice,
                        coverage=full-choice, full_ordinary=r["full_nll_ordinary"]-base["full_nll_ordinary"]))
    return out


def run(capture, matched, kind):
    p, native, source = S["inputs"](capture, kind)
    need("evaluation" not in p and p["archive_manifest"] == E["ROOTS"][2], "not the training-only capture")
    read, opened = A["training_reader"](matched, kind, p["source_plan"])
    identity = json.loads(read("identity.json"))
    need(identity == dict(base=p["base"], extension=E["P"]["EXTENSION"], backend="rust_f32_cpu_prepacked",
                          head="original_packed", accumulator="f32_wide_backward"), "wrong training runtime")
    sealed = json.loads(read("sealed.json"))
    seeds = (17, 31, 73) if kind == "gpt2" else (11,)
    order = {s: sorted(p["train"], key=lambda c: sha(("relative/%s/%s"%(s, c["key"])).encode())) for s in seeds}
    controls = M["controls"](p["train"])
    rows = [r for host, family in A["HOSTS"]
            for r in visits(p, native, read, sealed, host, family, order, controls)]
    anchors = anchor_changes(rows)
    need(len(rows) == len(seeds)*len(p["train"])*2 and len(anchors) == 2*len(seeds)*(len(seeds)-1),
         "missing visits or unexpected anchor support")
    return dict(schema="spiraltorch.prequential_margin_forecast.v1", kind=kind,
                analysis={name: sha((HERE/name).read_bytes()) for name in ("study.py", "metrics.py", "PROTOCOL.md")},
                identity=identity, capture_manifest=S["ARCHIVE"], capture_inputs=source,
                training_manifest=E["ROOTS"][2], training_inputs=opened,
                rows=rows, summary=M["summarize"](rows, seeds), anchor_changes=anchors,
                new_model_calls=0, new_updates=0, new_heldout_scores=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("rehearsal", "gpt2"))
    parser.add_argument("capture", type=Path)
    parser.add_argument("matched", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = enc(run(args.capture, args.matched, args.kind))
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(sha(raw), args.out.name)
