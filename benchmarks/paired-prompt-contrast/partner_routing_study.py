"""Train equal-size readout-vs-routing subspaces on one vs two partners per identity."""

import itertools
import math
import os
from pathlib import Path
import resource
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partner_projection as pp
import query_swap_binding as prior
from shared_prefix_study import obj, put, read, record, require, sha, write

STATES = ("initial", "subset_rank")
GRAPHS = ("matching", "cycle")
VIEWS = ("aligned", "broken")
STEPS = 8
LENGTH = 0.08
ROW = struct.Struct("<64f")
SCHEDULE = dict(
    steps=STEPS,
    length=LENGTH,
    coordinates=64,
    wall_seconds_per_phase=240,
    roster="8 new identity roles (not new vocabulary); even / odd edge colours and disjoint opposite partners",
    learning="32 prompts, 3 native dense pullbacks per prompt, 64 paired loss terms per update. Matching repeats even; cycle alternates even/odd. Same 8 identities every step.",
    control="Duplicate each pair for aligned; toggle the queried identity at both ends of the second copy for broken. Same prompt frequencies; no query-to-label signal in the symmetric control.",
    optimizer="Normalized full-minibatch descent, fixed path length; zero gradient is a recorded no-op. Never select or retry based on loss.",
    gate="For each frozen geometry origin separately: on >=3/4 unseen partner edges, all four order-averaged A/Q margins improve by >5e-6 against each of hold, matching-QK, cycle-output and cycle-QK-broken. Also need more correct binary choices than hold. No endpoint selection.",
    scope="Two dependent geometry states of a single synthetic decoder, not two model seeds. CPU only. No GPT-2 checkpoint, old evaluation, speed claim, or attention probability trace.",
)


def code():
    return {
        name: prior.signature(Path(__file__).with_name(name + ".py"))
        for name in ("partner_projection", "partner_routing_study", "partner_routing_audit")
    }


def plan(root, parent, parent_pin):
    ref = obj(parent / "plan.json", parent_pin)
    require(
        ref["kind"] == prior.KIND and ref["settings"] == prior.PLAN and ref["code"] == prior.sources(),
        "wrong prior",
    )
    root.mkdir(mode=0o700)
    for path in ("model", "folds"):
        (root / path).mkdir(mode=0o700)
    for name, pin in [
        ("model.safetensors", ref["files"]["weights"]),
        ("config.json", ref["files"]["config.json"]),
        *((h[7:] + ".json", h[7:]) for h in ref["origins"].values()),
    ]:
        write(root / "model" / name, read(parent / "model" / name, pin, 100000))
    write(root / "wheel-inventory.json", read(parent / "wheel-inventory.json", ref["inventory"], 2 << 20))
    seals = {k: put(root / "folds" / (k + ".json"), v) for k, v in pp.roster().items()}
    return put(
        root / "plan.json",
        {k: ref[k] for k in ("base", "lora", "origins", "inventory", "wheel", "native", "files")}
        | dict(
            kind="spiraltorch.partner_projection_learning.v1",
            parent=parent_pin,
            schedule=SCHEDULE,
            folds=seals,
            ids=pp.IDS,
            code=code(),
            dependencies=prior.sources(),
        ),
    )


def rows(root, p, fold):
    return obj(root / "folds" / (fold + ".json"), p["folds"][fold])


def lanes():
    return itertools.product(STATES, GRAPHS, pp.ROUTES, VIEWS)


def snapshot(st, origin, arch, config, blocks):
    raw = arch.edited(blocks)
    return (origin if raw == arch.raw else prior.rank.rebind(st, config, raw, origin)), raw


def native_row(model, prompt, route):
    weights = {}
    c = None
    for site in pp.SITES:
        r = model.projection_contrast_pullback(
            prompt, pp.task.IDLE, pp.task.ON, layer=0, site=site, basis="dense"
        )
        require(
            (
                r["base_model_identity"],
                r["geometry_artifact_id"],
                r["lora_artifact_id"],
                r["readout_accumulator"],
                r["layer"],
                r["site"],
                r["basis"],
                r["width"],
            )
            == (
                model.base_model_identity,
                model.geometry_artifact_id,
                model.lora_artifact_id,
                "f32_wide_backward",
                0,
                site,
                "dense",
                pp.W,
            ),
            "wrong pullback owner or site",
        )
        require(c is None or c == r["contrast"], "factual scores differ across sites")
        c = r["contrast"]
        weights[site] = struct.unpack("<1024d", r["weight_f64le"])
    slope = pp.pull(route, weights)
    require(all(map(math.isfinite, slope)), "nonfinite native pullback")
    return c["target_minus_alternative"], slope


def check_derivative(st, origin, arch, cfg, prompt, route):
    _, g = native_row(origin, prompt, route)
    norm = math.hypot(*g)
    axis = [v / norm for v in g] if norm else [1.0, *([0.0] * 63)]
    h = 0.01
    values = []
    for sign in (-1, 1):
        blocks = pp.lift(route, [sign * h * x for x in axis])
        physical, _ = snapshot(st, origin, arch, cfg, blocks)
        values.append(
            physical.next_token_contrast(prompt, pp.task.IDLE, pp.task.ON)["target_minus_alternative"]
        )
        for site, mat in blocks.items():
            one, _ = snapshot(st, origin, arch, cfg, {site: mat})
            ordinary = one.next_token_contrast(prompt, pp.task.IDLE, pp.task.ON)
            overlay = origin.projection_perturbed_contrast(
                prompt,
                pp.task.IDLE,
                pp.task.ON,
                layer=0,
                site=site,
                basis="dense",
                bias_f32le=bytes(4 * pp.W),
                weight_f32le=prior.rank.f32s(mat),
            )
            require(
                all(
                    abs(ordinary[k] - overlay[k]) <= 8e-6
                    for k in ("target_logit", "alternative_logit", "target_minus_alternative")
                ),
                "packed physical edit differs from native single-site overlay",
            )
    central = (values[1] - values[0]) / (2 * h)
    require(
        abs(central - norm) <= 1e-4 + 0.015 * norm, "native pullback differs from physical finite difference"
    )
    return dict(norm=norm, central=central, signed_gaps=values, radius=h)


def run_train(root, p, pin, out, st, models, arch, cfg):
    decks = {k: rows(root, p, k) for k in ("even", "odd")}
    probes = {
        s + "/" + r: check_derivative(st, models[s], arch, cfg, decks["even"][0], r)
        for s, r in itertools.product(STATES, pp.ROUTES)
    }
    put(out / "derivatives.json", probes)
    ends = {}
    with (out / "steps.jsonl").open("xb") as journal:
        for state, graph, route, view in lanes():
            name = ".".join((state, graph, route, view))
            theta = list(pp.ZERO)
            for step in range(STEPS):
                fold = "odd" if graph == "cycle" and step % 2 else "even"
                model, before = snapshot(st, models[state], arch, cfg, pp.lift(route, theta))
                measurements = [native_row(model, q, route) for q in decks[fold]]
                loss, alpha = pp.coefficients([g for g, _ in measurements], view == "broken")
                grad = [math.sumprod(alpha, col) for col in zip(*(v for _, v in measurements), strict=True)]
                norm = math.hypot(*grad)
                require(math.isfinite(norm) and math.isfinite(loss), "nonfinite update")
                next_theta = (
                    [x - LENGTH * (d / norm) for x, d in zip(theta, grad, strict=True)] if norm else theta
                )
                record(
                    journal,
                    dict(
                        lane=name,
                        step=step,
                        fold=fold,
                        before=sha(before),
                        loss=loss,
                        norm=norm,
                        distance=math.dist(theta, next_theta),
                        theta=next_theta,
                    ),
                )
                theta = next_theta
            final = arch.edited(pp.lift(route, theta))
            ends[name] = dict(theta=theta, bytes=write(out / (name + ".safetensors"), final))
    return put(
        out / "complete.json",
        dict(
            plan=pin,
            steps=STEPS * len(ends),
            vjp_per_step=96,
            journal=prior.signature(out / "steps.jsonl"),
            derivatives=prior.signature(out / "derivatives.json"),
            ends=ends,
        ),
    )


def score(root, p, pin, out, st, models, arch, cfg, training):
    fit = root.parent / "trained"
    t = obj(fit / "complete.json", training)
    require(
        t["plan"] == pin and t["steps"] == 128 and set(t["ends"]) == {".".join(lane) for lane in lanes()},
        "incomplete training",
    )
    decks = {k: rows(root, p, k) for k in ("even", "odd", "check")}
    scores = {}

    def score_model(name, model, folds):
        chunks = []
        results = {}
        for fold in folds:
            vectors = []
            for q in decks[fold]:
                row = ROW.pack(*model.forward(q, [1, 9], last_only=True).values())
                chunks.append(row)
                vectors.append(ROW.unpack(row))
            results[fold] = pp.summary(vectors)
        scores[name] = dict(
            folds=results, order=list(folds), binary=write(out / (name + ".f32le"), b"".join(chunks))
        )

    for state in STATES:
        score_model(state + ".hold", models[state], decks)
    for state, graph, route, view in lanes():
        name = ".".join((state, graph, route, view))
        end = t["ends"][name]
        blocks = pp.lift(route, end["theta"])
        physical, raw = snapshot(st, models[state], arch, cfg, blocks)
        require(
            read(fit / (name + ".safetensors"), end["bytes"], 100000) == raw,
            "cannot reproduce checkpoint bytes",
        )
        score_model(name, physical, decks)
        if route == "qk" and view == "aligned":
            for site in ("query", "key"):
                model, _ = snapshot(st, models[state], arch, cfg, {site: blocks[site]})
                score_model(name + "." + site, model, ["check"])
    decisions = {}
    for state in STATES:

        def check(lane):
            return scores[state + "." + lane]["folds"]["check"]

        target = check("cycle.qk.aligned")
        vs = {
            k: pp.wins(target, check(k))
            for k in ("hold", "matching.qk.aligned", "cycle.output.aligned", "cycle.qk.broken")
        }
        decisions[state] = dict(
            edges=vs,
            correct=target["correct"],
            correct_hold=check("hold")["correct"],
            passed=min(vs.values()) >= 3 and target["correct"] > check("hold")["correct"],
        )
    return put(out / "complete.json", dict(plan=pin, training=training, scores=scores, decisions=decisions))


def main(argv):
    import time

    os.umask(0o077)
    phase, directory, *args = argv
    root = Path(directory).resolve()
    if phase == "plan":
        parent, pin = args
        return plan(root, Path(parent), pin)
    require(phase in ("train", "score") and len(args) == (1 if phase == "train" else 2), "bad phase")
    pin = args[0]
    p = obj(root / "plan.json", pin)
    require(
        p["schedule"] == SCHEDULE and p["code"] == code() and p["dependencies"] == prior.sources(),
        "source or protocol drift",
    )
    out = root.parent / ("trained" if phase == "train" else "scored")
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(phase=phase, plan=pin, parent=args[1:], attempt=1))
    resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
    if phase == "train":

        def fence(event, args):
            if (
                event == "open"
                and isinstance(args[0], (str, bytes))
                and Path(os.fsdecode(args[0])).name == "check.json"
            ):
                raise PermissionError("unseen partner fold is sealed during training")

        sys.addaudithook(fence)
    start = time.monotonic()
    st, installed, models = prior.stack(root, p, True)
    put(out / "installation.json", installed)
    cfg = read(root / "model/config.json", p["files"]["config.json"]).decode()
    archive = pp.Archive(read(root / "model/model.safetensors", p["files"]["weights"], 100000))
    result = (
        run_train(root, p, pin, out, st, models, archive, cfg)
        if phase == "train"
        else score(root, p, pin, out, st, models, archive, cfg, args[1])
    )
    put(out / "finished.json", dict(complete=result, elapsed=time.monotonic() - start))
    return result


if __name__ == "__main__":
    print(main(sys.argv[1:]))
