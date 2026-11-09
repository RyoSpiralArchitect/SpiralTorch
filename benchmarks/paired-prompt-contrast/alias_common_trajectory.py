"""Matched synthetic geometry trajectories. Freeze, measure once on CPU, then audit offline."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alias_interference import CLASSES, STATES, fit_inputs, forbid_nontraining_files, signature
from alias_subset_study import MENU, RATE, STOP, native, open_model
from shared_prefix_study import SITE, obj, pack, put, read, record, require, sha, write

ARMS = ("equal", "common", "initial_ray")
STEPS, RADIUS, MARGIN, SECONDS = 32, .002, 5e-6, 120
CODE = ("alias_common_trajectory", "alias_common_trajectory_audit", "alias_common_step",
        "alias_interference", "alias_interference_audit", "alias_subset_study",
        "projection_language_fixture", "shared_prefix_study")
BUDGET = dict(gradients=3456, terminal_forwards=108, equal_steps=64, common_steps=66, ray_extensions=62)


def code_pins():
    return {name: sha(Path(__file__).with_name(name + ".py").read_bytes()) for name in CODE}


def endpoints(journal, state, arm):
    rows = [r for r in journal if (r.get("source"), r.get("policy"), r.get("radius")) == (state, arm, 1)]
    proposal, = [r for r in rows if r["event"] == "proposal"]
    scores = [r for r in rows if r["event"] == "score"]
    require([r["slot"] for r in scores] == list(range(18)), "incomplete prior full-cohort scores")
    return dict(artifact=proposal["pin"], receipt=sha(pack(proposal["receipt"])),
                scores=sha(pack([[r["cost"], r["absolute"], r["weak"]] for r in scores])))


def freeze(previous, pp, mp, out):
    p = obj(previous / "protocol/plan.json", pp)
    m = obj(previous / "measured/measured.json", mp)
    require(m["plan"] == pp and p["kind"] == "spiraltorch.alias_group_common_direction.v1" and
            p["states"] == list(STATES) and p["classes"] == list(CLASSES) and p["radii"][1] == RADIUS and
            m["counts"] == dict(gradients=36, common_proposals=4, equal_proposals=4, forwards=144),
            "unexpected prior measurement")
    journal = [json.loads(s) for s in read(previous / "measured/observations.jsonl", m["observations"], 2000000).splitlines()]
    first = {s: {a: endpoints(journal, s, a) for a in ("equal", "common")} for s in STATES}
    out.mkdir(mode=0o700)
    return put(out / "plan.json", dict(
        kind="spiraltorch.alias_common_trajectory.v1", code=code_pins(), native=m["native"], prior=[pp, mp],
        replay=first, fixture=p["fixture"], training=p["training"], groups=p["group_slots"], classes=CLASSES,
        base=p["base"], lora=p["lora"], origins=p["origins"], states=STATES, arms=ARMS, steps=STEPS, radius=RADIUS,
        rate=RATE, margin=MARGIN, seconds=SECONDS, budget=BUDGET, retries=0,
        primary="At both starts: all 32 common steps reduce every class mean conditional NLL by >5e-6; "
                "equal-mean endpoint increases idle conditional NLL by >5e-6. No averaging over failures.",
        ray_control="Same first common step, then extrapolate its fixed parameter direction from the origin. "
                    "Record whether any later ray step increases a group NLL by >5e-6 despite common passing; "
                    "separately compare the smallest of the three cumulative gains, threshold 5e-6. "
                    "Neither gate establishes superiority to other adaptive or fixed-weight optimizers.",
        secondary="Absolute accepted-set, baseline-weaker and current-worst alias NLL per group; "
                  "individual regressions. Same 18 training prompts throughout; no evaluation data.",
    ))


def scores(report, accepted):
    lines = report["line_nlls"]
    require(report["accepted_indices"] == accepted and report["stop_id"] == STOP and
            len(lines) == 6 and all(map(math.isfinite, [*lines, report["conditional_nll"], report["accepted_set_nll"]])),
            "invalid full-line scores")
    return dict(lines=lines, cost=report["conditional_nll"], absolute=report["accepted_set_nll"])


def checkpoint(path, model):
    pin = write(path, model.geometry_artifact().to_json().encode())
    require("sha256:" + pin == model.geometry_artifact_id, "artifact does not identify its serialized weights")
    return pin


def train(fixture, training, planned, pp, out):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (SECONDS, SECONDS))
    forbid_nontraining_files(fixture, training)
    pl = obj(planned / "plan.json", pp)
    require(pl["code"] == code_pins(), "code changed since freeze")
    fx, tr, examples = fit_inputs(fixture, pl["fixture"], training, pl["training"])
    groups = [[i for i, e in enumerate(examples) if e["meaning"] == c] for c in range(3)]
    require(groups == pl["groups"], "class membership changed")
    st = native()
    require(signature(st._rs.__file__) == pl["native"], "different native extension")
    base = open_model(st, fixture, fx["files"])
    h = tr["endpoints"]["subset_rank"]
    trained = st.GeometryAdapterArtifact.from_json(read(training / "subset_rank.json", h, 100000).decode(),
                                                   "sha256:" + h, pl["base"])
    starts = dict(initial=base, subset_rank=base.with_geometry(trained))
    lora_bytes = base.lora_to_json()
    counts = dict.fromkeys(BUDGET, 0)
    with (out / "journal.jsonl").open("xb") as journal:
        for state in STATES:
            origin = starts[state]
            require((origin.geometry_artifact_id, origin.base_model_identity, origin.lora_artifact_id) ==
                    (pl["origins"][state], pl["base"], pl["lora"]), "wrong starting model")
            checkpoint(out / (state + ".json"), origin)
            for arm in ARMS:
                model, first = origin, None
                for t in range(STEPS + 1):
                    owner = model.geometry_artifact_id
                    key = dict(state=state, arm=arm, time=t, owner=owner)
                    record(journal, dict(event="point", **key))
                    handles, replay = [], []
                    for i, e in enumerate(examples):
                        c = e["meaning"]
                        args = dict(accepted_indices=[c, c + 3], stop_id=STOP)
                        if t == STEPS:
                            r = scores(model.line_subset_loss(e["prompt"], MENU, **args), args["accepted_indices"])
                            counts["terminal_forwards"] += 1
                        else:
                            g = model.geometry_line_subset_gradients(e["prompt"], MENU, **args)
                            p = g.parameter_gradients()
                            require((g.geometry_artifact_id, g.base_model_identity, g.lora_artifact_id,
                                     g.readout_accumulator, g.target_tokens) ==
                                    (owner, pl["base"], pl["lora"], "f32_wide_backward", 18) and
                                    set(p) == {SITE} and set(p[SITE]) == {"bias", "gate"}, "wrong gradient owner or scope")
                            v = p[SITE]["bias"] + p[SITE]["gate"]
                            require(len(v) == 64 and all(map(math.isfinite, v)), "invalid gradient")
                            r = dict(**scores(g.line_subset_report(), args["accepted_indices"]), v=v, loss=g.loss)
                            handles.append(g)
                            counts["gradients"] += 1
                        record(journal, dict(event="example", slot=i, **key, **r))
                        replay.append([r["cost"], r["absolute"], max(r["lines"][c], r["lines"][c + 3])])
                    control = pl["replay"][state]["equal" if arm == "equal" else "common"]
                    if t == 1:
                        require(sha(pack(replay)) == control["scores"], "first finite scores differ from archive")
                    if t == STEPS:
                        break
                    if arm == "equal":
                        nxt, receipt = model.step_geometry_cohort_sgd(handles, RATE, target_l2=RADIUS)
                        counts["equal_steps"] += 1
                    elif arm == "common" or t == 0:
                        nxt, receipt = model.step_common_geometry_sgd([[handles[i] for i in ids] for ids in groups],
                                                                     RATE, target_l2=RADIUS)
                        counts["common_steps"] += 1
                    else:
                        # The control sees fresh gradients for measurement only. Its direction cannot change.
                        artifact, receipt = origin.geometry_artifact().match_trainable_delta(first, (t + 1) * RADIUS)
                        nxt = origin.with_geometry(artifact)
                        counts["ray_extensions"] += 1
                    after = checkpoint(out / f"{state}-{arm}-{t + 1:02d}.json", nxt)
                    if t == 0:
                        first = nxt.geometry_artifact()
                        require(after == control["artifact"] and sha(pack(receipt)) == control["receipt"],
                                "first step differs from the previous finite control")
                    length = model.geometry_artifact().trainable_delta_l2(nxt.geometry_artifact())
                    require(abs(length / RADIUS - 1.) <= 1e-4 and owner == model.geometry_artifact_id and
                            nxt.base_model_identity == pl["base"] and nxt.lora_to_json() == lora_bytes,
                            "unmatched path length or unfrozen owner")
                    record(journal, dict(event="step", **key, post=after, length=length, receipt=receipt))
                    model = nxt
    require(counts == pl["budget"], "unexpected number of model calls")
    return put(out / "measured.json", dict(plan=pp, native=pl["native"], counts=counts,
                                           journal=signature(out / "journal.jsonl")))


def run_once(fixture, training, planned, pin, out):
    pl = obj(planned / "plan.json", pin)
    require(pl["code"] == code_pins() and pl["seconds"] == SECONDS and pl["retries"] == 0, "invalid frozen trial")
    out.mkdir(mode=0o700)
    put(out / "attempt.json", dict(attempt=1, plan=pin, native=pl["native"]))
    env = dict(os.environ, SPIRALTORCH_BACKEND="cpu", RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", HF_HUB_OFFLINE="1",
               SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"))
    args = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_worker",
            str(fixture), str(training), str(planned), pin, str(out)]
    start = time.monotonic()
    with (out / "process.log").open("xb") as log:
        try:
            result = subprocess.run(args, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=SECONDS)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    put(out / "process.json", dict(exit=status, seconds=time.monotonic() - start, log=signature(out / "process.log")))
    require(status == 0, "measurement stopped, no retry: see process.log")
    return signature(out / "measured.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    for name, fields in {"freeze": "previous prior_plan_pin prior_measurement_pin out",
                         "measure": "fixture training planned plan_pin out",
                         "_worker": "fixture training planned plan_pin out",
                         "audit": "planned plan_pin measured measurement_pin out"}.items():
        args = subs.add_parser(name)
        for field in fields.split():
            args.add_argument(field, type=str if field.endswith("_pin") else lambda s: Path(s).resolve())
    kwargs = vars(parser.parse_args())
    cmd = kwargs.pop("cmd")
    from alias_common_trajectory_audit import audit
    return {"freeze": freeze, "measure": run_once, "_worker": train, "audit": audit}[cmd](*kwargs.values())


if __name__ == "__main__":
    print(main())
