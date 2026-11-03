"""One bounded, inference-only checkpoint interchange on a pinned public wheel."""

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design import (  # noqa: E402
    ORIGINAL,
    PATHS,
    PLAN,
    PRIOR,
    PROMPTS,
    ROSTER,
    digest,
    masked,
    project,
    read,
    require,
    swap_record,
)
from common import FORBIDDEN, guard_ml, save, sha  # noqa: E402
from worker import load_base  # noqa: E402
import byte_release_artifacts as wheels  # noqa: E402
import owned_process  # noqa: E402


def sources():
    return {p: sha(HERE / p) for p in ("plan.json", "design.py", "study.py")}


def prep(a):
    guard_ml()
    sys.path.append(str(a.tokenizers_site.resolve(strict=True)))
    import tokenizers

    tokenizer_file = a.model / "tokenizer.json"
    require(
        sha(HERE / "plan.json") == a.plan_pin
        and tokenizers.__version__ == ORIGINAL["tokenizers"]
        and sha(tokenizer_file) == ORIGINAL["model"]["tokenizer.json"],
        "preparation pins differ",
    )
    t = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    encoded = {}
    for p in PLAN["prompts"]:
        prefix = t.encode(p["prefix"], add_special_tokens=False).ids
        whole = t.encode(p["prefix"] + p["continuation"], add_special_tokens=False).ids
        labels = masked(prefix, whole)
        require(
            t.decode(prefix) == p["prefix"]
            and t.decode(whole) == p["prefix"] + p["continuation"],
            "authored tokenization does not round-trip",
        )
        encoded[p["id"]] = {"prefix": prefix, "whole": whole, "labels": labels}
    require(len(encoded) == len(PROMPTS) == 8, "invalid authored roster")
    save(
        a.encoded,
        {
            "plan_sha256": a.plan_pin,
            "tokenizer_sha256": sha(tokenizer_file),
            "prompts": encoded,
        },
    )


def checkpoints(original):
    require(
        sha(PRIOR / "results.json") == PLAN["prior_public_results_sha256"],
        "original result pin changed",
    )
    evidence = read(PRIOR / "results.json")["case_evidence"]
    expected = {}
    for s in PLAN["seeds"]:
        for arm in ("affine", "fixed4", "learned4"):
            k = f"{s}-{arm}"
            pin = "sha256:" + sha(original / "run" / k / "checkpoint.json")
            require(pin == evidence[k]["checkpoint_id"], "original checkpoint changed")
            expected[k] = {
                "checkpoint_id": pin,
                "geometry_id": evidence[k]["geometry_id"],
            }
    return expected


def models(st, base, a, expected):
    yield "base", base, [], 0
    for s in PLAN["seeds"]:
        originals = {}
        pins = {}
        for short, arm in (("affine", "affine"), ("F", "fixed4"), ("L", "learned4")):
            k = f"{s}-{arm}"
            pin = pins[short] = expected[k]["checkpoint_id"]
            model = st.NativeCausalLm.from_training_checkpoint(
                base, (a.original / "run" / k / "checkpoint.json").read_text(), pin
            )
            require(
                model.geometry_artifact_id == expected[k]["geometry_id"],
                "source geometry differs",
            )
            originals[short] = model
        yield f"{s}-affine", originals["affine"], [pins["affine"]], 0
        states = {k: read_state(originals[k]) for k in ("F", "L")}
        for pair in ("FF", "FL", "LF", "LL"):
            r, d = pair
            proposed, changed = swap_record(states[r], states[d])
            artifact = st.GeometryAdapterArtifact.create_json(
                json.dumps(proposed, allow_nan=False)
            )
            require(
                digest(json.loads(artifact.state_json())) == digest(proposed),
                "native values/layout changed",
            )
            if r == d:
                require(
                    changed == 0
                    and artifact.artifact_id == originals[r].geometry_artifact_id,
                    "sham reconstruction differs from original",
                )
                model = originals[r]
            else:
                require(
                    changed == len(PATHS)
                    and artifact.artifact_id != originals[r].geometry_artifact_id,
                    "expected four changed radii",
                )
                model = base.with_geometry(artifact)
            yield f"{s}-{pair}", model, [pins[r], pins[d]], changed


def read_state(model):
    return json.loads(model.geometry_artifact().state_json())


def record_digests(out):
    return {
        name: sha(out / name)
        for name in ["source.json", *(pid + ".json" for pid in PROMPTS)]
    }


def measure(st, model, tokens, out, pid):
    score = model.evaluate_documents(
        [tokens["whole"]],
        PLAN["evaluation_context"],
        microbatch_windows=1,
        labels=[tokens["labels"]],
    )
    require(
        score.target_tokens == len(tokens["whole"]) - len(tokens["prefix"])
        and score.window_count == 1,
        "wrong scored continuation",
    )
    g = st.NativeCausalLmGenerator(
        model, tokens["prefix"], json.dumps(PLAN["generation"])
    )
    with (out / f"{pid}.steps.jsonl").open("x", buffering=1) as stream:
        while not g.finished:
            stream.write(g.step().to_json() + "\n")
    require(
        g.token_ids == tokens["prefix"] + g.generated_ids, "generation prefix changed"
    )
    save(
        out / (pid + ".json"),
        {
            "score": json.loads(score.to_json()),
            "generated": g.generated_ids,
            "finish": g.finish_reason,
            "effective_config": json.loads(g.config_json()),
            "generator_checkpoint_id": g.checkpoint_id,
            "steps_sha256": sha(out / f"{pid}.steps.jsonl"),
        },
    )


def worker(a):
    declared = read(a.output / "declaration.json")
    require(
        sha(a.output / "declaration.json") == a.declaration_pin
        and declared["source"] == sources()
        and sha(a.encoded) == declared["encoded_sha256"]
        and checkpoints(a.original) == declared["sources"],
        "frozen source/data mismatch",
    )
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_inventory(a.wheel)
    package = wheels.installed_identity(inv)
    require(
        package["native_sha256"] == ORIGINAL["native_sha256"]
        and sha(st._rs.__file__) == package["native_sha256"]
        and package["wheel_sha256"]
        == declared["wheel_sha256"]
        == ORIGINAL["wheel_sha256"],
        "wrong wheel",
    )
    base = load_base(st, a.model)
    prompts = read(a.encoded)["prompts"]
    completed, digests = [], {}
    for key, model, parents, changed in models(st, base, a, declared["sources"]):
        require(key == ROSTER[len(completed)], "case order changed")
        out = a.output / key
        out.mkdir(mode=0o700)
        save(
            out / "source.json",
            {
                "geometry_id": model.geometry_artifact_id,
                "source_checkpoint_ids": parents,
                "changed_radius_sites": changed,
            },
        )
        for pid in PROMPTS:
            measure(st, model, prompts[pid], out, pid)
        completed.append(key)
        digests[key] = record_digests(out)
        print("measured " + key, flush=True)
    require(
        completed == ROSTER
        and wheels.installed_identity(inv) == package
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "incomplete or foreign runtime",
    )
    save(
        a.output / "worker.json",
        {
            "cases": completed,
            "package": package,
            "foreign_loaded": False,
            "declaration_sha256": a.declaration_pin,
            "record_digests": digests,
        },
    )


def audit(root, *, finalized=True):
    declaration = read(root / "declaration.json")
    require(
        not (root / "first-error.json").exists(),
        "first failure retained; no completed matrix",
    )
    if finalized:
        require(
            read(root / "status.json")
            == {
                "status": "complete",
                "declaration_sha256": sha(root / "declaration.json"),
                "results_sha256": sha(root / "results.json"),
            },
            "no completed matrix",
        )
    require(
        declaration["source"] == sources()
        and declaration["roster"] == ROSTER
        and declaration["plan_sha256"] == sha(HERE / "plan.json")
        and declaration["wheel_sha256"] == ORIGINAL["wheel_sha256"],
        "not this complete matrix",
    )
    worker_receipt = read(root / "worker.json")
    require(
        worker_receipt["cases"] == ROSTER
        and worker_receipt["foreign_loaded"] is False
        and worker_receipt["declaration_sha256"] == sha(root / "declaration.json")
        and worker_receipt["record_digests"]
        == {k: record_digests(root / k) for k in ROSTER}
        and worker_receipt["package"]["wheel_sha256"] == ORIGINAL["wheel_sha256"]
        and worker_receipt["package"]["native_sha256"] == ORIGINAL["native_sha256"],
        "wrong worker identity",
    )
    require(
        sha(PRIOR / "results.json") == PLAN["prior_public_results_sha256"],
        "wrong public sources",
    )
    expected = read(PRIOR / "results.json")["case_evidence"]
    require(
        declaration["sources"]
        == {
            k: {f: row[f] for f in ("checkpoint_id", "geometry_id")}
            for k, row in expected.items()
        },
        "wrong checkpoint roster",
    )
    cases = {}
    for key in ROSTER:
        out = root / key
        case = cases[key] = read(out / "source.json")
        if key == "base":
            require(
                case
                == {
                    "geometry_id": None,
                    "source_checkpoint_ids": [],
                    "changed_radius_sites": 0,
                },
                "wrong base",
            )
        else:
            seed, arm = key.split("-")
            src = (
                [seed + "-affine"]
                if arm == "affine"
                else [seed + "-" + {"F": "fixed4", "L": "learned4"}[x] for x in arm]
            )
            hybrid = arm in ("LF", "FL")
            require(
                case["source_checkpoint_ids"]
                == [expected[k]["checkpoint_id"] for k in src]
                and case["changed_radius_sites"] == (4 if hybrid else 0)
                and (
                    (case["geometry_id"] != expected[src[0]]["geometry_id"])
                    if hybrid
                    else (case["geometry_id"] == expected[src[0]]["geometry_id"])
                ),
                "wrong model parents",
            )
        case["records"] = {}
        for pid in PROMPTS:
            rec = read(out / (pid + ".json"))
            steps = [
                json.loads(line)
                for line in (out / f"{pid}.steps.jsonl").read_bytes().splitlines()
            ]
            ev = rec["score"]
            cfg = rec["effective_config"]
            require(
                ev["base_model_identity"] == ORIGINAL["base_id"]
                and ev["geometry_artifact_id"] == case["geometry_id"]
                and ev["window_count"] == 1
                and ev["skipped_windows"] == 0
                and ev["microbatches"] == 1
                and ev["target_tokens"] == declaration["targets"][pid]
                and ev["ignored_targets"] == declaration["ignored"][pid]
                and cfg["execution"] == PLAN["generation"]["execution"]
                and all(
                    cfg[k] == PLAN["generation"][k]
                    for k in ("max_new_tokens", "seed", "eos_token_ids")
                )
                and cfg["decoding"]["selection"] == "greedy"
                and cfg["decoding"]["distribution"]
                == PLAN["generation"]["decoding"]["distribution"]
                and cfg["decoding"].get("top_k") is None
                and cfg["decoding"].get("top_p", 1.0) in (None, 1.0)
                and [step["step"] for step in steps] == list(range(1, len(steps) + 1))
                and steps
                and [v["token_id"] for v in steps] == rec["generated"]
                and steps[-1]["finish_reason"] == rec["finish"]
                and sha(out / f"{pid}.steps.jsonl") == rec["steps_sha256"],
                "score / decoding mismatch",
            )
            case["records"][pid] = {
                "nll": ev["nll"],
                "targets": ev["target_tokens"],
                "data_id": ev["data_id"],
                "generated": rec["generated"],
                "finish": rec["finish"],
                "steps_sha256": rec["steps_sha256"],
                "raw_sha256": sha(out / (pid + ".json")),
            }
    report = project(cases)
    report.update(
        plan_sha256=declaration["plan_sha256"],
        declaration_sha256=sha(root / "declaration.json"),
        encoded_sha256=declaration["encoded_sha256"],
        wheel_sha256=declaration["wheel_sha256"],
    )
    if finalized:
        require(
            read(root / "results.json") == report,
            "saved projection differs from recorded pairs",
        )
    return report


def run(a):
    require(
        a.plan_pin == sha(HERE / "plan.json")
        and a.encoded_pin == sha(a.encoded)
        and sha(a.wheel) == ORIGINAL["wheel_sha256"],
        "pre-execution identity mismatch",
    )
    inputs = read(a.encoded)
    require(
        inputs["plan_sha256"] == a.plan_pin
        and inputs["tokenizer_sha256"] == ORIGINAL["model"]["tokenizer.json"]
        and set(inputs["prompts"]) == set(PROMPTS),
        "wrong prompt roster",
    )
    for row in inputs["prompts"].values():
        require(
            row["labels"] == masked(row["prefix"], row["whole"]), "wrong target mask"
        )
    declaration = {
        "plan_sha256": a.plan_pin,
        "encoded_sha256": a.encoded_pin,
        "wheel_sha256": ORIGINAL["wheel_sha256"],
        "roster": ROSTER,
        "sources": checkpoints(a.original),
        "source": sources(),
        "targets": {
            k: len(v["whole"]) - len(v["prefix"]) for k, v in inputs["prompts"].items()
        },
        "ignored": {k: len(v["prefix"]) - 1 for k, v in inputs["prompts"].items()},
    }
    a.output.mkdir(mode=0o700, exist_ok=False)
    save(a.output / "declaration.json", declaration)
    pin = sha(a.output / "declaration.json")
    os.environ.update(
        RAYON_NUM_THREADS=str(PLAN["budget"]["rayon_threads"]),
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    args = [
        str(a.python),
        "-I",
        "-B",
        str(HERE / "study.py"),
        "worker",
        "--declaration-pin",
        pin,
    ]
    for arg in ("output", "original", "model", "encoded", "wheel"):
        args.extend(["--" + arg, str(getattr(a, arg).resolve())])
    try:
        with (
            (a.output / "worker.stdout").open("xb") as out,
            (a.output / "worker.stderr").open("xb") as err,
        ):
            proc = owned_process.run(
                args, timeout=PLAN["budget"]["worker_seconds"], stdout=out, stderr=err
            )
        require(proc.returncode == 0, "worker stopped; keep first output, no retry")
        save(a.output / "results.json", audit(a.output, finalized=False))
        save(
            a.output / "status.json",
            {
                "status": "complete",
                "declaration_sha256": pin,
                "results_sha256": sha(a.output / "results.json"),
            },
        )
    except BaseException as error:
        save(
            a.output / "first-error.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    for name, paths, pins in (
        ("prep", ("model", "encoded", "tokenizers-site"), ("plan-pin",)),
        (
            "run",
            ("python", "model", "output", "original", "wheel", "encoded"),
            ("plan-pin", "encoded-pin"),
        ),
        (
            "worker",
            ("model", "output", "original", "wheel", "encoded"),
            ("declaration-pin",),
        ),
        ("audit", ("output",), ()),
    ):
        sub = subs.add_parser(name)
        for k in paths:
            sub.add_argument("--" + k, type=Path, required=True)
        for k in pins:
            sub.add_argument("--" + k, required=True)
    a = p.parse_args()
    if a.mode == "audit":
        print(json.dumps(audit(a.output), indent=2, sort_keys=True, allow_nan=False))
    else:
        {"prep": prep, "run": run, "worker": worker}[a.mode](a)
