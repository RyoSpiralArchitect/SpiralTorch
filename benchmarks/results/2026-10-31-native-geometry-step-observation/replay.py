"""One bounded developer-wheel replay of the previously frozen native fits."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "2026-10-31-native-book-ablation"
sys.path[:0] = [str(PARENT), str(HERE.parents[2] / "tools")]
import byte_release_artifacts as wheels  # noqa: E402
import owned_process  # noqa: E402
from common import (  # noqa: E402
    FORBIDDEN,
    PLAN as ORIGINAL,
    cases,
    guard_ml,
    ordering,
    read,
    require,
    save,
    sha,
    weighted,
)
from worker import artifact, load_base  # noqa: E402
from verify import audit as audit_original  # noqa: E402

PLAN = read(HERE / "plan.json")
PATHS = [f"transformer.h.{layer}.mlp" for layer in ORIGINAL["layers"]]


def score_development(model, docs):
    chapters = {}
    require(set(docs) == {str(i) for i in range(50, 56)}, "development roster differs")
    for key, windows in sorted(docs.items()):
        score = model.evaluate_documents(windows, 64, microbatch_windows=2)
        require(
            score.target_tokens == 504
            and not score.skipped_windows
            and not score.ignored_targets
            and math.isfinite(score.nll),
            "missing or nonfinite development measurement",
        )
        chapters[key] = {
            "nll": score.nll,
            "targets": score.target_tokens,
            "data_id": score.data_id,
        }
    return {
        "nll": weighted([(x["nll"], x["targets"]) for x in chapters.values()]),
        "targets": 3024,
        "chapters": chapters,
    }


def check_observation(observed, ordinary, data_id, arm):
    require(
        observed["schema"] == "spiraltorch.native_lm.observed_training_step.v1"
        and observed["base_model_identity"] == ORIGINAL["base_id"]
        and observed["data_id"] == data_id
        and observed["step"] == ordinary
        and set(observed["sites"]) == set(PATHS),
        "unbound step observation",
    )
    for site in observed["sites"].values():
        require(
            site["elements"] == 2 * 63 * 768
            and all(
                math.isfinite(site[k]) and site[k] >= 0
                for k in (
                    "input_rms",
                    "residual_rms",
                    "relative_residual_rms",
                    "trainable_gradient_l2",
                )
            )
            and site["input_rms"] > 0
            and math.isclose(
                site["relative_residual_rms"],
                site["residual_rms"] / site["input_rms"],
                rel_tol=1e-12,
            ),
            "invalid activation/gradient summary",
        )
        radius, gradient = site["pre_log_radius"], site["log_radius_gradient"]
        require(
            (radius is None if arm == "affine" else math.isfinite(radius))
            and (math.isfinite(gradient) if arm == "learned4" else gradient is None),
            "invalid radius semantics",
        )


def worker(args, root):
    case = f"{args.seed}-{args.arm}"
    previous = args.original / "run" / case
    directory = root / case
    directory.mkdir(mode=0o700)
    declaration = read(root / "declaration.json")
    require(
        sha(root / "declaration.json") == args.declaration_pin
        and sha(args.wheel) == declaration["wheel_sha256"]
        and all(sha(HERE / name) == pin for name, pin in declaration["source"].items()),
        "execution pins differ",
    )
    attempts = guard_ml()
    import spiraltorch as st

    # Inspect the arm64 development wheel; it is deliberately not a universal2 release.
    inventory = wheels.wheel_archive_inventory(args.wheel)
    package = wheels.installed_identity(inventory)
    require(
        package["native_sha256"] != ORIGINAL["native_sha256"]
        and sha(Path(st._rs.__file__)) == package["native_sha256"]
        and package["version"] == "0.4.33",
        "expected development native library, not the published binary",
    )
    published = read(PARENT / "results.json")["case_evidence"][case]
    require(
        sha(previous / "steps.jsonl") == published["steps_sha256"]
        and "sha256:" + sha(previous / "checkpoint.json") == published["checkpoint_id"],
        "original case bytes changed",
    )
    reference = (previous / "steps.jsonl").read_bytes().splitlines()
    require(len(reference) == PLAN["updates_per_case"], "original step roster differs")
    data = args.original / "data/train.json"
    require(
        sha(data) == declaration["train_sha256"]
        and sha(root / "development.json") == declaration["development_sha256"],
        "probe/training inputs changed",
    )
    train, dev = read(data), read(root / "development.json")
    base = load_base(st, args.model)
    preflight = args.original / "before/preflight"
    state = artifact(
        st, base, preflight / (args.arm + ".json"), declaration["initial_ids"][args.arm]
    )
    optimizer = st.NativeGeometryAdamW(
        state, max_gradient_norm=ORIGINAL["max_gradient_norm"]
    )
    trainer = st.NativeCausalLmTrainer(
        base.with_geometry(state),
        optimizer,
        [train[i] for i in ordering(args.seed, len(train))],
        **ORIGINAL["recipe"],
    )
    require(
        trainer.data_id == published["data_id"]
        and trainer.total_updates == len(reference),
        "training owner differs",
    )
    scores = {}
    saved_scores = args.original / "run/score"
    for name in ("initial", case):
        require(
            sha(saved_scores / (name + ".json"))
            == declaration["score_file_sha256"][name],
            "original score bytes changed",
        )

    def evaluate(u):
        scores[str(u)] = score_development(trainer.model, dev)
        if u in (0, len(reference)):
            original = read(saved_scores / (("initial" if u == 0 else case) + ".json"))[
                "development"
            ]
            require(
                scores[str(u)] == original,
                "original development scores do not reproduce exactly",
            )
        save(directory / f"score-{u:03d}.json", scores[str(u)])

    evaluate(0)
    with (directory / "observations.jsonl").open("x", buffering=1) as log:
        for u, expected in enumerate(reference, 1):
            result = trainer.step_observed()
            actual = result.step.to_json().encode()
            observation = json.loads(result.to_json())
            log.write(
                json.dumps(observation, allow_nan=False, separators=(",", ":")) + "\n"
            )
            if actual != expected:
                save(
                    directory / "first-step-mismatch.json",
                    {
                        "update": u,
                        "expected": json.loads(expected),
                        "actual": json.loads(actual),
                    },
                )
                raise ValueError("step differs from the original; stop without retry")
            check_observation(
                observation, json.loads(expected), trainer.data_id, args.arm
            )
            if u in PLAN["evaluation_updates"]:
                evaluate(u)
    require(
        trainer.exhausted
        and trainer.to_json().encode() == (previous / "checkpoint.json").read_bytes()
        and trainer.checkpoint_id == published["checkpoint_id"]
        and trainer.model.geometry_artifact_id == published["geometry_id"],
        "original final state differs",
    )
    require(
        wheels.installed_identity(inventory) == package
        and not (FORBIDDEN & {k.partition(".")[0] for k in sys.modules}),
        "unexpected installed package or foreign import",
    )
    save(
        directory / "result.json",
        {
            "seed": args.seed,
            "arm": args.arm,
            "matched_steps": len(reference),
            "data_id": trainer.data_id,
            "matched_checkpoint_id": trainer.checkpoint_id,
            "scores": scores,
            "observations_sha256": sha(directory / "observations.jsonl"),
            "package": package,
            "blocked_optional_imports": attempts,
            "foreign_loaded": False,
        },
    )


def main(args):
    os.umask(0o077)
    root = args.output.resolve()
    if args.worker:
        worker(args, root)
        return
    require(
        sha(HERE / "plan.json") == args.plan_pin
        and sha(args.wheel) == args.wheel_pin
        and args.wheel_pin != ORIGINAL["wheel_sha256"],
        "missing separate protocol/development wheel pins",
    )
    original = audit_original(args.original)
    require(
        original == read(PARENT / "audit.json")
        and original["private_results_sha256"] == PLAN["prior_private_results_sha256"]
        and original["public_results_sha256"] == PLAN["prior_public_results_sha256"],
        "unverified prior study",
    )
    root.mkdir(mode=0o700, exist_ok=False)
    data = args.original / "data"
    save(root / "development.json", read(data / "evaluation.json")["development"])
    keys = [f"{s}-{a}" for s, a in cases()]
    declaration = {
        "plan_sha256": args.plan_pin,
        "wheel_sha256": args.wheel_pin,
        "prior_audit": original,
        "initial_ids": read(args.original / "before/preflight/result.json")["initial"],
        "train_sha256": sha(data / "train.json"),
        "development_sha256": sha(root / "development.json"),
        "source": {p: sha(HERE / p) for p in ("plan.json", "replay.py", "analysis.py")},
        "score_file_sha256": {
            k: sha(args.original / "run/score" / (k + ".json"))
            for k in ["initial", *keys]
        },
        "cases": keys,
        "retries": 0,
    }
    save(root / "declaration.json", declaration)
    pin = sha(root / "declaration.json")
    os.environ.update(
        RAYON_NUM_THREADS="4",
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed, active = [], None
    deadline = time.monotonic() + PLAN["resources"]["total_seconds"]
    try:
        for seed, arm in cases():
            active = f"{seed}-{arm}"
            cmd = [
                str(args.python),
                "-I",
                "-B",
                str(HERE / "replay.py"),
                "--worker",
                "--seed",
                str(seed),
                "--arm",
                arm,
                "--output",
                str(root),
                "--original",
                str(args.original),
                "--model",
                str(args.model),
                "--wheel",
                str(args.wheel),
                "--declaration-pin",
                pin,
            ]
            remaining = min(
                deadline - time.monotonic(), PLAN["resources"]["case_seconds"]
            )
            with (
                (root / (active + ".stdout")).open("xb") as out,
                (root / (active + ".stderr")).open("xb") as err,
            ):
                result = owned_process.run(
                    cmd, timeout=remaining, stdout=out, stderr=err
                )
            require(
                result.returncode == 0
                and read(root / active / "result.json")["matched_steps"]
                == PLAN["updates_per_case"],
                "incomplete case: " + active,
            )
            completed.append(active)
            print("matched " + active, flush=True)
        save(
            root / "status.json",
            {"status": "complete", "cases": completed, "declaration_sha256": pin},
        )
    except BaseException as error:
        save(
            root / "status.json",
            {
                "status": "failed",
                "first_incomplete": active,
                "completed": completed,
                "not_completed": [k for k in keys if k not in completed],
                "error": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "wheel", "output", "original", "python"):
        p.add_argument("--" + name, type=Path, required=name != "python")
    p.add_argument("--plan-pin")
    p.add_argument("--wheel-pin")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--arm", choices=PLAN["arms"])
    p.add_argument("--seed", type=int, choices=PLAN["seeds"])
    p.add_argument("--declaration-pin")
    main(p.parse_args())
