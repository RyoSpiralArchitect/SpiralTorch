"""Own one local CPU cohort. Never infer a best checkpoint or resume a failed attempt."""

import argparse
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import (  # noqa: E402
    KEYS,
    ORIGINAL,
    PLAN,
    ROSTER,
    read,
    require,
    save,
    sha,
    train_windows,
)
from fit import initial_pin, radii, source_pins, wheels  # noqa: E402
from audit import audit  # noqa: E402
import owned_process  # noqa: E402


def run(a):
    clock = time.monotonic()
    os.umask(0o077)
    require(
        a.plan_pin == sha(HERE / "plan.json")
        and a.data_pin == sha(a.data / "manifest.json"),
        "unbound plan/data",
    )
    data = read(a.data / "manifest.json")
    require(
        data["plan_sha256"] == a.plan_pin
        and data["evaluation_sha256"] == sha(a.data / "evaluation.json")
        and data["train_sha256"] == PLAN["train_sha256"]
        and data["targets_per_book"] == 7056,
        "changed sampled data",
    )
    train_windows(a.train)
    inv = wheels.wheel_archive_inventory(a.wheel)
    require(
        inv["sha256"] == a.wheel_pin
        and inv["native"]["sha256"] == a.native_pin
        and a.wheel_pin != ORIGINAL["wheel_sha256"],
        "wrong development wheel",
    )
    ids = initial_pin(a.initials)
    for arm, label in (("fixed", "fixed4"), ("free", "learned4")):
        require(
            "sha256:" + sha(a.initials / (label + ".json")) == ids[arm],
            "original initial bytes differ",
        )
    initial = radii(read(a.initials / "fixed4.json"))
    require(
        initial == radii(read(a.initials / "learned4.json")), "unequal initial radii"
    )
    decl = {
        "plan_sha256": a.plan_pin,
        "manifest_sha256": a.data_pin,
        "train_sha256": PLAN["train_sha256"],
        "wheel_sha256": a.wheel_pin,
        "native_sha256": a.native_pin,
        "source": source_pins(),
        "roster": KEYS,
        "initial_ids": ids,
        "initial_radii_f32": initial,
        "chapters": data["chapters"],
        "evaluation_sha256": data["evaluation_sha256"],
        "retries": 0,
    }
    a.output.mkdir(mode=0o700, exist_ok=False)
    save(a.output / "declaration.json", decl)
    d = sha(a.output / "declaration.json")
    os.environ.update(
        RAYON_NUM_THREADS=str(PLAN["budgets"]["rayon_threads"]),
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    active = None

    def execute(task, label, extra=()):
        nonlocal active
        active = label
        args = [
            str(a.python),
            "-I",
            "-B",
            str(HERE / "fit.py"),
            task,
            "--declaration-pin",
            d,
        ]
        for k in ("output", "train", "model", "initials", "data", "wheel"):
            args.extend(["--" + k, str(getattr(a, k).resolve())])
        args.extend(extra)
        remaining = min(
            PLAN["budgets"]["per_worker_seconds"],
            clock + PLAN["budgets"]["total_seconds"] - time.monotonic(),
        )
        require(remaining > 0, "cohort deadline exhausted")
        with (
            (a.output / (label + ".stdout")).open("xb") as out,
            (a.output / (label + ".stderr")).open("xb") as err,
        ):
            result = owned_process.run(args, timeout=remaining, stdout=out, stderr=err)
        require(result.returncode == 0, "worker failed: " + label)
        print("completed " + label, flush=True)

    try:
        execute("init", "init")
        prep = sha(a.output / "preflight.json")
        completed = {}
        for seed, arm in ROSTER:
            name = f"{seed}-{arm}"
            execute(
                "fit",
                name,
                ["--seed", str(seed), "--arm", arm, "--preflight-pin", prep],
            )
            row = read(a.output / name / "result.json")
            require(
                row["checkpoint_id"] == row["stops"][-1]["checkpoint_id"]
                and row["stops"][-1]["updates"] == 256,
                "fit has no final observed checkpoint",
            )
            completed[name] = sha(a.output / name / "result.json")
        save(
            a.output / "fitted.json",
            {"order": list(completed), "cases": completed, "preflight_sha256": prep},
        )
        execute("score", "score", ["--fitted-pin", sha(a.output / "fitted.json")])
        active = "audit"
        save(a.output / "results.json", audit(a.output, finalized=False))
        save(
            a.output / "status.json",
            {"status": "complete", "results_sha256": sha(a.output / "results.json")},
        )
    except BaseException as error:
        save(
            a.output / "first-error.json",
            {"stage": active, "type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("python", "output", "train", "model", "initials", "data", "wheel"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("plan-pin", "data-pin", "wheel-pin", "native-pin"):
        p.add_argument("--" + k, required=True)
    run(p.parse_args())
