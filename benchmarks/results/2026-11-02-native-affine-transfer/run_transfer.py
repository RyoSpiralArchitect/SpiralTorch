"""Single-owner execution, terminal roster sealing and independent file-only audit."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer import (
    BOOKS,
    C1,
    C2,
    NAMES,
    PLAN,
    TERMINAL,
    archived,
    check_fit,
    initial_receipt,
    location,
    paths,
    read,
    require,
    save,
    sha,
    source_pins,
    weighted,
)
import owned_process


def cohort_delta(rows, seeds):
    return math.fsum(rows[str(seed)]["free_minus_affine"] for seed in seeds) / len(
        seeds
    )


def audit(a):
    decl = read(a.out / "declaration.json")
    dp = sha(a.out / "declaration.json")
    manifest = read(a.data / "manifest.json")
    require(
        sha(a.data / "manifest.json") == decl["manifest_sha256"]
        and sha(a.data / "windows.json") == decl["windows_sha256"]
        and decl["source"] == source_pins(),
        "declaration or data changed",
    )
    controls = initial_receipt(a.old)["initial"]
    require(
        decl["initial"] == controls and decl["archived"] == archived(a, controls),
        "archived roster changed",
    )
    pre = read(a.out / "preflight.json")
    require(
        pre["declaration_sha256"] == dp
        and pre["probes"]["affine"] == pre["probes"]["learned4"]
        and len(pre["probes"]["affine"]) == 4,
        "initial forward controls",
    )
    fitted = read(a.out / "fitted.json")
    require(
        fitted["declaration_sha256"] == dp
        and fitted["preflight_sha256"] == sha(a.out / "preflight.json")
        and set(fitted["cases"]) == set(TERMINAL),
        "fit roster",
    )
    for s in C2:
        name = f"{s}-affine"
        result = read(a.out / name / "result.json")
        require(
            result["declaration_sha256"] == dp
            and result["package"] == pre["package"]
            and check_fit(location(a, name), result, controls["affine"])
            == fitted["cases"][name],
            "new fit ring",
        )
    for name, prior in decl["archived"].items():
        require(fitted["cases"][name] == prior, "fitted identity differs from archive")
    scored = read(a.out / "scored.json")
    require(
        scored["declaration_sha256"] == dp
        and scored["package"] == pre["package"]
        and set(scored["files"]) == set(NAMES),
        "score roster",
    )
    scores = {}
    data_ids = {}
    for name in NAMES:
        path = a.out / "scores" / (name + ".json")
        require(sha(path) == scored["files"][name], "altered score file")
        obj = read(path)
        expected = (
            fitted["cases"][name]["geometry_id"]
            if name in TERMINAL
            else {
                "base": None,
                "initial-affine": controls["affine"],
                "initial-free": controls["learned4"],
            }[name]
        )
        require(
            obj["model"] == name
            and obj["geometry_id"] == expected
            and set(obj["scores"]) == set(BOOKS),
            "model/book roster changed",
        )
        scores[name] = {}
        for book, row in obj["scores"].items():
            sections = row["sections"]
            require(
                set(sections) == set(manifest["sections"][book])
                and len(sections) == 6
                and all(x["targets"] == 756 for x in sections.values())
                and row["targets"] == 4536
                and row["nll"]
                == weighted([(x["nll"], x["targets"]) for x in sections.values()]),
                "section list/denominator",
            )
            ids = {k: x["data_id"] for k, x in sections.items()}
            require(
                ids == data_ids.setdefault(book, ids),
                "unequal scoring data across states",
            )
            scores[name][book] = row
    require(
        scores["initial-affine"] == scores["initial-free"],
        "different initial section scores",
    )
    comparison = {}
    for book in BOOKS:
        rows = {}
        base, start = scores["base"][book]["nll"], scores["initial-affine"][book]["nll"]
        for seed in C1 + C2:
            fa, fr = (f"{seed}-{arm}" for arm in ("affine", "free"))
            require(
                fitted["cases"][fa]["data_id"] == fitted["cases"][fr]["data_id"],
                "paired training inputs differ",
            )
            affine, free = scores[fa][book], scores[fr][book]
            rows[str(seed)] = {
                "affine_nll": affine["nll"],
                "free_nll": free["nll"],
                "free_minus_affine": free["nll"] - affine["nll"],
                "affine_minus_base": affine["nll"] - base,
                "free_minus_base": free["nll"] - base,
                "affine_minus_initial": affine["nll"] - start,
                "free_minus_initial": free["nll"] - start,
                "section_free_minus_affine": {
                    k: v["nll"] - affine["sections"][k]["nll"]
                    for k, v in free["sections"].items()
                },
            }
        comparison[book] = {
            "base": base,
            "matched_initial": start,
            "seeds": rows,
            "mean_free_minus_affine_by_cohort": {
                k: cohort_delta(rows, seeds) for k, seeds in PLAN["cohorts"].items()
            },
            "free_wins": sum(r["free_minus_affine"] < 0 for r in rows.values()),
        }
    return {
        "schema": "spiraltorch.native_lm.affine_transfer_result.v1",
        "declaration_sha256": dp,
        "fitted_sha256": sha(a.out / "fitted.json"),
        "scores_sha256": sha(a.out / "scored.json"),
        "source_wheel_sha256": pre["package"]["wheel_sha256"],
        "targets_per_state_and_book": 4536,
        "section_evaluations": 270,
        "new_updates": 768,
        "reused_fits": 9,
        "equal_initial_section_scores": True,
        "primary_all_six_free_better": comparison["austen"]["free_wins"] == 6,
        "cases": fitted["cases"],
        "comparison": comparison,
    }


def run(a):
    clock = time.monotonic()
    os.umask(0o077)
    require(
        sha(HERE / "plan.json") == a.plan_pin
        and sha(a.data / "manifest.json") == a.data_pin,
        "unbound protocol",
    )
    data = read(a.data / "manifest.json")
    require(
        data["plan_sha256"] == a.plan_pin
        and data["windows_sha256"] == sha(a.data / "windows.json")
        and sha(a.old / "data" / "train.json") == PLAN["train_sha256"]
        and sha(a.wheel) == PLAN["development_wheel_sha256"],
        "unexpected sample/wheel",
    )
    controls = initial_receipt(a.old)["initial"]
    old = archived(a, controls)
    a.out.mkdir(mode=0o700, exist_ok=False)
    save(
        a.out / "declaration.json",
        {
            "plan_sha256": a.plan_pin,
            "manifest_sha256": a.data_pin,
            "windows_sha256": data["windows_sha256"],
            "source": source_pins(),
            "archived": old,
            "initial": controls,
            "roster": TERMINAL,
            "budgets": PLAN["limits"],
            "prior_evaluation_opened": False,
        },
    )
    pin = sha(a.out / "declaration.json")
    os.environ.update(
        RAYON_NUM_THREADS="4",
        VECLIB_MAXIMUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    stage = "preflight"

    def call(label, upstream=None):
        cmd = [
            str(a.python),
            "-I",
            "-B",
            str(HERE / "transfer.py"),
            label,
            "--pin",
            pin,
        ]
        for k in ("out", "data", "old", "drift", "model", "wheel"):
            cmd.extend(["--" + k, str(getattr(a, k).resolve())])
        if upstream:
            cmd.extend(["--stage-pin", upstream])
        timeout = min(
            PLAN["limits"]["per_worker_seconds"],
            clock + PLAN["limits"]["total_seconds"] - time.monotonic(),
        )
        require(timeout > 0, "study deadline")
        with (
            (a.out / (label + ".stdout")).open("xb") as out,
            (a.out / (label + ".stderr")).open("xb") as err,
        ):
            job = owned_process.run(cmd, timeout=timeout, stdout=out, stderr=err)
        require(job.returncode == 0, "failed worker: " + label)
        print("complete: " + label, flush=True)

    try:
        call(stage)
        init_pin = sha(a.out / "preflight.json")
        roster = dict(old)
        for s in C2:
            stage = f"{s}-affine"
            call(stage, init_pin)
            roster[stage] = check_fit(
                location(a, stage),
                read(a.out / stage / "result.json"),
                controls["affine"],
            )
        require(set(roster) == set(TERMINAL), "incomplete fit roster")
        save(
            a.out / "fitted.json",
            {"cases": roster, "declaration_sha256": pin, "preflight_sha256": init_pin},
        )
        stage = "score"
        call(stage, sha(a.out / "fitted.json"))
        stage = "audit"
        save(a.out / "results.json", audit(a))
        save(
            a.out / "status.json",
            {"status": "complete", "results_sha256": sha(a.out / "results.json")},
        )
    except BaseException as e:
        save(
            a.out / "first-error.json",
            {"stage": stage, "type": type(e).__name__, "message": str(e)},
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "audit"])
    paths(p)
    p.add_argument("--python", type=Path)
    p.add_argument("--plan-pin")
    p.add_argument("--data-pin")
    a = p.parse_args()
    if a.action == "run":
        run(a)
    else:
        result = audit(a)
        require(
            result == read(a.out / "results.json")
            and sha(a.out / "results.json")
            == read(a.out / "status.json")["results_sha256"],
            "final result mismatch",
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
