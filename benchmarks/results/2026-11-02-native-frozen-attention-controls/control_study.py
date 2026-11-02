"""Single-attempt native fixed-geometry training. 'audit' never imports SpiralTorch."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_contract import (
    CASES,
    HERE,
    PARENT,
    PLAN,
    audit_fit,
    check_step,
    claim,
    first_match,
    verify_primary,
)
from control_score import score, summarize
from joint_protocol import (
    PLAN as TRAIN,
    digest,
    guard_ml,
    ordering,
    read,
    require,
    rows,
    save,
    sha,
)
from joint_worker import boundary, initial_model, load_base, params, paths, wheels
from common import FORBIDDEN
import owned_process

ENV = {
    "RAYON_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def fit(st, base, a):
    seed, arm = a.case.split("-")
    init = read(a.out / "initial" / "receipt.json")
    folder = a.run / a.case
    path = a.old / "data" / "train.json"
    require(sha(path) == TRAIN["train_sha256"], "wrong training windows")
    data = read(path)
    order = ordering(int(seed), len(data))
    docs = [data[i] for i in order]
    reference = read(PARENT / "results.json")["fits"][a.case]
    if a.action == "first":
        folder.mkdir(mode=0o700)
        model = initial_model(st, base, a.out / "initial", arm, init["owners"])
        session = st.NativeCausalLmTrainingSession(
            model, docs, 64, learning_rate=0.01, mode="lora", epochs=1
        )
        previous = None
        start, stop = 0, 256
        before = params(model)
        require(before == init["parameter_hashes"][arm], "different initial parameters")
    else:
        previous = sha(folder / "first.json")
        require(a.upstream == previous, "wrong first stage")
        prev = read(folder / "first.json")
        b = prev["boundary"]
        require(
            prev["claim_sha256"] == a.pin
            and sha(folder / "first.jsonl") == prev["steps_sha256"]
            and b["update"] == 256,
            "unbound resume",
        )
        session = st.NativeCausalLmTrainingSession.from_json(
            base, docs, (folder / "checkpoint-256.json").read_text(), b["checkpoint_id"]
        )
        start, stop = 256, 512
        before = params(session.model)
        require(
            session.checkpoint_id == b["checkpoint_id"]
            and before == b["parameters"]
            and session.model.lora_artifact_id == b["post_lora_id"],
            "different restored state",
        )
    geom = init["owners"][arm]
    lora = session.model.lora_artifact_id
    require(
        session.model.geometry_artifact_id == geom
        and session.data_id == reference["data_id"]
        and (session.completed_updates, session.total_updates) == (start, 512),
        "wrong geometry, seed or cursor",
    )
    j = rows(a.out / a.case / "first.jsonl")[0]
    log = folder / (a.action + ".jsonl")
    with log.open("x", buffering=1) as stream:
        for u in range(start + 1, stop + 1):
            raw = session.step().to_json()
            stream.write(raw + "\n")
            r = json.loads(raw)
            lora = check_step(r, u, lora, geom)
            if u == 1:
                first_match(r, j)
    b = boundary(session, folder / f"checkpoint-{stop}.json")
    require(
        (b["post_lora_id"], b["post_geometry_id"], b["update"], session.exhausted)
        == (lora, geom, stop, stop == 512),
        "unexpected terminal boundary",
    )
    return folder / (a.action + ".json"), {
        "case": a.case,
        "before": before,
        "previous": previous,
        "order_sha256": digest(order),
        "steps_sha256": sha(log),
        "boundary": b,
    }


def worker(a, expected):
    require(
        read(a.run / "claim.json") == expected
        and sha(a.run / "claim.json") == a.pin
        and all(os.environ.get(k) == v for k, v in ENV.items())
        and not (a.run / "status.json").exists()
        and not (a.run / "first-error.json").exists(),
        "invalid worker claim",
    )
    label = "score" if a.action == "score" else a.case + "-" + a.action
    require(
        read(a.run / (label + ".admit.json"))
        == {"claim_sha256": a.pin, "upstream": a.upstream},
        "unadmitted stage",
    )
    save(
        a.run / (label + ".started.json"),
        {"claim_sha256": a.pin, "upstream": a.upstream},
    )
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg == read(a.out / "initial" / "receipt.json")["package"]
        and pkg["native_sha256"] == sha(st._rs.__file__),
        "different installed study wheel",
    )
    base = load_base(st, a.model)
    if a.action == "score":
        require(
            sha(a.run / "fitted.json") == a.upstream
            and sha(a.data / "windows.json")
            == read(PARENT / "execution.json")["held_out_windows_sha256"],
            "unsealed fits or changed sample",
        )
        result = score(st, base, a, read(a.run / "fitted.json"))
        path = a.run / "scored.json"
    else:
        path, result = fit(st, base, a)
    require(
        wheels.installed_identity(inv) == pkg
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "runtime changed",
    )
    save(path, {**result, "claim_sha256": a.pin, "package": pkg})


def audit(a):
    require(claim(a) == read(a.run / "claim.json"), "changed study sources")
    verify_primary(a)
    fits = read(a.run / "fitted.json")
    require(set(fits) == set(CASES), "incomplete fit roster")
    cp = sha(a.run / "claim.json")

    def started(stage, upstream):
        require(
            read(a.run / (stage + ".admit.json"))
            == read(a.run / (stage + ".started.json"))
            == {"claim_sha256": cp, "upstream": upstream},
            "stage not admitted and started",
        )

    for case, f in fits.items():
        started(case + "-first", None)
        started(case + "-second", f["stages"]["first"])
        require(audit_fit(a, case, f["stages"]) == f, "fit summary differs")
    seen = {p.name for p in a.run.glob("*.started.json")}
    require(len(seen) == 2 * len(CASES) + 1, "unexpected worker count")
    sc = read(a.run / "scored.json")
    started("score", sha(a.run / "fitted.json"))
    require(
        sc["claim_sha256"] == cp
        and sc["package"] == read(a.out / "initial" / "receipt.json")["package"]
        and sc["fitted_sha256"] == sha(a.run / "fitted.json")
        and sc["scalar_sha256"] == sha(a.run / "scalar.json"),
        "unbound scores",
    )
    files = {p.stem: p for p in (a.run / "native-scores").glob("*.json")}
    require(
        {k: sha(p) for k, p in files.items()} == sc["native_scores"],
        "changed raw reports",
    )
    return summarize(
        fits, {k: read(p) for k, p in files.items()}, read(a.run / "scalar.json")
    )


def run(a, expected):
    require(a.python and a.python.is_file(), "study venv interpreter required")
    a.run.parent.mkdir(mode=0o700, exist_ok=True)
    a.run.mkdir(mode=0o700)
    save(a.run / "claim.json", expected)
    pin = sha(a.run / "claim.json")
    start = time.monotonic()
    os.environ.update(ENV)
    current = "preflight"

    def call(action, case=None, upstream=None):
        nonlocal current
        current = case + "-" + action if case else action
        save(
            a.run / (current + ".admit.json"),
            {"claim_sha256": pin, "upstream": upstream},
        )
        cmd = [
            str(a.python),
            "-I",
            "-B",
            str(HERE / "control_study.py"),
            action,
            "--pin",
            pin,
        ]
        for name in ("out", "old", "data", "wheel", "model"):
            cmd.extend(["--" + name, str(getattr(a, name).resolve(strict=True))])
        if case:
            cmd.extend(["--case", case])
        if upstream:
            cmd.extend(["--upstream", upstream])
        timeout = min(
            PLAN["limits"]["per_process_seconds"],
            start + PLAN["limits"]["total_seconds"] - time.monotonic(),
        )
        require(timeout > 0, "study deadline")
        with (
            (a.run / (current + ".stdout")).open("xb") as out,
            (a.run / (current + ".stderr")).open("xb") as err,
        ):
            done = owned_process.run(cmd, timeout=timeout, stdout=out, stderr=err)
        require(done.returncode == 0, "first worker failed: " + current)
        print("complete:", current, flush=True)

    try:
        verify_primary(a)
        fits = {}
        for case in CASES:
            call("first", case)
            first = sha(a.run / case / "first.json")
            call("second", case, first)
            fits[case] = audit_fit(
                a, case, {"first": first, "second": sha(a.run / case / "second.json")}
            )
        save(a.run / "fitted.json", fits)
        call("score", upstream=sha(a.run / "fitted.json"))
        current = "evidence"
        result = audit(a)
        require(
            time.monotonic() - start <= PLAN["limits"]["total_seconds"],
            "total study deadline",
        )
        save(a.run / "results.json", result)
        save(
            a.run / "status.json",
            {
                "status": "complete",
                "attempts": 1,
                "claim_sha256": pin,
                "result_sha256": sha(a.run / "results.json"),
            },
        )
    except BaseException as exc:
        save(
            a.run / "first-error.json",
            {"stage": current, "exception": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument(
        "action", choices=("check", "run", "first", "second", "score", "audit")
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--pin")
    parser.add_argument("--upstream")
    a = parser.parse_args()
    expected = claim(a)
    if a.action == "check":
        verify_primary(a)
        print(
            "Matched primary, initial states, data, engine and prior raw receipts verified; no worker started."
        )
    elif a.action == "audit":
        result = audit(a)
        require(
            not (a.run / "first-error.json").exists()
            and result == read(a.run / "results.json")
            and read(a.run / "status.json")
            == {
                "status": "complete",
                "attempts": 1,
                "claim_sha256": sha(a.run / "claim.json"),
                "result_sha256": sha(a.run / "results.json"),
            },
            "not a sealed success",
        )
        print(
            "Six trajectories, six fixed geometries, 108 original anchors and 108 new scores verified."
        )
    elif a.action == "run":
        run(a, expected)
    else:
        require(a.action == "score" or a.case in CASES, "missing case")
        worker(a, expected)
