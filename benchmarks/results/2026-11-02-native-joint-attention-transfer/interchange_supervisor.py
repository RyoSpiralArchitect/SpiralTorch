"""One numerical attempt per (primary result, prospective plan) under a primary run."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from joint_protocol import HERE, digest, read, require, save, sha
from joint_interchange import INTERCHANGE_SHA256
import owned_process

LEGACY_WORKER_SHA256 = (
    "b68d48d3dc22265635841b4d5f90c669a553800b8fa10531a3aa79b80bb8ad29"
)


def key(primary):
    return digest(
        [
            "spiraltorch.native_lm.attention_interchange_attempt.v2",
            primary,
            INTERCHANGE_SHA256,
        ]
    )


def attempt_path(primary_run, primary):
    return primary_run.resolve(strict=True) / ".interchange-claims" / key(primary)


def identify(a):
    require(
        sha(HERE / "interchange_plan.json") == INTERCHANGE_SHA256
        and sha(a.out / "results.json") == a.result_pin
        and read(a.out / "status.json")
        == {"status": "complete", "results_sha256": a.result_pin},
        "unsealed primary or changed crossover plan",
    )
    a.attempt = attempt_path(a.out, a.result_pin)
    require(
        not (a.attempt.parent.is_symlink() or a.attempt.is_symlink()),
        "claim registry must stay inside its private primary run",
    )
    # Compare directory identities, not spelling: APFS may fold case or Unicode.
    parent = a.receipt.parent.resolve(strict=True)
    require(parent.is_dir() and not a.receipt.is_symlink(), "invalid export path")
    require(
        not any(p.samefile(a.out) for p in (parent, *parent.parents))
        and not (a.receipt.exists() and a.receipt.samefile(a.out)),
        "export must be outside the private primary run",
    )
    a.receipt = parent / a.receipt.name


def admissible_result(attempt, primary):
    require(
        not (attempt / "first-error.json").exists(), "failed attempt cannot publish"
    )
    status = read(attempt / "status.json")
    claim = read(attempt / "claim.json")
    require(
        claim["primary_result_sha256"] == primary
        and claim["plan_sha256"] == INTERCHANGE_SHA256
        and claim["experiment_key"] == key(primary)
        and status["claim_sha256"] == sha(attempt / "claim.json")
        and status["result_sha256"] == sha(attempt / "candidate.json"),
        "invalid completed attempt",
    )
    if status["status"] == "complete":
        from verify_public import raw_score_hashes

        require(
            claim["kind"] == "numerical"
            and status["attempts"] == 1
            and read(attempt / "worker-start.json")["claim_sha256"]
            == status["claim_sha256"],
            "missing worker completion",
        )
        require(
            status["native_score_roster_sha256"]
            == digest(raw_score_hashes(attempt, read(attempt / "candidate.json"))),
            "completed native reports changed",
        )
    else:
        require(
            status["status"] == "adopted"
            and claim["kind"] == "verified_legacy_import"
            and status["new_worker_starts"] == 0
            and not (attempt / "worker-start.json").exists()
            and status["result_sha256"] == sha(HERE / "interchange_results.json")
            and claim["legacy_evidence_sha256"]
            == sha(HERE / "interchange_attempt.json"),
            "invalid imported evidence",
        )
    return attempt / "candidate.json"


def publish(a):
    source = admissible_result(a.attempt, a.result_pin)
    # Write next to the destination, then install without ever overwriting it.
    with tempfile.NamedTemporaryFile(dir=a.receipt.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with source.open("rb") as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        require(
            sha(temp) == read(a.attempt / "status.json")["result_sha256"],
            "export changed",
        )
        os.link(temp, a.receipt)
    finally:
        temp.unlink(missing_ok=True)


def owned(a, claim_pin, seconds):
    cmd = [sys.executable, "-I", "-B", str(HERE / "joint_interchange.py")]
    for k in ("old", "out", "data", "model", "wheel", "receipt"):
        cmd.extend(["--" + k, str(getattr(a, k).resolve())])
    cmd.extend(["--result-pin", a.result_pin, "--worker-claim", claim_pin])
    with (
        (a.attempt / "stdout").open("xb") as out,
        (a.attempt / "stderr").open("xb") as err,
    ):
        job = owned_process.run(cmd, timeout=seconds, stdout=out, stderr=err)
    require(job.returncode == 0, "first crossover worker failed")


def supervise(a, launch=owned):
    from verify_public import check_cross, raw_score_hashes

    limits = read(HERE / "interchange_plan.json")["limits"]
    a.attempt.parent.mkdir(mode=0o700, exist_ok=True)
    a.attempt.mkdir(mode=0o700, exist_ok=False)
    claim_file = a.attempt / "claim.json"
    start = time.monotonic()
    try:
        require(not a.receipt.exists(), "refuse existing export")
        claim = {
            "schema": "spiraltorch.native_lm.attention_interchange_claim.v2",
            "kind": "numerical",
            "experiment_key": key(a.result_pin),
            "primary_result_sha256": a.result_pin,
            "plan_sha256": INTERCHANGE_SHA256,
            "worker_sha256": sha(HERE / "joint_interchange.py"),
            "supervisor_sha256": sha(Path(__file__)),
            "limits": limits,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save(claim_file, claim)
        os.environ.update(
            RAYON_NUM_THREADS=str(limits["rayon_threads"]),
            OMP_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        pin = sha(claim_file)
        launch(a, pin, limits["per_worker_seconds"])
        candidate = a.attempt / "candidate.json"
        require(
            read(a.attempt / "worker-start.json")["claim_sha256"] == pin,
            "unbound candidate",
        )
        staged = read(candidate)
        check_cross(read(a.out / "results.json"), staged)
        score_pin = digest(raw_score_hashes(a.attempt, staged))
        elapsed = time.monotonic() - start
        require(
            elapsed <= limits["per_worker_seconds"],
            "crossover exceeded its sealed budget",
        )
        save(
            a.attempt / "status.json",
            {
                "status": "complete",
                "attempts": 1,
                "claim_sha256": pin,
                "result_sha256": sha(candidate),
                "elapsed_seconds": elapsed,
                "native_score_roster_sha256": score_pin,
            },
        )
    except BaseException as error:
        save(
            a.attempt / "first-error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        raise
    publish(a)


def adopt(a):
    from freeze_binding import bind_inputs, bind_run
    from verify_public import check_cross, extract_attempt

    x = read(a.receipt)
    require(
        sha(a.receipt) == sha(HERE / "interchange_results.json")
        and x["primary_result_sha256"] == a.result_pin,
        "not the retained historical result",
    )
    binding = bind_inputs(a)
    require(binding == read(HERE / "input_binding.json"), "different input binding")
    bind_run(a, binding)
    check_cross(read(a.out / "results.json"), x)
    require(
        extract_attempt(a.receipt, x) == read(HERE / "interchange_attempt.json"),
        "legacy attempt evidence differs",
    )
    a.attempt.parent.mkdir(mode=0o700, exist_ok=True)
    a.attempt.mkdir(mode=0o700, exist_ok=False)
    claim = {
        "schema": "spiraltorch.native_lm.attention_interchange_claim.v2",
        "kind": "verified_legacy_import",
        "experiment_key": key(a.result_pin),
        "primary_result_sha256": a.result_pin,
        "plan_sha256": INTERCHANGE_SHA256,
        "legacy_evidence_sha256": sha(HERE / "interchange_attempt.json"),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    }
    save(a.attempt / "claim.json", claim)
    save(a.attempt / "candidate.json", x)
    save(
        a.attempt / "status.json",
        {
            "status": "adopted",
            "claim_sha256": sha(a.attempt / "claim.json"),
            "result_sha256": sha(a.receipt),
            "new_worker_starts": 0,
        },
    )
    admissible_result(a.attempt, a.result_pin)


def dispatch(a, kernel):
    identify(a)
    if a.worker_claim:
        claim = read(a.attempt / "claim.json")
        require(
            not (a.attempt / "first-error.json").exists()
            and not (a.attempt / "status.json").exists()
            and sha(a.attempt / "claim.json") == a.worker_claim
            and claim["kind"] == "numerical"
            and claim["worker_sha256"] == sha(HERE / "joint_interchange.py")
            and claim["supervisor_sha256"] == sha(Path(__file__))
            and claim["primary_result_sha256"] == a.result_pin
            and claim["plan_sha256"] == INTERCHANGE_SHA256
            and claim["limits"] == read(HERE / "interchange_plan.json")["limits"]
            and claim["experiment_key"] == key(a.result_pin)
            and os.environ["RAYON_NUM_THREADS"] == str(claim["limits"]["rayon_threads"])
            and os.environ["OMP_NUM_THREADS"]
            == os.environ["VECLIB_MAXIMUM_THREADS"]
            == "1",
            "unbound crossover worker",
        )
        save(a.attempt / "worker-start.json", {"claim_sha256": a.worker_claim})
        a.receipt = a.attempt / "candidate.json"
        save(a.receipt, kernel(a))
    elif a.adopt_legacy:
        adopt(a)
    elif a.export_only:
        publish(a)
    else:
        supervise(a)
