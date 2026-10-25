"""Fixed release matrix and an append-only, two-candidate attempt ledger."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import byte_release_artifacts as artifacts
import byte_release_long_resume as long_resume
import byte_release_schedule as schedule

DECISION = "spiraltorch.byte_release.2026-10-25.v1"
SCHEMA = "spiraltorch.byte_release.admission.v1"
STUDY = "bindings/st-py/tests/test_nn_byte_study.py"
DECODER = "bindings/st-py/tests/test_nn_byte_decoder.py"
LONG = "bindings/st-py/tests/test_nn_byte_release_qualification.py"
TESTS = (
    (STUDY, "StudyGpu", "test_adamw_optimizer_history_survives_study_resume"),
    (STUDY, "StudyGpu", "test_clipped_contextual_mixtures_preserve_all_frozen_clocks"),
    (STUDY, "StudyGpu", "test_clipped_learning_freeze_and_resume_are_one_rust_contract"),
    (STUDY, "StudyGpu", "test_detached_calls_share_a_recipe_not_mutable_models"),
    (STUDY, "StudyGpu", "test_full_run_and_split_resume_are_identical"),
    (STUDY, "StudyGpu", "test_preparation_chain_uses_one_shared_runner"),
    (DECODER, "ByteGpu", "test_clipped_query_learning_resumes_from_one_identical_boundary"),
    (DECODER, "ByteGpu", "test_clipped_sgd_uses_only_active_norm_and_retains_raw_gradients"),
    (DECODER, "ByteGpu", "test_clipped_adamw_moments_use_one_norm_across_active_tensors"),
    (LONG, "LongResumeGpu", "test_128_updates_and_two_saved_cursors"),
)
CAPTURES = (
    {"adamw_input": 1, "adamw_full": 1, "adamw_partial": 1, "adamw_resumed": 1, "adamw_ordinary": 1},
    {"contextual_source": 1, "contextual_input": 1, "contextual_full": 1, "contextual_partial": 1, "contextual_resumed": 1},
    {label: 2 for label in ("learning_legacy_input", "learning_legacy", "learning_disabled_input",
                           "learning_disabled", "learning_input", "learning_full", "learning_partial", "learning_resumed")},
    {"detached_input": 1, "detached_expected": 1, "detached_round": 4},
    {label: 1 for label in ("full_input", "full_ordinary", "full_partial", "full_resumed", "full_complete_noop", "full_repeated")},
    {"prepared_source": 1, "prepared_input": 6, "prepared_partial": 6, "prepared_resumed": 6, "prepared_full": 6},
    {"decoder_input": 2, "query_saved": 2, "query_losses": 2, "query_checkpoint": 2,
     "query_parameters": 2, "query_continued_loss": 4, "query_continued_checkpoint": 4,
     "query_restored": 2, "query_clock": 6, "query_continued_clock": 4},
    {"decoder_input": 1, "sgd_before": 1, "sgd_after": 1, "sgd_clock": 1},
    {"decoder_input": 1, "moments_before": 1, "moments_after": 1, "moments_clock": 1},
    {},
)
CAPTURE_FILES = {
    **dict.fromkeys(("adamw_input", "contextual_source", "contextual_input", "learning_legacy_input",
                    "learning_disabled_input", "learning_input", "detached_input", "full_input",
                    "prepared_source", "prepared_input"), ("request.json",)),
    **dict.fromkeys(("adamw_full", "adamw_partial", "adamw_resumed", "contextual_full", "contextual_partial",
                    "contextual_resumed", "learning_full", "learning_partial", "learning_resumed",
                    "full_partial", "full_resumed", "full_complete_noop", "prepared_partial", "prepared_resumed"),
                   ("report.json", "checkpoint.json")),
    **dict.fromkeys(("adamw_ordinary", "learning_legacy", "learning_disabled", "detached_expected",
                    "full_ordinary", "full_repeated", "prepared_full"), ("report.json",)),
    **dict.fromkeys(("query_losses", "query_continued_loss", "sgd_retained_gradient"), ("expected.bin", "observed.bin")),
    **dict.fromkeys(("query_checkpoint", "query_continued_checkpoint", "query_restored", "query_continued_clock"),
                   ("expected.json", "observed.json")),
    **dict.fromkeys(("query_clock", "sgd_clock", "moments_clock"), ("revision.json",)),
    "detached_round": ("report-0.json", "report-1.json"),
    "decoder_input": ("initial-checkpoint.json", "windows.json"),
    "query_saved": ("checkpoint.json", "controls.json"),
    "query_parameters": ("initial.json", "final.json"),
    "sgd_before": ("gradients.json", "parameters.json", "controls.json"),
    "sgd_after": ("parameters.json",),
    "moments_before": ("gradients.json", "parameters.bin", "controls.json"),
    "moments_after": ("checkpoint.json",),
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path, limit=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("unsafe or oversized contract file")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate contract key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def is_hash(value, length=64):
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def relative_file(value):
    return (type(value) is str and bool(value) and "\\" not in value
            and not value.startswith("/") and all(part not in ("", ".", "..") for part in value.split("/")))


def source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("qualification requires a clean committed source tree")
    commit, tree = git("rev-parse", "HEAD", "HEAD^{tree}").decode().splitlines()
    names = [name for name in git("ls-files", "-z").decode().split("\0") if name]
    selected = [name for name in names if (
        name.endswith(".py") and name.startswith(("tools/", "bindings/st-py/tests/", "bindings/st-py/examples/")))
        or name in ("Cargo.lock", "bindings/st-py/Cargo.toml", "bindings/st-py/pyproject.toml",
                    ".github/workflows/release_wheels.yml", ".github/workflows/byte_release_qualification.yml",
                    "docs/ops/byte-release-qualification.md")]
    for directory in ("tools", "bindings/st-py/tests", "bindings/st-py/examples"):
        if any((Path(root) / directory).rglob("*.pyc")):
            raise ValueError("source tooling must not load stale cached bytecode")
    return {"commit": commit, "tree": tree,
            "files": {name: artifacts.sha256(Path(root) / name) for name in sorted(selected)}}


def validate_admission(value):
    if (type(value) is not dict or value.get("schema") != SCHEMA or value.get("decision") != DECISION
            or type(value.get("candidate")) is not int or value["candidate"] not in (1, 2)
            or value.get("tests") != [list(row) for row in TESTS]
            or value.get("rounds") != 3 or value.get("budget_seconds") != 5400
            or value.get("long_request_sha256") != long_resume.REQUEST_SHA256
            or value.get("release_ready") is not False):
        raise ValueError("admission does not match the frozen release policy")
    source = value.get("source", {})
    if (type(source) is not dict or not is_hash(source.get("commit"), 40) or not is_hash(source.get("tree"), 40)
            or type(source.get("files")) is not dict or "Cargo.lock" not in source["files"]
            or any(not relative_file(k) or not is_hash(v) for k, v in source["files"].items())):
        raise ValueError("admission source identity is incomplete")
    wheels = value.get("wheels")
    if type(wheels) is not dict or set(wheels) != {"linux", "macos", "windows"}:
        raise ValueError("all three final platform wheels must be frozen")
    version = value.get("version")
    if (type(version) is not str or not re.fullmatch(r"\d+\.\d+\.\d+", version)
            or tuple(map(int, version.split("."))) <= (0, 4, 32)):
        raise ValueError("old development artifacts cannot qualify for publication")
    for platform, wheel in wheels.items():
        if (type(wheel) is not dict or wheel.get("platform") != platform or wheel.get("version") != version
                or not is_hash(wheel.get("sha256")) or type(wheel.get("native")) is not dict
                or not relative_file(wheel["native"].get("path")) or not is_hash(wheel["native"].get("sha256"))):
            raise ValueError("candidate wheel identities differ")
    builds = value.get("builds", {})
    if type(builds) is not dict or set(builds) != set(wheels):
        raise ValueError("platform build receipts are required")
    for platform, build in builds.items():
        if (type(build) is not dict or build.get("source_commit") != source["commit"]
                or build.get("wheel_sha256") != wheels[platform]["sha256"]
                or not is_hash(build.get("receipt_sha256")) or type(build.get("toolchain")) is not str
                or not build["toolchain"].strip() or type(build.get("features")) is not list
                or not build["features"] or any(type(item) is not str or not item.strip() for item in build["features"])):
            raise ValueError("build receipt identity is incomplete")
    if value["candidate"] == 1:
        if value.get("prior_admission_sha256") is not None or value.get("repair_review") is not None:
            raise ValueError("initial candidate cannot claim a prior repair")
    elif (not is_hash(value.get("prior_admission_sha256")) or type(value.get("repair_review")) is not str
          or not re.fullmatch(r"https://github.com/RyoSpiralArchitect/SpiralTorch/pull/\d+(?:#.*)?", value["repair_review"])):
        raise ValueError("repair candidate requires prior evidence and a reviewed change")
    return value


def admit(ledger, admission):
    """Append only; an existing candidate slot cannot be replaced or renamed."""
    validate_admission(admission)
    ledger = Path(ledger)
    if ledger.is_symlink():
        raise ValueError("ledger cannot be a symbolic link")
    decision = ledger / "decision.json"
    if read(decision) != {"decision": DECISION, "candidate_limit": 2}:
        raise ValueError("ledger belongs to another decision")
    candidate = admission["candidate"]
    directory = ledger / f"candidate-{candidate}"
    if directory.is_symlink():
        raise ValueError("candidate cannot be a symbolic link")
    if candidate == 2:
        previous = validate_admission(read(ledger / "candidate-1/admission.json"))
        if digest(previous) != admission["prior_admission_sha256"]:
            raise ValueError("repair candidate has the wrong predecessor")
        outcomes = []
        for environment in schedule.ENVIRONMENTS:
            prior = ledger / "candidate-1"
            result_path = prior / (environment + ".result.json")
            if (prior / (environment + ".start.json")).exists() and not result_path.is_file():
                raise ValueError("previous candidate still has an unfinished environment")
            if result_path.is_file():
                row = read(result_path)
                if (row.get("admission_sha256") != digest(previous) or row.get("release_ready") is not False
                        or not is_hash(row.get("evidence_sha256"))):
                    raise ValueError("previous result identity differs")
                outcomes.append(row)
        if not any(row.get("status") in ("failed", "timeout", "cancelled") for row in outcomes):
            raise ValueError("previous candidate is not a retained terminal failure")
        if (admission["source"] == previous["source"] and admission["wheels"] == previous["wheels"]
                and admission["builds"] == previous["builds"]):
            raise ValueError("an unchanged retry is not an implementation repair")
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "admission.json"
    if path.exists():
        if read(path) != admission:
            raise ValueError("candidate slot already contains different artifacts")
    else:
        save(path, admission)
    return directory


def initialize_ledger(ledger):
    """Explicit one-time preparation; execution never creates a new decision."""
    ledger = Path(ledger)
    ledger.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(ledger / "decision.json", {"decision": DECISION, "candidate_limit": 2})


def claim_environment(directory, environment):
    directory = Path(directory)
    admission = validate_admission(read(directory / "admission.json"))
    if admission["candidate"] == 1 and (directory.parent / "candidate-2").exists():
        raise ValueError("candidate 1 has already been superseded")
    if environment not in schedule.ENVIRONMENTS:
        raise ValueError("unknown qualification environment")
    if environment == "hosted-metal":
        local = read(directory / "local-metal.result.json")
        if (local.get("status") != "verified_rounds" or local.get("admission_sha256") != digest(admission)
                or not is_hash(local.get("evidence_sha256"))):
            raise ValueError("hosted execution requires completed matching local evidence")
    save(directory / (environment + ".start.json"), {
        "admission_sha256": digest(admission), "environment": environment,
        "pid": os.getpid(), "started_unix": time.time(), "release_ready": False,
    })
    return directory / environment


def finish_environment(directory, environment, result):
    admission = read(Path(directory) / "admission.json")
    start = read(Path(directory) / (environment + ".start.json"))
    if (environment not in schedule.ENVIRONMENTS or start.get("admission_sha256") != digest(admission)
            or start.get("environment") != environment
            or result.get("schema") != "spiraltorch.byte_release.environment_schedule.v1"
            or result.get("admission_sha256") != digest(admission) or result.get("environment") != environment
            or result.get("release_ready") is not False
            or result.get("status") not in ("verified_rounds", "failed", "timeout", "cancelled")):
        raise ValueError("environment result belongs to another admission")
    if result["status"] == "verified_rounds" and (
            len(result.get("rounds", [])) != 3 or result.get("planned_rounds") != 3
            or result.get("budget_seconds") != 5400
            or any(row.get("round") != index + 1 or row.get("status") != "verified"
                   or row.get("returncode") != 0 or not is_hash(row.get("evidence_sha256"))
                   for index, row in enumerate(result["rounds"]))):
        raise ValueError("environment success lacks three verified rounds")
    save(Path(directory) / (environment + ".result.json"), {
        "admission_sha256": digest(admission), "status": result["status"],
        "evidence_sha256": digest(result), "release_ready": False,
    })
