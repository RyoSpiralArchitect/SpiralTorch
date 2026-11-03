"""One instrumented Metal trio. Default: CPU preflight only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "tools")]
from byte_release_artifacts import installed_identity, sha256, wheel_archive_inventory
from compare_embedding_bits import SELECTION, checked, compare, compare_reports, decode_json, report_bytes, require
from inspect_retained import PINS
from owned_process import run as owned_run
import hosted_embedding_environment as hosted

WHEEL = "d37b225ca991e6782d180f044dfbb44de9fc82fb45d9b0fbf6d6b077c7540f68"
NATIVE = "022f6e4e057dba9b74d62406b6784aaf8dac8b12d131675ec8824d1cc413b56c"
ARCHIVE = PINS["synthetic-byte-study-detached-failure"]
LIMIT_SECONDS = 600
ROLES = ("serial", "worker0", "worker1")


def raw_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def record(path, value):
    raw_new(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def preflight(wheel, archive):
    require(sys.flags.isolated and sys.dont_write_bytecode and sys.prefix != sys.base_prefix,
            "use the installed wheel's isolated venv interpreter with -I -B")
    require(sys.platform == "darwin" and platform.machine() == "arm64", "expected local Apple Silicon")
    require(all(importlib.util.find_spec(name) is None for name in ("torch", "numpy")),
            "expected the dependency-free validation environment")
    require(sha256(wheel) == WHEEL, "different wheel")
    inventory = wheel_archive_inventory(wheel)
    require(inventory["native"]["sha256"] == NATIVE, "different wheel native library")
    installed = installed_identity(inventory)
    if archive is None:
        from test_byte_corpus_study import fixture
        raw = fixture()[0]
        provenance = {"request_source": "public_fixture_pinned_bytes", "archive_transferred": False}
    else:
        require(sha256(archive) == ARCHIVE, "different input archive")
        with zipfile.ZipFile(archive) as z:
            raw = z.read("round-0/request.json")
        provenance = {"archive_sha256": ARCHIVE}
    require(hashlib.sha256(raw).hexdigest() == SELECTION["request_sha256"], "different request")
    import spiraltorch as st
    native = Path(importlib.import_module("spiraltorch.spiraltorch").__file__).resolve()
    require(native.is_relative_to(Path(sys.prefix).resolve()) and sha256(native) == NATIVE,
            "loaded native module differs from pinned venv")
    study = st.nn.ByteCorpusStudy.from_json(raw)
    require(decode_json(study.embedding_update_selection_json("geometry", 1, 2, 0)) == SELECTION,
            "different input coordinate")
    return study, {"schema": "spiraltorch.embedding_trio.preflight.v1", "selection": SELECTION,
                   **provenance, "installation": installed,
                   "python": platform.python_version(), "macos": platform.mac_ver()[0],
                   "architecture": platform.machine(), "gpu_calls": 0, "release_ready": False}


def exercise(study, root, *, expected_adapter=None):
    for role in ROLES:
        (root / role).mkdir(mode=0o700)

    def invoke(role, ready=None):
        path = root / role
        if ready is not None:
            ready.wait(timeout=15)
        record(path / "entered.json", {"role": role})
        start = time.monotonic_ns()
        try:
            obs = study.run_wgpu_with_embedding_probe("geometry", 1, 2, 0)
            end = time.monotonic_ns()
            raw = obs.run_outcome_json.encode("utf-8")
            raw_new(path / "run-outcome.json", raw)
            record(path / "host-call.json", {"start_ns": start, "end_ns": end})
            report = decode_json(report_bytes(raw))
            require(report["adapter"].endswith("backend: Metal }") and "device_type: Cpu," not in report["adapter"],
                    "not a reported Metal GPU")
            if expected_adapter is not None:
                require(report["adapter"] == expected_adapter, "different historical Metal adapter")
            return obs, raw, (start, end)
        except Exception as error:
            record(path / "call-error.json", {"type": type(error).__name__, "message": str(error)[:2048]})
            raise

    def read(role, result):
        obs, raw, _ = result
        outcome = decode_json(raw)
        require(outcome["capture_status"] == "views_retained", "embedding views were not retained")
        bits = obs.read_bits_json().encode("utf-8")
        raw_new(root / role / "bits.json", bits)
        pair = outcome, decode_json(bits)
        checked(*pair)
        return pair

    serial = invoke("serial")
    reference = read("serial", serial)
    serial_raw = serial[1]
    del serial
    ready = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(invoke, role, ready) for role in ROLES[1:]]
    # Both host calls are terminal before either worker's views are read.
    workers = [job.result() for job in jobs]
    reports = compare_reports(serial_raw, workers[0][1], workers[1][1])
    record(root / "full-reports.json", reports)
    lo, hi = (worker[2] for worker in workers)
    overlap = min(lo[1], hi[1]) - max(lo[0], hi[0])
    require(overlap > 0, "worker host calls did not overlap; no new pair may be launched")
    pairs = [read(role, result) for role, result in zip(ROLES[1:], workers, strict=True)]
    coord = compare(reference, *pairs)
    different = any(not row["raw_equal"] for row in reports["workers"]) or any(
        row["differences"] for row in coord["workers"])
    return {"schema": "spiraltorch.embedding_trio.result.v1",
            "status": "difference" if different else "no_difference_in_this_instrumented_trio",
            "native_calls": 3, "worker_host_overlap_ns": overlap, "full_reports": reports,
            "coordinate": coord, "release_ready": False}


def worker(root, wheel, archive=None):
    record(root / "worker-started.json", {"max_native_calls": 3, "max_worker_pairs": 1})
    try:
        if archive is None:
            observed = hosted.inspect()
            bad = hosted.mismatches(observed)
            record(root / "host-admission.json", {"observed": observed, "mismatches": bad})
            require(not bad, "hosted admission failed: " + ", ".join(bad))
        study, receipt = preflight(wheel, archive)
        record(root / "preflight.json", receipt)
        result = exercise(study, root, expected_adapter=hosted.ADAPTER if archive is None else None)
    except Exception as error:
        record(root / "error.json", {"status": "error", "type": type(error).__name__, "message": str(error)[:2048]})
        return 1
    record(root / "result.json", result)
    return 2 if result["status"] == "difference" else 0


def execute_once(root, wheel, archive):
    root.mkdir(mode=0o700, exist_ok=False)
    record(root / "protocol.json", {"roles": ROLES, "max_seconds": LIMIT_SECONDS, "retries": 0,
                                   "wheel_sha256": WHEEL, "native_sha256": NATIVE,
                                   **({"archive_sha256": ARCHIVE} if archive is not None else
                                      {"request_source": "public_fixture_pinned_bytes", "archive_transferred": False}),
                                   "selection": SELECTION, "harness_sha256": sha256(__file__),
                                   "comparator_sha256": sha256(HERE / "compare_embedding_bits.py"),
                                   "release_ready": False})
    command = [sys.executable, "-I", "-B", "-c",
               "import sys,runpy; from pathlib import Path; "
               "raise SystemExit(runpy.run_path(sys.argv[1])['worker'](*map(Path,sys.argv[2:])))",
               str(Path(__file__).resolve()), str(root.resolve()), str(wheel.resolve())]
    if archive is not None:
        command.append(str(archive.resolve()))
    with (root / "worker.log").open("xb") as log:
        try:
            code = owned_run(command, timeout=LIMIT_SECONDS, stdout=log).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record(root / "exit.json", {"returncode": code, "retries": 0})
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("archive", type=Path, nargs="?")
    parser.add_argument("--hosted-fixture", action="store_true",
                        help="recreate only the pinned request; execution also requires the admitted hosted environment")
    parser.add_argument("--execute-once", type=Path, metavar="NEW_PRIVATE_DIRECTORY")
    args = parser.parse_args()
    if args.hosted_fixture != (args.archive is None):
        parser.error("select exactly one of the pinned archive or --hosted-fixture")
    if args.execute_once:
        raise SystemExit(execute_once(args.execute_once, args.wheel, args.archive))
    print(json.dumps(preflight(args.wheel, args.archive)[1], indent=2))
