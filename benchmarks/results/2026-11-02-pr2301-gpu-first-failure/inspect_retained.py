"""File-only inspection of PR 2301's exact detached WGPU failure, without loading a wheel."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from byte_study_diagnostics import exact_report_difference_summary

RUN = 41431026840
HEAD = "72772b7c68911e23dd5dbabecb9a9715db10301e"
MERGE = "8d9843e6ec2ec1236ed4932691d2d29e442f440d"
PINS = {
    "unqualified-wheel-ci-gpu": "47048ae3c38fd3717d9886db1fa82d8cd63e6b4b1e0bd241f86eb4510ca6d349",
    "unqualified-wheel-records-ci-gpu": "f8c6bcee0dd7abd69028b6d84fac90eadd294036c608f1ae1bdccd0aaeb39f42",
    "synthetic-byte-study-detached-failure": "d259725a7b22b3a37e5b0a8e02062d84c29cc45c33ac60bed827d495b10750d5",
    "synthetic-byte-study-boundary-diagnostic": "5c538ba798b3bfb6a7417cc6cc913d2710f3d5c286c9d30d48e5c03f59a910db",
}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32_order(value):
    word = struct.unpack(">I", struct.pack(">f", value))[0]
    return 0x80000000 - (word & 0x7FFFFFFF) if word >> 31 else 0x80000000 + word


def input_occurrences(request, token):
    steps = request["config"]["steps"]
    require(steps > 0, "invalid causal context")
    counts = []
    for batch in request["train_batches"]:
        n = 0
        for doc, start in batch:
            require(0 <= doc < len(request["train_documents"]), "invalid document")
            ids = request["train_documents"][doc]
            require(0 <= start and start + steps < len(ids), "missing shifted target")
            n += ids[start : start + steps].count(token)
        counts.append(n)
    return counts


def members(data, limit):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = z.infolist()
        require(
            0 < len(files) <= 8192
            and len({f.filename for f in files}) == len(files)
            and sum(f.file_size for f in files) <= limit,
            "unsafe ZIP member count/size",
        )
        return {f.filename: z.read(f) for f in files if not f.is_dir()}


def inspect(root):
    run = json.loads((root / "run-identity.json").read_bytes())
    merge = json.loads((root / "merge-identity.json").read_bytes())
    require(
        (run["id"], run["run_attempt"], run["head_sha"], run["conclusion"])
        == (RUN, 1, HEAD, "failure")
        and merge["sha"] == MERGE
        and merge["parents"][-1] == HEAD,
        "wrong run or test merge",
    )
    roster = json.loads((root / "artifacts.json").read_bytes())["artifacts"]
    files = {}
    for name, pin in PINS.items():
        (meta,) = [f for f in roster if f["name"] == name]
        archive = root / (name + ".zip")
        require(
            archive.stat().st_size == meta["size_in_bytes"] <= 16 * 1024**2,
            "different archive size",
        )
        data = archive.read_bytes()
        require(
            meta["workflow_run"]["id"] == RUN
            and meta["workflow_run"]["head_sha"] == HEAD
            and "sha256:" + sha(data) == meta["digest"] == "sha256:" + pin,
            "unbound artifact",
        )
        files[name] = members(data, 32 * 1024**2)
    original = files["unqualified-wheel-ci-gpu"]
    records = files["unqualified-wheel-records-ci-gpu"]
    require(original["build.json"] == records["build.json"], "different build record")
    build = json.loads(records["build.json"])
    installed = json.loads(records["installed.json"])
    raw = original[build["wheel"]["filename"]]
    wheel = members(raw, 256 * 1024**2)
    native = build["wheel"]["native"]
    require(
        build["source_commit"] == MERGE
        and build["role"] == "ci-gpu"
        and sha(raw)
        == build["wheel_sha256"]
        == installed["wheel_sha256"]
        == json.loads(original["original.json"])["sha256"]
        and sha(wheel[native["path"]]) == native["sha256"] == installed["native_sha256"]
        and installed["all_original_files_match"] is True
        and installed["release_ready"] is False
        and installed["build_receipt_sha256"] == sha(records["build.json"]),
        "wheel, native library or install receipt differs",
    )
    z = files["synthetic-byte-study-detached-failure"]
    require(
        set(z)
        == {
            "round-0/" + n + ".json"
            for n in (
                "summary",
                "request",
                "expected",
                "actual-0",
                "actual-1",
                "runtime",
            )
        },
        "different round/members",
    )
    summary = json.loads(z["round-0/summary.json"])
    for key, raw in z.items():
        if key != "round-0/summary.json":
            require(
                summary["files"][key.split("/")[1]] == "sha256:" + sha(raw),
                "changed diagnostic input",
            )
    runtime = json.loads(z["round-0/runtime.json"])
    require(
        runtime["native_library"]["sha256"] == native["sha256"]
        and runtime["adapter"]["backend"] == "Metal"
        and summary["round"] == 0
        and summary["comparison"] == "exact_json_bytes_no_tolerance",
        "different engine or comparison",
    )
    request = json.loads(z["round-0/request.json"])
    expected = z["round-0/expected.json"].decode()
    comparisons = []
    for worker in range(2):
        actual = z[f"round-0/actual-{worker}.json"].decode()
        diff = exact_report_difference_summary(expected, actual)
        require(
            diff == summary["difference_summaries"][worker]
            and diff["complete"]
            and not diff["groups"]["final_parameters"]["truncated"]
            and diff["groups"]["other"]["count"] == 0,
            "incomplete or different diff",
        )
        differences = []
        for e in diff["groups"]["final_parameters"]["examples"]:
            match = re.fullmatch(
                r"\$\.cases\[(\d+)\]\.final_parameters\[(\d+)\]\[(\d+)\]", e["path"]
            )
            require(match is not None, "unknown parameter path")
            case, block, index = map(int, match.groups())
            p = request["cases"][case]
            differences.append(
                {
                    "case": p["name"],
                    "parameter": p["parameters"][block]["name"],
                    "index": index,
                    "abs_delta": abs(e["actual"] - e["expected"]),
                    "f32_ulps": abs(f32_order(e["actual"]) - f32_order(e["expected"])),
                }
            )
            spec = p["parameters"][block]
            if spec["name"] == "token_embedding":
                vocab, width = spec["shape"]
                require(
                    width == request["config"]["width"] and 0 <= index < vocab * width,
                    "invalid embedding address",
                )
                token, col = divmod(index, width)
                differences[-1]["embedding_coordinate"] = [token, col]
                differences[-1]["input_occurrences_by_update"] = input_occurrences(
                    request, token
                )
        comparisons.append(
            {
                "worker": worker,
                "byte_equal": actual == expected,
                "differences": differences,
                "other_field_differences": diff["groups"]["other"]["count"],
            }
        )
    boundary = json.loads(
        files["synthetic-byte-study-boundary-diagnostic"]["result.json"]
    )
    require(
        boundary["status"] == "not_selected"
        and boundary["reason"] == "no_prepared_failure",
        "different boundary evidence",
    )
    return {
        "schema": "spiraltorch.retained_detached_pr2301.v1",
        "run": run,
        "test_merge": merge,
        "archive_sha256": PINS,
        "wheel_sha256": installed["wheel_sha256"],
        "native_sha256": native["sha256"],
        "request_sha256": sha(z["round-0/request.json"]),
        "backend": runtime["adapter"]["backend"],
        "round_zero_based": 0,
        "workers": comparisons,
        "boundary_probe": boundary["status"],
        "cause": "not established",
        "gpu_reruns": 0,
        "release_candidate": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "artifacts", type=Path, help="retained ZIPs and API identity/metadata JSONs"
    )
    print(
        json.dumps(
            inspect(p.parse_args().artifacts), sort_keys=True, indent=2, allow_nan=False
        )
    )
