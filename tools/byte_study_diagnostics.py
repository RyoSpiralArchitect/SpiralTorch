"""Bounded diagnostics for exact synthetic native-byte-study comparisons."""

import hashlib
import json
from pathlib import Path


def exact_report_differences(expected, actual, *, limit=16):
    """Describe differing paths without weakening the caller's byte equality."""
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("difference limit must be in [1,64]")
    if expected == actual:
        return []
    try:
        left, right = json.loads(expected), json.loads(actual)
    except (ValueError, TypeError):
        return [{"path": "$", "kind": "invalid_json"}]
    differences = []

    def compare(a, b, path):
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append({"path": path, "kind": "type", "expected": type(a).__name__,
                                "actual": type(b).__name__})
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                differences.append({"path": path, "kind": "keys",
                                    "missing": sorted(a.keys() - b.keys()),
                                    "extra": sorted(b.keys() - a.keys())})
            for key in sorted(a.keys() & b.keys()):
                compare(a[key], b[key], f"{path}.{key}")
                if len(differences) >= limit:
                    break
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": path, "kind": "length", "expected": len(a), "actual": len(b)})
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{index}]")
                if len(differences) >= limit:
                    break
        elif a != b:
            differences.append({"path": path, "kind": "value",
                                "expected": a[:160] if isinstance(a, str) else a,
                                "actual": b[:160] if isinstance(b, str) else b})

    compare(left, right, "$")
    return differences or [{"path": "$", "kind": "serialization_only"}]


def retain_detached_failure(directory, request, expected, actuals, round_index):
    """Retain a single bounded synthetic comparison only at an explicit path."""
    if type(round_index) is not int or not 0 <= round_index < 4 or len(actuals) != 2:
        raise ValueError("expected one of four declared two-worker comparisons")
    payloads = {"request.json": request, "expected.json": expected,
                **{f"actual-{index}.json": raw for index, raw in enumerate(actuals)}}
    payloads = {name: raw.encode("utf-8") if isinstance(raw, str) else raw for name, raw in payloads.items()}
    if any(type(raw) is not bytes or len(raw) > 64 * 1024 * 1024 for raw in payloads.values()):
        raise ValueError("diagnostic payload must be at most 64 MiB")
    target = Path(directory) / f"round-{round_index}"
    target.mkdir(parents=True, exist_ok=False)
    ids = {}
    for name, raw in payloads.items():
        with (target / name).open("xb") as stream:
            stream.write(raw)
        ids[name] = "sha256:" + hashlib.sha256(raw).hexdigest()
    report = {"schema": "spiraltorch.synthetic_detached_diagnostic.v1", "round": round_index,
              "comparison": "exact_json_bytes_no_tolerance", "files": ids,
              "differences": [exact_report_differences(expected, raw) for raw in actuals]}
    with (target / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def retain_resume_failure(directory, request, *, full_report, first_report, resumed_report,
                          full_checkpoint, first_checkpoint, resumed_checkpoint):
    """Keep the first synthetic resume mismatch, including the restart boundary.

    Independent full and partial runs do not isolate checkpoint restoration:
    preserve their histories and optimizer states, not just final weights.
    """
    payloads = {"request.json": request, "full-report.json": full_report,
                "first-report.json": first_report, "resumed-report.json": resumed_report,
                "full-checkpoint.json": full_checkpoint, "first-checkpoint.json": first_checkpoint,
                "resumed-checkpoint.json": resumed_checkpoint}
    payloads = {name: raw.encode("utf-8") if isinstance(raw, str) else raw
                for name, raw in payloads.items()}
    if any(type(raw) is not bytes or len(raw) > 64 * 1024 * 1024 for raw in payloads.values()):
        raise ValueError("diagnostic payload must be at most 64 MiB")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    ids = {}
    for name, raw in payloads.items():
        with (target / name).open("xb") as stream:
            stream.write(raw)
        ids[name] = "sha256:" + hashlib.sha256(raw).hexdigest()
    report = {"schema": "spiraltorch.synthetic_resume_diagnostic.v1",
              "comparison": "exact_json_bytes_no_tolerance", "files": ids,
              "report_differences": exact_report_differences(full_report, resumed_report),
              "checkpoint_differences": exact_report_differences(full_checkpoint, resumed_checkpoint)}
    with (target / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def assert_exact_resume(directory, request, *, full_report, first_report, resumed_report,
                        full_checkpoint, first_checkpoint, resumed_checkpoint):
    """Fail either exact comparison, whether or not retention is enabled."""
    if full_report == resumed_report and full_checkpoint == resumed_checkpoint:
        return
    if directory:
        retain_resume_failure(directory, request, full_report=full_report, first_report=first_report,
                              resumed_report=resumed_report, full_checkpoint=full_checkpoint,
                              first_checkpoint=first_checkpoint, resumed_checkpoint=resumed_checkpoint)
    raise AssertionError("exact contextual-mixture resume comparison: " + json.dumps({
        "reports": exact_report_differences(full_report, resumed_report),
        "checkpoints": exact_report_differences(full_checkpoint, resumed_checkpoint),
    }))
