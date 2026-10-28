"""Bounded diagnostics for exact synthetic native-byte-study comparisons."""

import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys


def runtime_identity(st):
    """Post-failure identity only: no environment, host name or model payloads.

    The tensor device reuses the study's process-default runtime. This does not
    dispatch or read back tensors, and must not run before the comparison.
    """
    native = importlib.import_module("spiraltorch.spiraltorch")
    library = Path(native.__file__)
    digest = hashlib.sha256()
    with library.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema": "spiraltorch.synthetic_gpu_runtime.v1",
        "python": sys.version,
        "system": platform.system(), "release": platform.release(),
        "os_version": platform.version(), "machine": platform.machine(),
        "native_library": {"filename": library.name, "sha256": digest.hexdigest()},
        "adapter": st.wgpu.WgpuTensorDevice.create().adapter_info(),
    }


def _runtime_payload(collect):
    try:
        value = collect()
        if not isinstance(value, dict):
            raise ValueError("runtime context must be an object")
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        if len(raw) > 16 * 1024:
            raise ValueError("runtime context exceeds 16 KiB")
        return raw
    except Exception as error:
        # Metadata failures must not discard the numerical failure evidence.
        return json.dumps({"status": "collection_failed", "error_type": type(error).__name__}).encode("utf-8")


def _difference_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError("difference limit must be in [1,64]")


def _unique_object(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError("non-finite JSON value")


def _finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON value")
    return value


def _report_differences(expected, actual):
    if expected == actual:
        return
    try:
        left, right = (json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
                                 parse_float=_finite_float)
                       for raw in (expected, actual))
    except RecursionError:
        yield False, {"path": "$", "kind": "analysis_limit"}
        return
    except (ValueError, TypeError):
        yield False, {"path": "$", "kind": "invalid_json"}
        return

    def compare(a, b, path, parts):
        if len(parts) > 128:
            raise RecursionError("difference traversal depth exceeded")
        parameter = (len(parts) >= 3 and parts[0] == "cases" and type(parts[1]) is int
                     and parts[2] == "final_parameters")
        if type(a) is not type(b):
            yield parameter, {"path": path, "kind": "type", "expected": type(a).__name__,
                              "actual": type(b).__name__}
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                missing, extra = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
                yield parameter, {"path": path, "kind": "keys", "missing": missing[:16],
                                  "extra": extra[:16], "missing_count": len(missing),
                                  "extra_count": len(extra)}
            for key in sorted(a.keys() & b.keys()):
                yield from compare(a[key], b[key], f"{path}.{key}", (*parts, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                yield parameter, {"path": path, "kind": "length", "expected": len(a), "actual": len(b)}
            for index, (x, y) in enumerate(zip(a, b)):
                yield from compare(x, y, f"{path}[{index}]", (*parts, index))
        elif a != b or (isinstance(a, float) and math.copysign(1., a) != math.copysign(1., b)):
            yield parameter, {"path": path, "kind": "value",
                              "expected": a[:160] if isinstance(a, str) else a,
                              "actual": b[:160] if isinstance(b, str) else b}

    found = False
    try:
        for item in compare(left, right, "$", ()):
            found = True
            yield item
    except RecursionError:
        yield False, {"path": "$", "kind": "analysis_limit"}
        return
    if not found:
        yield False, {"path": "$", "kind": "serialization_only"}


def exact_report_differences(expected, actual, *, limit=16):
    """Describe differing paths without weakening the caller's byte equality."""
    _difference_limit(limit)
    return [item for _, item in itertools.islice(_report_differences(expected, actual), limit)]


def exact_report_difference_summary(expected, actual, *, limit=16):
    """Count the whole comparison with separate bounded parameter/other samples.

    Counts describe mismatch events: a missing subtree is one structural event,
    not a claim about its absent leaves. Unparseable inputs have unknown totals.
    This is post-failure analysis, never a replacement for exact raw equality.
    """
    _difference_limit(limit)
    groups = {name: {"count": 0, "examples": [], "truncated": False}
              for name in ("final_parameters", "other")}
    incomplete = None
    for parameter, item in _report_differences(expected, actual):
        if item["kind"] in ("invalid_json", "analysis_limit"):
            incomplete = item["kind"]
        group = groups["final_parameters" if parameter else "other"]
        group["count"] += 1
        if len(group["examples"]) < limit:
            group["examples"].append(item)
        else:
            group["truncated"] = True
    count = sum(group["count"] for group in groups.values()) if incomplete is None else None
    if incomplete is not None:
        for group in groups.values():
            group["count"] = None
    return {"schema": "spiraltorch.synthetic_exact_difference_summary.v1",
            "raw_equal": expected == actual, "complete": incomplete is None,
            "incomplete_reason": incomplete,
            "difference_count": count, "groups": groups}


def retain_detached_failure(directory, request, expected, actuals, round_index, *, runtime_context=None):
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
    if runtime_context is not None:
        # Persist numerical evidence before invoking fallible metadata collection.
        raw = _runtime_payload(runtime_context)
        with (target / "runtime.json").open("xb") as stream:
            stream.write(raw)
        ids["runtime.json"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    report = {"schema": "spiraltorch.synthetic_detached_diagnostic.v1", "round": round_index,
              "comparison": "exact_json_bytes_no_tolerance", "files": ids,
              "differences": [exact_report_differences(expected, raw) for raw in actuals],
              "difference_summaries": [exact_report_difference_summary(expected, raw) for raw in actuals]}
    with (target / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def retain_resume_failure(directory, request, *, full_report, first_report, resumed_report,
                          full_checkpoint, first_checkpoint, resumed_checkpoint, runtime_context=None):
    """Keep the first synthetic resume mismatch, including the restart boundary.

    Independent full and partial runs do not isolate checkpoint restoration:
    preserve their histories and optimizer states, not just final weights.
    """
    payloads = {"request.json": request, "full-report.json": full_report,
                "first-report.json": first_report, "resumed-report.json": resumed_report,
                "full-checkpoint.json": full_checkpoint, "first-checkpoint.json": first_checkpoint,
                "resumed-checkpoint.json": resumed_checkpoint}
    if runtime_context is not None:
        payloads["runtime.json"] = _runtime_payload(runtime_context)
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
                        full_checkpoint, first_checkpoint, resumed_checkpoint, runtime_context=None):
    """Fail either exact comparison, whether or not retention is enabled."""
    if full_report == resumed_report and full_checkpoint == resumed_checkpoint:
        return
    if directory:
        retain_resume_failure(directory, request, full_report=full_report, first_report=first_report,
                              resumed_report=resumed_report, full_checkpoint=full_checkpoint,
                              first_checkpoint=first_checkpoint, resumed_checkpoint=resumed_checkpoint,
                              runtime_context=runtime_context)
    raise AssertionError("exact contextual-mixture resume comparison: " + json.dumps({
        "reports": exact_report_differences(full_report, resumed_report),
        "checkpoints": exact_report_differences(full_checkpoint, resumed_checkpoint),
    }))


def assert_exact_run_resume(directory, request, full_report, resumed_report, *,
                            boundaries, runtime_context=None):
    """Compare the uninterrupted API without replacing it by checkpoint capture.

    `run_wgpu` deliberately returns no checkpoint. Collect only the already
    available partial/resumed boundaries, lazily after an actual mismatch.
    """
    if full_report == resumed_report:
        return
    differences = exact_report_differences(full_report, resumed_report)
    if directory:
        first_report, first_checkpoint, resumed_checkpoint = boundaries()
        payloads = {"request.json": request, "full-report.json": full_report,
                    "resumed-report.json": resumed_report, "first-report.json": first_report,
                    "first-checkpoint.json": first_checkpoint, "resumed-checkpoint.json": resumed_checkpoint}
        if runtime_context is not None:
            payloads["runtime.json"] = _runtime_payload(runtime_context)
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
        report = {"schema": "spiraltorch.synthetic_run_resume_diagnostic.v1", "files": ids,
                  "comparison": "exact_report_bytes_no_tolerance", "report_differences": differences,
                  "full_checkpoint": "not_returned_by_run_api"}
        with (target / "summary.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    raise AssertionError("exact uninterrupted/resumed study comparison: " + json.dumps(differences))
