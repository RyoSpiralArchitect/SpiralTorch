"""Fixed 128-update acceptance exercise; Rust executes, Python retains/compares.

Importing this module does not import SpiralTorch or acquire a runtime. This
exercise alone is not an artifact-qualified round or publication authority.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

import byte_corpus_study as protocol
import byte_corpus_optimizer as optimizer
import byte_corpus_gradient_clip as clipping
from test_byte_corpus_gradient_clip import bind, learning_design
from test_byte_corpus_optimizer import wrap

PROFILE = "spiraltorch.byte_release_long_resume.v1"
UPDATES = 128
SPLITS = (1, 64)
CASE_NAMES = ("plain", "geometry", "frozen")
REQUEST_SHA256 = "6c8feea42a494c855fd659a86b2f3b132b3d877277f3912b66af1be215f440f2"
MAX_FILE_BYTES = 64 * 1024 * 1024


def request_bytes():
    design = learning_design()
    pattern = design["train_batches"]
    design["train_batches"] = [copy.deepcopy(pattern[i % len(pattern)]) for i in range(UPDATES)]
    design["checkpoint_every"] = 64
    request = bind(wrap(design))
    if protocol.request_version(request) != 10:
        raise ValueError("long qualification requires the clipped AdamW study contract")
    raw = protocol.encoded(request)
    if protocol.digest(raw) != REQUEST_SHA256:
        raise ValueError("frozen long-study recipe changed; do not silently redefine this profile")
    return raw


def _raw(value):
    if type(value) is str:
        value = value.encode("utf-8")
    if type(value) is not bytes or not value or len(value) > MAX_FILE_BYTES:
        raise ValueError("expected nonempty bounded native bytes")
    return value


def _save(path, value):
    raw = _raw(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def require_exact(expected, actual, label):
    if _raw(expected) != _raw(actual):
        raise AssertionError("exact byte mismatch: " + label)


def _bits(values):
    if type(values) is not list or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("parameter values must be finite numbers")
    return struct.pack(f"<{len(values)}f", *values)


def validate_completed_report(raw, request_raw):
    """Reject empty identical outputs and frozen/no-op learning before acceptance."""
    if protocol.digest(request_raw) != REQUEST_SHA256:
        raise ValueError("long-study request identity differs")
    request = protocol.decode_json(request_raw)
    report = protocol.decode_json(_raw(raw))
    cases = report.get("cases") if isinstance(report, dict) else None
    if (not isinstance(report, dict)
            or report.get("schema") != "spiraltorch.byte_corpus.result.v10"
            or report.get("request_sha256") != protocol.digest(request_raw)
            or type(cases) is not list or len(cases) != len(CASE_NAMES)
            or any(type(case) is not dict for case in cases)
            or tuple(case.get("name") for case in cases) != CASE_NAMES):
        raise ValueError("long report identity or complete case order differs")
    optimizer.metadata(request, report)
    clipping.metadata(request, report)
    controls = optimizer.recipe(request)
    for spec, case in zip(request["cases"], cases):
        for field in ("seed", "geometry", "geometry_update"):
            if type(case.get(field)) is not type(spec[field]) or case[field] != spec[field]:
                raise ValueError("long report case metadata differs")
        history = case.get("training")
        if (type(history) is not list or len(history) != UPDATES
                or any(type(point) is not dict or type(point.get("revision")) is not int
                       or point["revision"] != i + 1
                       or type(point.get("ce")) not in (int, float) or not math.isfinite(point["ce"])
                       for i, point in enumerate(history))):
            raise ValueError("long report must retain every finite update")
        validation = case.get("validation")
        if (type(validation) is not list or len(validation) != 3
                or any(type(point) is not dict or type(point.get("revision")) is not int
                       or point["revision"] != revision
                       or type(point.get("batch_losses")) is not list or len(point["batch_losses"]) != 1
                       or any(type(v) not in (int, float) or not math.isfinite(v)
                              for v in point["batch_losses"])
                       for point, revision in zip(validation, (0, 64, UPDATES)))):
            raise ValueError("long report validation schedule differs")
        final = case.get("final_parameters")
        if type(final) is not list or len(final) != len(spec["parameters"]):
            raise ValueError("long report parameter coverage differs")
        changed_backbone = changed_geometry = False
        for descriptor, values in zip(spec["parameters"], final):
            if type(values) is not list or len(values) != len(descriptor["values"]):
                raise ValueError("long report parameter shape differs")
            changed = _bits(values) != _bits(descriptor["values"])
            if protocol.parameter_frozen(spec, descriptor["name"]) and changed:
                raise ValueError("frozen parameter changed")
            if descriptor["name"].startswith("geometry."):
                changed_geometry |= changed
            else:
                changed_backbone |= changed
        if not changed_backbone or (spec["geometry"] and not protocol.frozen(spec) and not changed_geometry):
            raise ValueError("long report does not demonstrate active parameter updates")
        optimizer.validate_state(spec, case.get("optimizer_state"), controls, UPDATES,
                                 protocol.parameter_frozen)
    return report


def exercise(st, output):
    """One ordinary full/advance/split-1/split-64 sequence, without a retry.

    The enclosing admitted-candidate supervisor must own the deadline, children,
    import guards and exact wheel identity. A fresh output is mandatory.
    """
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"schema": PROFILE, "status": "incomplete", "release_ready": False,
              "stage": "preflight", "splits": [], "updates": UPDATES}
    started = time.monotonic()
    try:
        raw = _save(output / "request.json", request_bytes())
        study = st.nn.ByteCorpusStudy.from_json(raw)
        if study.total_updates != UPDATES or study.request_sha256 != protocol.digest(raw):
            raise ValueError("native study identity differs")
        for cursor in (*SPLITS, UPDATES):
            study.validate_segment(cursor)
        result["request_sha256"] = protocol.digest(raw)
        result["stage"] = "ordinary_full"
        ordinary = _save(output / "ordinary-report.json", study.run_wgpu())
        validate_completed_report(ordinary, raw)
        result["stage"] = "full_segment"
        full = study.advance_wgpu()
        full_report = _save(output / "full-report.json", full.report_json)
        full_checkpoint = _save(output / "full-checkpoint.json", full.checkpoint.to_json())
        require_exact(ordinary, full_report, "run_wgpu versus complete advance_wgpu")
        if full.checkpoint.completed_updates != UPDATES:
            raise ValueError("complete checkpoint cursor differs")
        for cursor in SPLITS:
            result["stage"] = f"split_{cursor}"
            directory = output / f"split-{cursor}"
            directory.mkdir()
            first = study.advance_wgpu(cursor)
            _save(directory / "first-report.json", first.report_json)
            saved = _save(directory / "first-checkpoint.json", first.checkpoint.to_json())
            loaded = study.checkpoint_from_json(saved)
            if first.checkpoint.completed_updates != cursor or loaded.completed_updates != cursor:
                raise ValueError("partial checkpoint cursor differs")
            require_exact(saved, loaded.to_json(), "saved checkpoint parsing")
            resumed = study.advance_wgpu(resume=loaded)
            resumed_report = _save(directory / "resumed-report.json", resumed.report_json)
            resumed_checkpoint = _save(directory / "resumed-checkpoint.json", resumed.checkpoint.to_json())
            require_exact(ordinary, resumed_report, f"split-{cursor} report")
            require_exact(full_checkpoint, resumed_checkpoint, f"split-{cursor} checkpoint")
            require_exact(saved, first.checkpoint.to_json(), "original saved checkpoint immutability")
            require_exact(saved, loaded.to_json(), "loaded saved checkpoint immutability")
            result["splits"].append(cursor)
        result.update(status="passed", stage="complete")
        return result
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        result["files"] = {
            str(path.relative_to(output)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                           "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file()
        }
        _save(output / "result.json", json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
