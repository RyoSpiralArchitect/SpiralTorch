"""Input contract for the two frozen projective-trajectory diagnostics."""
import hashlib
import json
import math
import re


FIXTURE_SHA256 = "07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358"
REPORT_SCHEMA = "spiraltorch.resident_byte_geometry_projective.validation.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(values, length, label):
    require(isinstance(values, list) and len(values) == length, f"{label}: wrong vector length")
    require(all(finite(v) for v in values), f"{label}: non-finite or non-numeric vector")


def vectors(values, lengths, label):
    require(isinstance(values, list) and len(values) == len(lengths), f"{label}: wrong tensor count")
    for slot, (value, length) in enumerate(zip(values, lengths, strict=True)):
        vector(value, length, f"{label}[{slot}]")


def validate_report(fixture, report, *, schema=REPORT_SCHEMA):
    require(report.get("schema") == schema, "unexpected runtime report schema")
    cases, checks = fixture["cases"], report.get("checks")
    require(isinstance(checks, list) and len(checks) == len(cases), "wrong runtime case count")
    for case, actual in zip(cases, checks, strict=True):
        name = case["name"]
        require(actual.get("name") == name, "runtime case identity/order mismatch")
        descriptors = case["parameters"]
        shapes = [p["shape"] for p in descriptors]
        lengths = [math.prod(s) for s in shapes]
        require(actual.get("parameter_count") == len(descriptors), f"{name}: wrong parameter count")
        require(actual.get("parameter_names") == [p["name"] for p in descriptors], f"{name}: parameter names changed")
        require(actual.get("parameter_shapes") == shapes, f"{name}: parameter shapes changed")
        vector(actual["output"], len(case["output"]), f"{name}.output")
        vectors(actual["parameter_gradients"], lengths, f"{name}.initial gradients")
        learning = actual["learning"]
        trace = learning["trace"]
        require(learning.get("steps") == 16 and isinstance(trace, list) and len(trace) == 16,
                f"{name}: expected exactly 16 updates")
        geometry_lengths = [len(g) for g in case["learning"]["trace"][0]["geometry_gradients"]]
        for step, row in enumerate(trace, 1):
            label = f"{name}.step{step}"
            require(type(row.get("revision")) is int and row["revision"] == step, f"{label}: revision/order mismatch")
            require(finite(row["loss"]), f"{label}: invalid loss")
            vectors(row["parameters"], lengths, f"{label}.parameters")
            vectors(row["geometry_gradients"], geometry_lengths, f"{label}.geometry gradients")


def inherited_failures(actual):
    """Publish only named gates/scalars, never arbitrary report payloads."""
    rows = []
    for failure in actual["qualification_failures"]:
        require(failure["gate"] == "geometry_relative_l2", "unexpected legacy failure gate")
        step, slot = failure["step"], failure["geometry_slot"]
        require(type(step) is int and 1 <= step <= 16 and type(slot) is int
                and 0 <= slot < len(actual["learning"]["trace"][step-1]["geometry_gradients"]),
                "invalid legacy failure location")
        require(failure["threshold"] == .002, "legacy threshold changed")
        rows.append({"gate": failure["gate"], "step": step, "geometry_slot": slot, "threshold": .002})
    query = actual.get("query_controls_error")
    if query is not None:
        match = re.fullmatch(r"geometry derivative is inert or inaccurate: norm=([^,]+), actual=([^,]+), relative=(.+)", query)
        require(match is not None, "unexpected legacy query error")
        query = dict(zip(("reference_norm", "actual_norm", "relative_l2"), map(float, match.groups()), strict=True))
        require(all(finite(v) for v in query.values()), "non-finite legacy query error")
    return rows, query


def load_inputs(fixture_path, current):
    hashes = []

    def read(path):
        raw = path.read_bytes()
        hashes.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    fixture = read(fixture_path)
    require(hashes[0]["sha256"] == FIXTURE_SHA256, "frozen fixture changed")
    native = read(current / "native-projective.json")
    browser = read(current / "browser-lm.json")
    require(browser.get("schema") == "spiraltorch.query_pullback_bundle.browser.v1"
            and browser.get("fixture_sha256") == FIXTURE_SHA256, "browser fixture identity mismatch")
    reports = {"native": native, "browser": browser["legacy"]}
    for report in reports.values():
        validate_report(fixture, report)
        for case in report["checks"]:
            inherited_failures(case)
    return fixture, reports, hashes
