"""Synthetic admission tests, not GPU or performance evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_module_resident_intervals as validation


def fixture():
    p, routes = validation.PROTOCOL, validation.ROUTES
    native = dict(cases=[dict(seed=seed, shape=shape, depth=depth, reference=[0., 1.])
                        for seed in (17, 29, 43)
                        for shape, depth in (([2, 3, 7], 2), ([2, 8, 64], 8), ([4, 8, 128], 16))])
    total = (p["warmup"] + p["samples"]) * p["forwards"]
    document = dict(schema="spiraltorch.module_completed_intervals.v1", status="passed", page_errors=[],
        fixture_request="nn-module-intervals", protocol=p.copy(), routes=routes.copy(),
        console_messages=[], cross_origin_isolated=False, browser_version="synthetic", user_agent="synthetic",
        clock=dict(positive_deltas_ms=[0.1], equal_reads=9, minimum_positive_delta_ms=0.1), cases=[])
    for matrix in range(p["matrices"]):
        for frozen in native["cases"] if matrix % 2 == 0 else reversed(native["cases"]):
            case = {k: copy.deepcopy(frozen[k]) for k in ("seed", "shape", "depth")}
            case.update(matrix=matrix, setup_order=["baseline", "candidate"] if matrix % 2 == 0 else ["candidate", "baseline"],
                last_outputs={route: [0., 1.] for route in routes},
                warmup_completed_reads={route: p["warmup"] * p["forwards"] for route in routes},
                warmup_max_abs_error={route: 0. for route in routes},
                adapters={version: dict(backend="BrowserWebGpu", device_type="Other") for version in ("baseline", "candidate")},
                cache={version: dict(compilations="1", cache_hits=str(total), submitted_forwards=str(total+1))
                       for version in ("baseline", "candidate")},
                scalar_dispatches={version: str(total+1) for version in ("baseline", "candidate")}, samples=[])
            for block in range(p["samples"]):
                offset = (matrix + frozen["seed"] + block + p["warmup"]) % len(routes)
                order = routes[offset:] + routes[:offset]
                for route in order:
                    case["samples"].append(dict(block=block, route=route, order=order,
                        forwards=p["forwards"], completed_reads=p["forwards"], submitted_forwards=str(p["forwards"]),
                        elapsed_ms=40. if route == "candidate_module_d2h" else 50., max_abs_error=0.))
            document["cases"].append(case)
    return document, native


class Intervals(unittest.TestCase):
    def test_same_terminal_api_requires_explicit_fixture_and_both_api_labels(self):
        document, native = fixture()
        document.update(schema="spiraltorch.module_terminal_intervals.v1",
                        fixture_request="nn-module-terminal-matched-intervals",
                        module_apis=dict(baseline="forwardSnapshot", candidate="forwardSnapshot"))
        self.assertEqual(validation.validate(document, native, terminal_capture=True,
                                            same_terminal_api=True)["checked_forwards_including_warmup"], 405504)
        with self.assertRaises(ValueError): validation.validate(document, native)
        with self.assertRaises(ValueError): validation.validate(document, native, same_terminal_api=True)
        with self.assertRaises(ValueError): validation.validate(document, native, terminal_capture=True)
        for mutate in (
            lambda d: d.__setitem__("fixture_request", "nn-module-terminal-intervals"),
            lambda d: d["module_apis"].__setitem__("baseline", "forward_then_snapshot"),
            lambda d: d["module_apis"].__setitem__("candidate", "forward_then_snapshot"),
        ):
            bad = copy.deepcopy(document); mutate(bad)
            with self.assertRaises(ValueError):
                validation.validate(bad, native, terminal_capture=True, same_terminal_api=True)

    def test_terminal_mode_requires_distinct_api_and_schema_boundaries(self):
        document, native = fixture()
        with self.assertRaises(ValueError): validation.validate(document,native,terminal_capture=True)
        document.update(schema="spiraltorch.module_terminal_intervals.v1",fixture_request="nn-module-terminal-intervals",
                        module_apis=dict(baseline="forward_then_snapshot",candidate="forwardSnapshot"))
        self.assertEqual(validation.validate(document,native,terminal_capture=True)["checked_forwards_including_warmup"],405504)
        with self.assertRaises(ValueError): validation.validate(document,native)
        document["module_apis"]["candidate"]="forward_then_snapshot"
        with self.assertRaises(ValueError): validation.validate(document,native,terminal_capture=True)

    def test_fixed_matrix_and_per_forward_normalization(self):
        document, native = fixture()
        report = validation.validate(document, native)
        self.assertEqual(len(report["cases"]), 36)
        self.assertEqual(report["checked_forwards_including_warmup"], 405504)
        self.assertEqual(report["checked_values_including_warmup"], 811008)
        self.assertEqual(report["minimum_interval_over_observed_clock_delta"], 400.)
        self.assertEqual(report["cases"][0]["summary"]["candidate_module_d2h"]["median_ms_per_forward"], 40./256)
        for row in report["aggregates"]:
            self.assertEqual(row["module"]["median_of_case_median_ratios"], 0.8)
            self.assertEqual(row["scalar"]["median_of_case_median_ratios"], 1.)

    def test_slow_interval_is_retained_even_when_median_is_unchanged(self):
        document, native = fixture()
        row = next(sample for sample in document["cases"][0]["samples"] if sample["route"] == "candidate_module_d2h")
        row["elapsed_ms"] = 10000.
        report = validation.validate(document, native)
        self.assertEqual(report["aggregates"][0]["module"]["median_of_case_median_ratios"], 0.8)
        self.assertGreater(report["aggregates"][0]["module"]["pooled_total_ratio"], 1.)
        self.assertEqual(report["cases"][0]["summary"]["candidate_module_d2h"]["max_interval_ms"], 10000.)

    def test_missing_matrix_reordered_cases_and_routes_fail(self):
        document, native = fixture()
        mutations = [lambda d: d["cases"].pop(), lambda d: d["cases"].reverse(),
                     lambda d: d["cases"][0]["setup_order"].reverse(),
                     lambda d: d["cases"][0]["samples"].pop(),
                     lambda d: d["cases"][0]["samples"].reverse(),
                     lambda d: d["cases"][0].__setitem__("shape", [1, 1]),
                     lambda d: d["protocol"].__setitem__("forwards", 128)]
        for mutation in mutations:
            bad = copy.deepcopy(document); mutation(bad)
            with self.assertRaises(ValueError): validation.validate(bad, native)

    def test_final_read_only_or_faked_counts_fail(self):
        document, native = fixture()
        for key, value in (("completed_reads", 1), ("completed_reads", True), ("forwards", 8),
                           ("submitted_forwards", "255"), ("block", True)):
            bad = copy.deepcopy(document); bad["cases"][0]["samples"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): validation.validate(bad, native)
        for key, value in (("scalar_dispatches", {"baseline": "1", "candidate": "1"}),
                           ("warmup_completed_reads", {route: 1 for route in validation.ROUTES})):
            bad = copy.deepcopy(document); bad["cases"][0][key] = value
            with self.assertRaises(ValueError): validation.validate(bad, native)

    def test_invalid_errors_timings_and_clock_are_rejected(self):
        document, native = fixture()
        for key, value in (("elapsed_ms", 0), ("elapsed_ms", float("nan")), ("elapsed_ms", True),
                           ("max_abs_error", float("inf")), ("max_abs_error", -1), ("max_abs_error", 1.)):
            bad = copy.deepcopy(document); bad["cases"][0]["samples"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): validation.validate(bad, native)
        for key, value in (("positive_deltas_ms", [0]), ("minimum_positive_delta_ms", 0.2),
                           ("equal_reads", True)):
            bad = copy.deepcopy(document); bad["clock"][key] = value
            with self.assertRaises(ValueError): validation.validate(bad, native)

    def test_guard_failure_or_wrong_device_is_not_admitted(self):
        document, native = fixture()
        for mutate in (lambda d: d["page_errors"].append("device lost"),
                       lambda d: d["console_messages"].append(dict(type="error", text="invalid buffer")),
                       lambda d: d["cases"][0]["adapters"]["baseline"].__setitem__("device_type", "Cpu"),
                       lambda d: d["cases"][0]["cache"]["candidate"].__setitem__("compilations", "2"),
                       lambda d: d["cases"][0]["last_outputs"][validation.ROUTES[0]].__setitem__(0, True),
                       lambda d: d["cases"][0]["last_outputs"][validation.ROUTES[0]].__setitem__(0, 2.)):
            bad = copy.deepcopy(document); mutate(bad)
            with self.assertRaises(ValueError): validation.validate(bad, native)

    def test_products_require_success_and_hashes_including_javascript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "client").mkdir()
            for name in ("spiraltorch_wasm.js", "spiraltorch_wasm_bg.wasm"):
                (root / "client" / name).write_bytes(name.encode())
            receipt = dict(status="passed", source=dict(commit="synthetic"),
                products={"client/" + name: validation.digest(root / "client" / name)
                          for name in ("spiraltorch_wasm.js", "spiraltorch_wasm_bg.wasm")})
            path = root / "receipt.json"; path.write_text(json.dumps(receipt))
            _, assets = validation.product_assets(path, "/baseline/")
            self.assertEqual(set(assets), {"/baseline/spiraltorch_wasm.js", "/baseline/spiraltorch_wasm_bg.wasm"})
            (root / "client/spiraltorch_wasm.js").write_bytes(b"different shim")
            with self.assertRaises(ValueError): validation.product_assets(path, "/baseline/")
            receipt["status"] = "error"; path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): validation.product_assets(path, "/baseline/")

    def test_streamed_cases_must_match_document_even_with_updated_digest(self):
        document, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            browser = Path(directory) / "browser.json"
            stream = browser.with_name(browser.name + ".cases.jsonl")
            stream.write_text("".join(json.dumps(row) + "\n" for row in document["cases"]))
            document["interval_state_artifacts"] = dict(path=stream.name, rows=36,
                bytes=stream.stat().st_size, sha256=validation.digest(stream))
            self.assertEqual(validation.admit_case_stream(document, browser), stream)
            changed = copy.deepcopy(document["cases"])
            changed[0]["samples"][0]["elapsed_ms"] = 0.001
            stream.write_text("".join(json.dumps(row) + "\n" for row in changed))
            document["interval_state_artifacts"].update(bytes=stream.stat().st_size, sha256=validation.digest(stream))
            with self.assertRaises(ValueError): validation.admit_case_stream(document, browser)


if __name__ == "__main__":
    unittest.main()
