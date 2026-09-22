"""Synthetic mutation tests, not timing evidence."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softmax_protocol as p

TEST_KEYS = {(1, 3, 0), (1, 3, 4)}


def fixture(family="native"):
    torch = family == "torch"
    report = dict(schema=p.TORCH if torch else p.SCHEMA, status="passed", warmup=3, blocks=9,
                  bursts=[1, 4], comparison=p.COMPARISON, finite_domain_fix_in_both=True,
                  domain_cases=48, subgroup_exercised=False, adapter="synthetic fixture",
                  page_errors=[], console_messages=[], browser_version="synthetic",
                  browser_adapter_probe=dict(is_fallback_adapter=False),
                  asset_sha256={"synthetic": "0" * 64}, input_protocol=p.SCHEMA,
                  devices=["cpu", "mps"], torch_version="synthetic", compiled=False,
                  preallocated_outputs=True, intra_op_threads=4, inter_op_threads=1, cases=[])
    for rows, cols, mode in sorted(p.KEYS):
        values = p.inputs(rows, cols)
        expected = p.oracle(values, rows, cols, mode)
        items = []
        for burst in (1, 4):
            for block in range(9):
                order = [0, 1] if (block + 3 + rows + cols + mode) % 2 == 0 else [1, 0]
                for route in (("cpu", "mps") if torch else ("redundant", "deduplicated")):
                    items.append(dict(block=block, burst=burst, route=route, order=order,
                                      elapsed_ms=1., max_abs_error=0.))
        report["cases"].append(dict(rows=rows, cols=cols, mode=mode, input=values,
                                    reference=expected, last_outputs=[expected, expected], intervals=items))
    return report


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(p, "KEYS", TEST_KEYS).start()

    def test_complete_summary_roundtrip(self):
        result = p.analyze([fixture()] * 3, [fixture("browser")] * 3, [fixture("torch")] * 3)
        p.validate_summary(result)
        self.assertEqual(result["descriptive_summary"]["native"]["geomean_redundant_over_deduplicated"], 1.)
        self.assertEqual(sum(len(c["intervals"]) for c in result["cases"]), 648)

    def test_identity_domain_and_control_mutations(self):
        for field, value in [("schema", "old"), ("comparison", "other"),
                             ("finite_domain_fix_in_both", False), ("domain_cases", 47),
                             ("subgroup_exercised", 0), ("adapter", "device_type: Cpu")]:
            with self.subTest(field=field):
                raw = fixture()
                raw[field] = value
                with self.assertRaises(ValueError):
                    p.admit(raw, "native")
        for field, value in [("preallocated_outputs", False), ("compiled", True),
                             ("devices", ["cpu"]), ("input_protocol", "other"), ("intra_op_threads", 8)]:
            raw = fixture("torch")
            raw[field] = value
            with self.assertRaises(ValueError):
                p.admit(raw, "torch")

    def test_missing_duplicate_nonfinite_wrong_bytes_and_output(self):
        for problem in ("missing", "duplicate", "input", "output", "reference", "nan",
                        "duration", "interval", "route", "order", "error", "bool"):
            raw = copy.deepcopy(fixture())
            c = raw["cases"][0]
            i = c["intervals"][0]
            if problem == "missing":
                raw["cases"].pop()
            elif problem == "duplicate":
                raw["cases"][1] = c
            elif problem == "input":
                c["input"][0] += 1.
            elif problem == "output":
                c["last_outputs"][0][0] = 0.
            elif problem == "reference":
                c["reference"] = []
            elif problem == "nan":
                c["last_outputs"][0][0] = float("nan")
            elif problem == "duration":
                i["elapsed_ms"] = float("inf")
            elif problem == "interval":
                c["intervals"].append(i)
            elif problem == "route":
                i["route"] = "subgroup"
            elif problem == "order":
                i["order"] = list(reversed(i["order"]))
            elif problem == "error":
                i["max_abs_error"] = -1.
            else:
                i["block"] = False
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                p.admit(raw, "native")

    def test_browser_errors_and_incomplete_rounds_fail(self):
        for issue in ("page_errors", "console_messages", "fallback"):
            raw = fixture("browser")
            if issue == "fallback":
                raw["browser_adapter_probe"]["is_fallback_adapter"] = True
            else:
                raw[issue] = ["error"]
            with self.assertRaises(ValueError):
                p.admit(raw, "browser")
        with self.assertRaises(ValueError):
            p.analyze([fixture()] * 2, [fixture("browser")] * 3, [fixture("torch")] * 3)

    def test_summary_mutations_fail_even_when_resealed(self):
        result = p.analyze([fixture()] * 3, [fixture("browser")] * 3, [fixture("torch")] * 3)
        for issue in ("median", "interval", "aggregate", "scaled", "hash", "grid"):
            bad = copy.deepcopy(result)
            c = bad["cases"][0]
            if issue == "median":
                c["summary"][0]["median_interval_ms"]["cpu"] = 99.
            elif issue == "interval":
                c["intervals"].pop()
            elif issue == "aggregate":
                bad["descriptive_summary"]["native"]["cells_over_1"] = 99
            elif issue == "scaled":
                c["max_scaled_errors"]["mps"] = float("nan")
            elif issue == "hash":
                c["input_sha256"] = "x"
            else:
                bad["cases"].pop()
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                p.validate_summary(bad)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch CPU arithmetic check")
    def test_torch_reuses_output_and_preserves_all_tied_peaks(self):
        import torch
        import softmax_torch
        case = dict(rows=1, cols=3, mode=4, input=[-2e30, -2e30, -3e30])
        with torch.inference_mode():
            prepared = softmax_torch.prepare(case, "cpu")
            first = softmax_torch.compute(prepared)
            self.assertEqual(first.tolist(), [0.5, 0.5, 0., 1., 1., 0.])
            second = softmax_torch.compute(prepared)
            self.assertEqual(first.data_ptr(), second.data_ptr())


if __name__ == "__main__":
    unittest.main()
