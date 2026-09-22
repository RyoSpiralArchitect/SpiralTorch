from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consensus_protocol as p

TEST_KEYS = {(1, 31, 2), (1, 31, 4)}
TEST_CONTROL = hashlib.sha256(b"synthetic").hexdigest()


def fixture(family):
    torch = family == "torch"
    routes = ("cpu", "mps") if torch else p.VARIANTS
    report = dict(schema=p.TORCH if torch else p.SCHEMA, status="passed", warmup=3, blocks=9, bursts=[1, 4], cases=[])
    if torch:
        report.update(input_protocol=p.SCHEMA, devices=["cpu", "mps"], compiled=False, preallocated_outputs=True,
                      intra_op_threads=4, inter_op_threads=1, torch_version="synthetic")
    else:
        report.update(adapter="synthetic noncpu", control_shader="synthetic", domain_cases=12, domain_bitwise_equal=True,
                      routes=list(p.VARIANTS), ownership=dict(ordered_snapshot=True, empty_prefixes=True, preflight_errors=5,
                      unread_drop=True, source_drop=True, pending_map_cancellation=family == "browser"))
    if family == "browser":
        report.update(page_errors=[], console_messages=[], browser_adapter_probe=dict(is_fallback_adapter=False),
                      browser_version="synthetic", asset_sha256={"fixture": "0" * 64},
                      consensus_state_artifacts=dict(path="synthetic.cases.jsonl", rows=12, bytes=1, sha256="0" * 64))
    for r, c, n in sorted(TEST_KEYS):
        case = dict(rows=r, cols=c, count=n, input=p.inputs(r, c))
        reference = p.oracle(case["input"], r, c, n)
        case.update(reference=reference, last_outputs=[reference[:] for _ in routes], intervals=[])
        for b in range(9):
            for burst in (1, 4):
                for d in routes:
                    case["intervals"].append(dict(block=b, burst=burst, route=d, order=p.order(case, b+3, torch),
                        elapsed_ms=1.+routes.index(d)/10., max_abs_error=0., max_scaled_error=0.))
        report["cases"].append(case)
    return report


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(p, "KEYS", TEST_KEYS).start()
        patch.object(p, "CONTROL_TOKENS", TEST_CONTROL).start()

    def test_complete_rounds_all_factors_and_negative_cells_retained(self):
        reports = [[fixture(f) for _ in range(3)] for f in ("native", "browser", "torch")]
        result = p.analyze(*reports)
        p.validate_summary(result)
        self.assertEqual(sum(len(c["intervals"]) for c in result["cases"]), 1080)
        self.assertEqual(result["descriptive_summary"]["native"]["4"]["combined"]["cells_over_1"], 0)
        with self.assertRaises(ValueError):
            p.analyze(reports[0][:-1], *reports[1:])

    def test_source_and_ownership_are_admission_gates(self):
        for family in ("native", "browser", "torch"):
            report = fixture(family)
            p.admit(report, family)
            changed = deepcopy(report)
            if family == "torch":
                changed["compiled"] = True
            else:
                changed["control_shader"] += " changed"
            with self.assertRaises(ValueError):
                p.admit(changed, family)
        report = fixture("browser")
        report["ownership"]["pending_map_cancellation"] = False
        with self.assertRaises(ValueError):
            p.admit(report, "browser")
        report = fixture("browser")
        report["consensus_state_artifacts"]["rows"] = 11
        with self.assertRaises(ValueError):
            p.admit(report, "browser")

    def test_input_output_and_mask_order_not_just_finiteness(self):
        for field in ("input", "reference", "last_outputs"):
            report = fixture("native")
            values = report["cases"][0][field]
            if field == "last_outputs":
                values = values[0]
            values[0] += 1.
            with self.assertRaises(ValueError):
                p.admit(report, "native")
        probabilities = p.oracle([0.] * 31, 1, 31, 2)
        self.assertEqual(probabilities[31:], [1.] * 31)

    def test_missing_duplicate_zero_time_and_wrong_order_rejected(self):
        for mutation in (lambda v: v.pop(), lambda v: v.append(v[0]),
                         lambda v: v[0].update(elapsed_ms=0.), lambda v: v[0].update(order=[0, 1]),
                         lambda v: v[0].update(max_scaled_error=1.01)):
            report = fixture("native")
            mutation(report["cases"][0]["intervals"])
            with self.assertRaises(ValueError):
                p.admit(report, "native")

    def test_summary_cannot_hide_slow_cells(self):
        result = p.analyze(*[[fixture(f) for _ in range(3)] for f in ("native", "browser", "torch")])
        result["cases"][0]["summary"][0]["median_interval_ms"]["cpu"] *= 2
        with self.assertRaises(ValueError):
            p.validate_summary(result)


if __name__ == "__main__":
    unittest.main()
