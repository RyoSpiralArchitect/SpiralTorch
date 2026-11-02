"""Synthetic records exercise the comparator, not Metal behavior."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_embedding_bits import FINAL, SELECTION, STAGES, compare, compare_reports, report_bytes


def pair():
    case = {"name": "geometry", "training": [{"revision": 1}, {"revision": 2}],
            "final_parameters": [[0.] * 512]}
    report = {"schema": "spiraltorch.byte_corpus.result.v1", "adapter": "test-only", "engine": "spiraltorch",
              "request_sha256": SELECTION["request_sha256"], "cases": [{**copy.deepcopy(case), "name": "plain"}, case]}
    return ({"schema": "spiraltorch.byte_embedding_update.observation.v1", "selection": SELECTION.copy(),
             "status": "completed", "capture_status": "views_retained", "release_ready": False, "report": report},
            {"schema": "spiraltorch.byte_embedding_update.bits.v1", "selection": SELECTION.copy(),
             "attempted_revisions": [0, 1], "f32_bits": dict.fromkeys(STAGES, 0)})


class Comparison(unittest.TestCase):
    def test_report_keeps_exact_bytes_and_does_not_depend_on_bit_read_success(self):
        outcome, _ = pair()
        report = json.dumps(outcome.pop("report"), indent=1).encode()
        outcome["capture_status"] = "error"
        outcome["note"] = {"report": "not the top-level value"}
        prefix = json.dumps(outcome).encode()[:-1] + b', "\\u0072eport" : '
        first = prefix + report + b" }"
        self.assertEqual(report_bytes(first), report)
        workers = compare_reports(first, prefix + report.replace(b"0.0", b"0e0") + b"}", first)["workers"]
        self.assertFalse(workers[0]["raw_equal"])
        self.assertEqual(workers[0]["groups"]["other"]["examples"][0]["kind"], "serialization_only")
        self.assertTrue(workers[1]["raw_equal"])
        with self.assertRaises(ValueError):
            report_bytes(prefix + report + b',"report":{}}')

    def test_equal_coordinate_does_not_mask_a_different_parameter(self):
        serial, w0, w1 = pair(), pair(), pair()
        w0[0]["report"]["cases"][1]["final_parameters"][0][10] = 1.
        self.assertFalse(compare(serial, w0, w1)["workers"][0]["differences"])
        raw = [json.dumps(p[0]).encode() for p in (serial, w0, w1)]
        result = compare_reports(*raw)["workers"]
        self.assertEqual(result[0]["groups"]["final_parameters"]["examples"][0]["path"],
                         "$.cases[1].final_parameters[0][10]")
        self.assertTrue(result[1]["raw_equal"])

    def test_first_boundary_and_both_workers_remain_visible(self):
        for stage in (*STAGES, FINAL):
            serial, w0, w1 = pair(), pair(), pair()
            if stage != FINAL:
                w0[1]["f32_bits"][stage] = 0x80000000
            w0[0]["report"]["cases"][1]["final_parameters"][0][4] = -0.
            result = compare(serial, w0, w1)["workers"]
            self.assertEqual(result[0]["first_observed_difference"], stage)
            self.assertEqual(result[0]["differences"][-1]["stage"], FINAL)
            self.assertEqual(result[1]["differences"], [])

    def test_fail_closed_instead_of_reporting_agreement(self):
        for side, keys, bad in [
            (0, ["capture_status"], "error"), (0, ["status"], "error"),
            (0, ["report", "adapter"], "another"), (1, ["attempted_revisions"], [False, True]),
            (0, ["report", "cases", 1, "training", 0, "revision"], True),
            (1, ["selection", "feature"], False), (1, ["selection", "case"], "plain"),
            (1, ["f32_bits", STAGES[0]], -1), (1, ["f32_bits", STAGES[1]], 0x7fc00000),
        ]:
            actual = copy.deepcopy(pair())
            field = actual[side]
            for key in keys[:-1]:
                field = field[key]
            field[keys[-1]] = bad
            with self.subTest(keys=keys, bad=bad), self.assertRaises(ValueError):
                compare(pair(), pair(), actual)
