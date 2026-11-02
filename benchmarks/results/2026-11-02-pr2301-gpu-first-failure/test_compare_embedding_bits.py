"""Synthetic records exercise the comparator, not Metal behavior."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_embedding_bits import FINAL, SELECTION, STAGES, compare


def pair():
    case = {"name": "geometry", "training": [{"revision": 1}, {"revision": 2}],
            "final_parameters": [[0.] * 512]}
    report = {"schema": "spiraltorch.byte_corpus.result.v1", "adapter": "test-only", "engine": "spiraltorch",
              "request_sha256": SELECTION["request_sha256"], "cases": [{**case, "name": "plain"}, case]}
    return ({"schema": "spiraltorch.byte_embedding_update.observation.v1", "selection": SELECTION.copy(),
             "status": "completed", "capture_status": "views_retained", "release_ready": False, "report": report},
            {"schema": "spiraltorch.byte_embedding_update.bits.v1", "selection": SELECTION.copy(),
             "attempted_revisions": [0, 1], "f32_bits": dict.fromkeys(STAGES, 0)})


class Comparison(unittest.TestCase):
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
