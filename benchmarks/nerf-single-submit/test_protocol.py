"""Reuse the numerical admission grid without conflating route identities."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol
from test_contract import fixture


def source():
    report=fixture()
    report["schema"]=protocol.SCHEMA
    report["comparison"]=protocol.COMPARISON
    for case in report["cases"]:
        for interval in case["intervals"]:
            interval["route"]={"staged":"separate","direct":"single"}[interval["route"]]
    return report


class ProtocolTests(unittest.TestCase):
    def test_normalization_does_not_change_raw_reports(self):
        raw=source(); before=copy.deepcopy(raw)
        admitted=protocol.normalize(raw)
        self.assertEqual(raw,before)
        self.assertEqual(admitted["cases"][0]["intervals"][0]["route"],"staged")
        self.assertEqual(admitted["cases"][0]["ray_inputs"],raw["cases"][0]["ray_inputs"])

    def test_old_and_unknown_routes_are_not_accepted(self):
        for raw in [fixture(),{**source(),"comparison":"staged_vs_direct"}]:
            with self.assertRaises((ValueError,KeyError)): protocol.normalize(raw)
        for route in ["staged","direct","mps","unknown"]:
            raw=source(); raw["cases"][0]["intervals"][0]["route"]=route
            with self.assertRaises(ValueError): protocol.normalize(raw)

    def test_inherits_missing_duplicate_and_numeric_rejection(self):
        for kind in ["missing","duplicate","nan","bad_output"]:
            raw=source()
            if kind=="missing": raw["cases"].pop()
            elif kind=="duplicate": raw["cases"][0]=raw["cases"][1]
            elif kind=="nan": raw["cases"][0]["intervals"][0]["elapsed_ms"]=float("nan")
            else: raw["cases"][0]["last_outputs"][0][0]=99.
            with self.assertRaises(ValueError): protocol.normalize(raw)

    def test_summary_names_and_recomputation(self):
        result=protocol.analyze([source()]*3,[source()]*3,[fixture("torch")]*3)
        protocol.validate_summary(result)
        encoded=json.dumps(result)
        self.assertNotIn("native_staged",encoded)
        self.assertNotIn("browser_direct",encoded)
        self.assertEqual(result["descriptive_summary"]["native"]["geomean_separate_over_single"],1.)
        result["cases"][0]["summary"][0]["median_interval_ms"]["native_separate"]=99.
        with self.assertRaises(ValueError): protocol.validate_summary(result)

    def test_input_mismatch_is_not_hidden_by_normalization(self):
        browser=[source() for _ in range(3)]
        browser[2]["cases"][0]["ray_inputs"][0][0]=2.
        with self.assertRaises(ValueError): protocol.analyze([source()]*3,browser,[fixture("torch")]*3)


if __name__ == "__main__":
    unittest.main()
