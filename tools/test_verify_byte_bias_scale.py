"""Fabricated observations test rejection gates, not device correctness."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load("verify_byte_bias_scale")
legacy = load("test_verify_byte_flat_metric")


def inputs():
    reference, learning = legacy.inputs()
    reference["schema"] = "spiraltorch.resident_byte_bias_scale.torch_fixture.v1"
    report = {"schema": "spiraltorch.resident_byte_bias_scale.validation.v1", "passed": True,
              "learning": learning, "calibration": []}
    f32 = lambda value: struct.unpack("!f", struct.pack("!f", value))[0]
    for index, case in enumerate(reference["cases"]):
        blocks = index + 1
        case["windows"] = [[97, 98, 97, 0, 255], [120, 121, 120, 128, 122]]
        case["uncalibrated_parameters"] = copy.deepcopy(case["parameters"])
        old = [[.1, .1] for _ in range(blocks)]
        gains = [[f32(math.log(math.expm1(verify.softplus(.1) * scale))) for scale in (2., .5)]
                 for _ in range(blocks)]
        scales = [[verify.softplus(g) / verify.softplus(o) for g, o in zip(gs, os)]
                  for gs, os in zip(gains, old)]

        def scores(factors):
            return [[[f32(-factors[i][h] * (m + b + 1) * (q - k) ** 2) if k <= q else 0.
                      for b in range(2) for h in range(2) for q in range(4) for k in range(4)]
                     for i in range(blocks)] for m in range(2)]

        record = {"shape": [2, 2, 4, 4], "windows": [case["windows"], [[13, 19, 5, 2, 255], [1, 3, 15, 8, 254]]],
                  "valid_pairs_per_head": 40, "relative_tolerance": 1e-5,
                  "old_raw_gains": old, "raw_gains": gains,
                  "reference_scores": scores([[2., .5] for _ in range(blocks)]),
                  "candidate_scores": scores([[1., 1.] for _ in range(blocks)]), "fitted_scores": scores(scales)}
        for prefix in ("reference", "candidate", "fitted"):
            record[prefix + "_rms"] = verify.moments(record[prefix + "_scores"], blocks)
        for i in range(blocks):
            case["parameters"][6 + i]["values"] = gains[i]
        case["calibration"] = copy.deepcopy(record)
        record.update({"name": case["name"], "requested_scales": [[2., .5] for _ in range(blocks)],
                       "realized_relative_errors": [abs(f / r - 1.) for fs, rs in
                           zip(record["fitted_rms"], record["reference_rms"]) for f, r in zip(fs, rs)],
                       "no_updates_consumed": True, "targets_and_external_bias_excluded": True,
                       "ordinary_and_out_of_range_absent": True,
                       "positive_control": True})
        for key, descriptors in (("before_checkpoint_json", case["uncalibrated_parameters"]),
                                 ("calibrated_checkpoint_json", case["parameters"])):
            record[key] = json.dumps(verify.flat.expected_checkpoint(case, [p["values"] for p in descriptors], 0))
        report["calibration"].append(record)
        report["learning"]["checks"][index]["initial_checkpoint_json"] = record["calibrated_checkpoint_json"]
    return copy.deepcopy(reference), copy.deepcopy(report)


class BiasScaleVerifierTests(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(verify.verify(*inputs())["passed"])

    def reject(self, change):
        reference, report = inputs()
        change(reference, report)
        with self.assertRaises((ValueError, KeyError)):
            verify.verify(reference, report)

    def test_missing_and_false_controls(self):
        _, report = inputs()
        for key in report["calibration"][0]:
            self.reject(lambda r, a: a["calibration"][0].pop(key))
        for key in ("no_updates_consumed", "targets_and_external_bias_excluded", "positive_control",
                    "ordinary_and_out_of_range_absent"):
            for bad in (False, 1):
                self.reject(lambda r, a: a["calibration"][0].__setitem__(key, bad))

    def test_identity_coverage_and_gate(self):
        for key, bad in (("name", "other"), ("valid_pairs_per_head", True), ("valid_pairs_per_head", 39),
                         ("shape", [2, 2, 4, 5]), ("relative_tolerance", .01)):
            self.reject(lambda r, a: a["calibration"][0].__setitem__(key, bad))
        self.reject(lambda r, a: a["calibration"].pop())
        self.reject(lambda r, a: a["calibration"][0]["windows"].reverse())
        self.reject(lambda r, a: a["calibration"][0]["windows"][0][0].__setitem__(4, 1))
        for key in ("reference_scores", "fitted_scores", "raw_gains", "realized_relative_errors"):
            self.reject(lambda r, a: a["calibration"][0][key].pop())

    def test_reported_statistics_recomputed(self):
        for key in ("reference_rms", "candidate_rms", "fitted_rms", "requested_scales"):
            self.reject(lambda r, a: a["calibration"][0][key][0].__setitem__(0, 99.))
        self.reject(lambda r, a: a["calibration"][0]["realized_relative_errors"].__setitem__(0, 1.))

    def test_numeric_errors_including_masked_scores(self):
        for bad in (True, float("nan"), float("inf"), 1e300):
            self.reject(lambda r, a: a["calibration"][0]["candidate_scores"][0][0].__setitem__(1, bad))
            self.reject(lambda r, a: a["calibration"][0]["raw_gains"][0].__setitem__(0, bad))
            self.reject(lambda r, a: a["calibration"][0]["candidate_rms"][0].__setitem__(0, bad))

    def test_gain_fit_and_source(self):
        self.reject(lambda r, a: a["calibration"][0]["old_raw_gains"][0].__setitem__(0, .2))
        self.reject(lambda r, a: a["calibration"][0]["raw_gains"][0].__setitem__(0, .1))
        self.reject(lambda r, a: r["cases"][0]["parameters"][0]["values"].__setitem__(0, .2))

    def test_coordinated_statistic_forgery_still_hits_torch_oracle(self):
        def change(r, a):
            record = a["calibration"][0]
            # Row offsets do not change centered RMS or the fitted gain.
            # They must still agree with the frozen reference score observations.
            for prefix in ("reference", "candidate", "fitted"):
                for batch in record[prefix + "_scores"]:
                    batch[0] = [value + 100. for value in batch[0]]
                record[prefix + "_rms"] = verify.moments(record[prefix + "_scores"], 1)
        self.reject(change)

    def test_checkpoint_full_state_and_revision(self):
        for field in ("before_checkpoint_json", "calibrated_checkpoint_json"):
            for mutate in (lambda cp: cp.__setitem__("attempted_revision", "1"),
                           lambda cp: cp["model"]["geometry"].__setitem__("pair_metric", "poincare_squared.v1"),
                           lambda cp: cp["model"]["geometry"]["raw_gains"][0].__setitem__(0, .2),
                           lambda cp: cp["model"]["token"]["values"].__setitem__(0, .2),
                           lambda cp: cp["model"].pop("head")):
                def change(r, a):
                    record = a["calibration"][0]
                    cp = json.loads(record[field])
                    mutate(cp)
                    record[field] = json.dumps(cp)
                self.reject(change)

    def test_learning_and_resume_are_not_skipped(self):
        self.reject(lambda r, a: a["learning"]["checks"][0].pop("initial_checkpoint_json"))
        self.reject(lambda r, a: a["learning"]["checks"][0]["learning"]["trace"].pop())
        self.reject(lambda r, a: a["learning"]["checks"][0]["resume_trajectory"]["trace"][8]["parameters"][0].__setitem__(0, .1))
        self.reject(lambda r, a: a["learning"]["checks"][0]["parameter_gradients"][2].__setitem__(0, 0.))

    def test_one_ulp_fit_splice_without_changing_learner(self):
        def change(r, a):
            record = a["calibration"][0]
            gain = record["raw_gains"][0][0]
            old_bits = struct.unpack("!I", struct.pack("!f", gain))[0]
            one_ulp = struct.unpack("!f", struct.pack("!I", old_bits + 1))[0]
            record["raw_gains"][0][0] = one_ulp
            cp = json.loads(record["calibrated_checkpoint_json"])
            cp["model"]["geometry"]["raw_gains"][0][0] = one_ulp
            record["calibrated_checkpoint_json"] = json.dumps(cp)
        self.reject(change)


if __name__ == "__main__":
    unittest.main()
