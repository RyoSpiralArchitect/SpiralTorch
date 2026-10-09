import copy
import importlib.util
import math
import unittest

import byte_corpus_study as study
import byte_corpus_calibration as calibration
from test_byte_corpus_study import metric_fixture


def fixture():
    raw, reference, actual = metric_fixture()
    request = study.decode_json(raw)
    request["schema"] = study.REQUEST_V4
    request["bias_calibration"] = dict(source_request_sha256=study.digest(raw), train_batch_indices=[0, 1], relative_tolerance=1e-5)
    for c in request["cases"]:
        c["bias_initialization"] = study.ORIGINAL
    for original in request["cases"][3:5]:
        extra = copy.deepcopy(original)
        extra["name"] += "_rms_matched"
        extra["bias_initialization"] = study.MATCHED
        extra["parameters"][6]["values"] = [.25]
        request["cases"].append(extra)
    raw = study.encoded(request)
    for report in (reference, actual):
        report["cases"] += copy.deepcopy(report["cases"][3:5])
        report.update(schema="spiraltorch.byte_corpus.result.v4", request_sha256=study.digest(raw), bias_calibration=request["bias_calibration"])
        for i, (c, spec) in enumerate(zip(report["cases"], request["cases"])):
            c["name"] = spec["name"]
            c["bias_initialization"] = spec["bias_initialization"]
            if i >= 5:
                c["final_parameters"][6] = [study.f32(.25) if study.frozen(spec) else .26]
                loss = 3.6 if i == 5 else 3.8
                c["validation"][-1].update(batch_losses=[loss], mean_ce=loss, bits_per_byte=loss / math.log(2))
    reference["calibration_verification"] = dict(passed=True, no_fitting=True, no_updates_consumed=True,
        cases=[dict(seed=11, valid_pairs_per_head=6, reference_rms=[[2.]], fitted_rms=[[2.]], realized_relative_errors=[[0.]])])
    return raw, reference, actual


class Calibration(unittest.TestCase):
    def test_missing_failed_or_incomplete_torch_qualification_rejects(self):
        for mutation in ("missing", "passed", "no_fitting", "no_updates_consumed", "seed", "count", "head", "zero", "gate", "error"):
            raw, reference, actual = fixture()
            record = reference["calibration_verification"]
            if mutation == "missing":
                reference.pop("calibration_verification")
            elif mutation in ("passed", "no_fitting", "no_updates_consumed"):
                record[mutation] = False
            elif mutation == "seed":
                record["cases"][0]["seed"] = 23
            elif mutation == "count":
                record["cases"][0]["valid_pairs_per_head"] = 7
            elif mutation == "head":
                record["cases"][0]["reference_rms"] = [[]]
            elif mutation == "zero":
                record["cases"][0]["reference_rms"] = [[False]]
            elif mutation == "gate":
                record["cases"][0]["fitted_rms"] = [[2.1]]
            else:
                record["cases"][0]["realized_relative_errors"] = [[1e-7]]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.compare(raw, reference, actual)

    def test_calibrated_contrasts_are_selected_by_treatment_not_order(self):
        raw, reference, actual = fixture()
        expected = study.compare(raw, reference, actual)["paired_deltas"]
        c = expected[0]
        self.assertAlmostEqual(c["flat_minus_poincare_bpb"], -.2 / math.log(2))
        self.assertAlmostEqual(c["calibrated_flat_minus_flat_bpb"], -.1 / math.log(2))
        self.assertAlmostEqual(c["calibration_by_training_interaction_bpb"], .1 / math.log(2))
        self.assertTrue(all(c["same_sign_as_torch"].values()))
        request = study.decode_json(raw)
        for value in (request, reference, actual):
            value["cases"].reverse()
        raw = study.encoded(request)
        for report in (reference, actual):
            report["request_sha256"] = study.digest(raw)
        self.assertEqual(study.compare(raw, reference, actual)["paired_deltas"], expected)

    def test_only_gain_difference_between_treatments_is_permitted(self):
        for i in range(1, 7):
            for slot in (0, 2, 3, 4, 5, 6):
                request = study.decode_json(fixture()[0])
                request["cases"][i]["parameters"][slot]["values"][0] += .01
                with self.subTest(case=i, slot=slot), self.assertRaises(ValueError):
                    study.request_version(request)

    def test_null_legacy_missing_and_relaxed_calibration_metadata_reject(self):
        for mutation in ("missing", "null", "gate", "duplicate", "heldout", "index", "unknown", "sha", "init", "old"):
            request = study.decode_json(fixture()[0])
            spec = request["bias_calibration"]
            if mutation == "missing":
                request.pop("bias_calibration")
            elif mutation == "null":
                request["bias_calibration"] = None
            elif mutation == "gate":
                spec["relative_tolerance"] = .001
            elif mutation == "duplicate":
                spec["train_batch_indices"] = [0, 0]
            elif mutation == "heldout":
                spec["validation_batch_indices"] = [0]
            elif mutation == "index":
                spec["train_batch_indices"] = [2]
            elif mutation == "unknown":
                request["cases"][5]["bias_initialization"] = "automatic"
            elif mutation == "sha":
                spec["source_request_sha256"] = "A" * 64
            elif mutation == "init":
                request["cases"][5].pop("bias_initialization")
            else:
                request["schema"] = study.REQUEST_V3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.request_version(request)

    def test_case_capacity_stays_three_seeds(self):
        request = study.decode_json(fixture()[0])
        group = request["cases"]
        request["cases"] = []
        for seed in (11, 23, 37, 41):
            copy_group = copy.deepcopy(group)
            for case in copy_group:
                case.update(seed=seed, name=f"s{seed}_{case['name']}")
            request["cases"] += copy_group
            if seed != 41:
                self.assertEqual(study.request_version(request), 4)
        with self.assertRaises(ValueError):
            study.request_version(request)

    def test_report_relabel_metadata_and_fitted_freeze_drift_reject(self):
        for field in ("init", "metadata", "boolean_metadata", "freeze"):
            raw, reference, actual = fixture()
            if field == "init":
                actual["cases"][5]["bias_initialization"] = study.ORIGINAL
            elif field == "metadata":
                actual["bias_calibration"] = dict(actual["bias_calibration"], train_batch_indices=[1, 0])
            elif field == "boolean_metadata":
                actual["bias_calibration"] = dict(actual["bias_calibration"], train_batch_indices=[False, True])
            else:
                actual["cases"][6]["final_parameters"][6] = [.2500001]
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.compare(raw, reference, actual)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU PyTorch")
    def test_independent_statistic_never_fits_or_reads_heldout(self):
        import torch
        import torch.nn.functional as functional
        request = study.decode_json(fixture()[0])
        request["validation_documents"] = object()
        for c in request["cases"][5:]:
            c["parameters"][6]["values"] = request["cases"][1]["parameters"][6]["values"].copy()
        for seed in (23, 37):
            group = copy.deepcopy(request["cases"][:7])
            for case in group:
                case["seed"] = seed
                case["name"] = f"s{seed}_{case['name']}"
            request["cases"].extend(group)
        before = copy.deepcopy(request["cases"])
        def wave(drive, decay, phase, initial, curvature):
            return drive * .1, initial
        def flat_like(coordinates, gain, curvature):
            distance = (coordinates[:, :, None, :] - coordinates[:, None, :, :]).square().sum(-1)
            return (-functional.softplus(gain)[None, :, None, None] * (4 * distance[:, None])).tril()
        result = calibration.verify_initial_rms(request, wave, flat_like)
        self.assertTrue(result["no_fitting"])
        self.assertEqual(request["cases"], before)
        self.assertEqual(result["cases"][0]["valid_pairs_per_head"], 6)
        self.assertEqual([c["seed"] for c in result["cases"]], [11, 23, 37])
        request["cases"][5]["parameters"][6]["values"][0] += .1
        with self.assertRaisesRegex(ValueError, "RMS gate"):
            calibration.verify_initial_rms(request, wave, flat_like)


if __name__ == "__main__":
    unittest.main()
